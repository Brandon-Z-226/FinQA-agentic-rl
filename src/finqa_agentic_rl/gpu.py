"""Explicit Linux GPU preparation commands; no training on import."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def validate_training_config(config: DictConfig) -> None:
    """Catch the FSDP LoRA regression and accidental multi-GPU profiles."""
    get = lambda key: OmegaConf.select(config, key)  # noqa: E731
    if get("actor_rollout_ref.actor.strategy") != "fsdp":
        raise ValueError("The A800 profile requires FSDP.")
    rank = get("actor_rollout_ref.model.lora_rank")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("FSDP LoRA is OFF: actor_rollout_ref.model.lora_rank must be positive.")
    if rank != get("actor_rollout_ref.model.lora.rank") or get("actor_rollout_ref.model.lora_alpha") != get("actor_rollout_ref.model.lora.alpha"):
        raise ValueError("Flat FSDP and nested LoRA rank/alpha must agree.")
    for key in (
        "trainer.nnodes",
        "trainer.n_gpus_per_node",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.data_parallel_size",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size",
    ):
        if get(key) != 1:
            raise ValueError(f"Single GPU profile requires {key}=1.")
    if get("rllm.async_training.enable") or not get("actor_rollout_ref.hybrid_engine"):
        raise ValueError("Use colocated actor/rollout with async_training.enable=false.")
    if get("rllm.rollout.n") < 2:
        raise ValueError("GRPO requires at least two rollouts per question.")
    if get("data.max_prompt_length") + get("data.max_response_length") > get("actor_rollout_ref.rollout.max_model_len"):
        raise ValueError("Prompt plus response budget exceeds max_model_len.")
    for split in ("train", "val"):
        limit = get(f"finqa.{split}_limit")
        if limit is not None and (not isinstance(limit, int) or limit < get(f"data.{split}_batch_size")):
            raise ValueError(f"finqa.{split}_limit must cover a full {split} batch.")
        sample_size = get(f"finqa.{split}_sample_size")
        sample_seed = get(f"finqa.{split}_sample_seed")
        if limit is not None and sample_size is not None:
            raise ValueError(f"Use either finqa.{split}_limit or finqa.{split}_sample_size, not both.")
        if sample_size is not None:
            if not isinstance(sample_size, int) or sample_size < get(f"data.{split}_batch_size"):
                raise ValueError(f"finqa.{split}_sample_size must cover a full {split} batch.")
            if not isinstance(sample_seed, int):
                raise ValueError(f"finqa.{split}_sample_seed must be an integer.")
    if get("finqa.save_final_checkpoint"):
        final_step = get("finqa.expected_checkpoint_step")
        if not isinstance(final_step, int) or final_step <= 0:
            raise ValueError("finqa.expected_checkpoint_step must be positive when final checkpoint saving is enabled.")
        if get("rllm.trainer.total_batches") != final_step or get("trainer.total_training_steps") != final_step:
            raise ValueError("Final checkpoint step must match both configured training-step limits.")
        if get("trainer.test_freq") <= final_step:
            raise ValueError("Final checkpoint saving requires a positive test_freq above the total steps for one final validation only.")
        fresh = bool(get("finqa.require_empty_checkpoint_dir"))
        resume_mode = get("trainer.resume_mode")
        if fresh and resume_mode != "disable":
            raise ValueError("A fresh full baseline requires resume_mode=disable.")
        if not fresh and resume_mode != "auto":
            raise ValueError("A full-baseline recovery requires resume_mode=auto.")


def require_judge() -> None:
    from .finqa_eval import _JUDGE_CLIENT

    if _JUDGE_CLIENT is None:
        raise RuntimeError("Configure a valid judge key in .env before GPU evaluation/training.")


def require_disk(path: Path, minimum_gib: float) -> float:
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum_gib:
        raise RuntimeError(f"{path}: only {free:.1f} GiB free; need {minimum_gib:g} GiB for checkpoint/headroom. Expand the data disk.")
    return round(free, 2)


def check_runtime() -> dict:
    """Run real CUDA/FlashAttention kernels, without loading model weights."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("GPU runtime requires Linux x86_64.")
    import torch
    import verl  # noqa: F401
    import vllm  # noqa: F401
    from flash_attn import flash_attn_func

    expected = {"torch": "2.11.0", "vllm": "0.22.1", "verl": "0.8.0", "flash-attn": "2.8.3", "tensordict": "0.10.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if any(versions[name].split("+")[0] != version for name, version in expected.items()):
        raise RuntimeError(f"Unexpected CUDA dependency versions: {versions}")
    # Exercise the actual conversion before paying for model/rollout startup.
    from verl import DataProto

    probe = torch.tensor([[1, 2]], dtype=torch.int64)
    converted = DataProto.from_dict(tensors={"input_ids": probe}).to_tensordict()
    if not torch.equal(converted["input_ids"], probe):
        raise RuntimeError("DataProto to TensorDict conversion failed")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expose exactly one BF16-capable GPU via CUDA_VISIBLE_DEVICES.")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 75 * 2**30:
        raise RuntimeError("This profile targets an 80 GB GPU; available device capacity is smaller.")
    x = torch.randn(32, 32, dtype=torch.bfloat16, device="cuda")
    (x @ x).sum().item()
    qkv = [torch.randn(1, 128, 8, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True) for _ in range(3)]
    flash_attn_func(*qkv, causal=True).float().sum().backward()
    torch.cuda.synchronize()
    return {"versions": versions, "torch_cuda": torch.version.cuda, "gpu": properties.name, "vram_gib": round(properties.total_memory / 2**30, 2), "cuda_bf16_and_flash_attention_backward": "passed"}


def model_path(root: Path, project: Path) -> Path:
    if os.environ.get("FINQA_MODEL_PATH"):
        return Path(os.environ["FINQA_MODEL_PATH"]).expanduser().resolve()
    spec = json.loads((project / "environments/cuda/model.json").read_text())
    cache = Path(os.environ.get("HF_HUB_CACHE", str(root / "huggingface/hub")))
    return cache / ("models--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]


def check_model(path: Path) -> None:
    if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise RuntimeError(f"Model weights missing in {path}; run download first.")


def check_lora(path: Path) -> dict:
    """Check PEFT target selection on a meta model; no full weights allocated."""
    from accelerate import init_empty_weights
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(config)
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=32, target_modules="all-linear", bias="none"))
    trainable = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    total = sum(parameter.numel() for parameter in model.parameters())
    count = sum(trainable.values())
    if not trainable or any("lora_" not in name for name in trainable) or count / total > 0.05:
        raise RuntimeError("LoRA parameter inspection failed: base parameters may be trainable.")
    return {
        "trainable_parameters": count,
        "total_parameters": total,
        "trainable_fraction": count / total,
        "only_lora_trainable": True,
        "method": "meta model; actual FSDP worker still requires smoke verification",
    }


def verify_lora_checkpoint(path: Path, rank: int, alpha: int) -> None:
    metadata = path / "lora_train_meta.json"
    if not metadata.is_file():
        raise RuntimeError(f"Smoke checkpoint lacks LoRA metadata: {metadata}")
    saved = json.loads(metadata.read_text())
    if saved.get("r") != rank or saved.get("lora_alpha") != alpha:
        raise RuntimeError("Smoke checkpoint LoRA rank/alpha do not match the training configuration.")
    print(f"LoRA checkpoint verified: {metadata}")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def require_bounded_full_launcher(action: str, audit_dir: str | None) -> None:
    if action == "baseline-full" and not audit_dir:
        raise RuntimeError("baseline-full must be launched through scripts/run_smoke_audit.py so metrics and the final checkpoint hook are active")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "prepare", "check", "lora-check", "config", "smoke", "baseline-small", "baseline-full", "train", "serve", "eval"])
    args, extra = parser.parse_known_args()
    project = Path(os.environ.get("FINQA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    root = Path(os.environ["FINQA_GPU_ROOT"])
    path = model_path(root, project)
    os.environ["FINQA_MODEL_PATH"] = str(path)
    spec = json.loads((project / "environments/cuda/model.json").read_text())
    artifacts = root / "checks"
    artifacts.mkdir(parents=True, exist_ok=True)

    if args.action == "download":
        from huggingface_hub import snapshot_download

        if extra:
            parser.error("download accepts no extra arguments; edit model.json to change the pinned model.")
        # Reuse the HF snapshot directly, avoiding a second 8 GB model copy.
        downloaded = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"])
        check_model(Path(downloaded))
        (artifacts / "model.json").write_text(json.dumps({**spec, "downloaded_path": downloaded}, indent=2) + "\n")
        print(downloaded)
    elif args.action == "prepare":
        run([sys.executable, "-m", "finqa_agentic_rl.prepare_finqa_data", *extra])
    elif args.action in {"check", "lora-check"}:
        if extra:
            parser.error(f"{args.action} accepts no extra arguments.")
        if args.action == "check":
            report = check_runtime()
            report["free_disk_gib"] = require_disk(root, 24)
        else:
            check_model(path)
            report = check_lora(path)
        (artifacts / f"{args.action}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    elif args.action in {"config", "smoke", "baseline-small", "baseline-full", "train"}:
        require_bounded_full_launcher(args.action, os.environ.get("FINQA_SMOKE_AUDIT_DIR"))
        profile = {
            "smoke": "a800_smoke",
            "baseline-small": "a800_baseline_4step",
            "baseline-full": "a800_baseline_full",
        }.get(args.action, "a800_80gb")
        command = [sys.executable, "-m", "finqa_agentic_rl.train", "--config-path", str(project / "configs"), "--config-name", profile]
        if args.action == "config":
            os.environ["FINQA_SKIP_PRELOAD"] = "1"
            command += ["--cfg", "job", "--resolve"]
        run([*command, *extra])
    elif args.action == "serve":
        check_model(path)
        run(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(path),
                "--served-model-name",
                "finqa-4b",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "8",
                "--gpu-memory-utilization",
                "0.5",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--enforce-eager",
                *extra,
            ]
        )
    elif args.action == "eval":
        require_judge()
        run(
            [
                str(Path(sys.executable).with_name("rllm")),
                "eval",
                "finqa",
                "--agent",
                "finqa",
                "--evaluator",
                "finqa",
                "--model",
                "finqa-4b",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--split",
                "val",
                "--max-examples",
                "2",
                "--temperature",
                "0.6",
                "--max-tokens",
                "4096",
                *extra,
            ]
        )


if __name__ == "__main__":
    main()
