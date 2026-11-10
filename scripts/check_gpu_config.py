"""Compose real verl configs without installing the GPU backend on macOS.

With --fetch, download the official verl 0.8.0 wheel, verify PyPI SHA256,
and extract only its configuration files. No package is installed or imported.
Subsequent runs are offline. This checks YAML and the pinned rLLM sync function,
not verl dataclass instantiation, CUDA kernels, or a training step.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from finqa_agentic_rl.gpu import validate_training_config

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/gpu-static"


def fetch_configs() -> None:
    with urllib.request.urlopen("https://pypi.org/pypi/verl/0.8.0/json", timeout=60) as response:
        metadata = json.load(response)
    artifact = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
    with urllib.request.urlopen(artifact["url"], timeout=60) as response:
        wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != artifact["digests"]["sha256"]:
        raise RuntimeError("Official verl wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in archive.namelist():
            if name.startswith("verl/trainer/config/") and name.endswith((".yaml", ".py")) and ".." not in Path(name).parts:
                target = CACHE / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    (CACHE / "source.json").write_text(json.dumps({"url": artifact["url"], "sha256": artifact["digests"]["sha256"]}, indent=2) + "\n")


def load_sync_config():
    # Execute only configuration functions from the installed, pinned source.
    # Importing the backend module itself would require Ray and CUDA packages.
    spec = importlib.util.find_spec("rllm")
    rllm = Path(next(iter(spec.submodule_search_locations)))
    namespace = {"OmegaConf": OmegaConf, "DictConfig": DictConfig, "logger": logging.getLogger(__name__)}
    selections = {
        "trainer/algorithms/config.py": {"_explicit_override_keys", "_plain", "sync_shared_keys"},
        "trainer/verl/utils.py": {"_SHARED_KEYS", "_TOTAL_TRAINING_STEPS_KEY", "sync_config"},
    }
    for relative, names in selections.items():
        path = rllm / relative
        tree = ast.parse(path.read_text())
        selected = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in names) or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names)
        ]
        selected_names = {node.name if isinstance(node, ast.FunctionDef) else node.target.id for node in selected}
        if selected_names != names:
            raise RuntimeError(f"Pinned rLLM config implementation changed: {relative}")
        module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *selected], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["sync_config"]


def check_validation_schedule_source() -> None:
    spec = importlib.util.find_spec("rllm")
    rllm = Path(next(iter(spec.submodule_search_locations)))
    source = (rllm / "trainer/unified_trainer.py").read_text()
    periodic = "trainer_state.global_step % self.rllm_config.trainer.test_freq == 0"
    final = "if self.rllm_config.trainer.test_freq > 0:\n            await self._validate_async(trainer_state)"
    full_validation = "StatefulTaskDataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)"
    if periodic not in source or final not in source or full_validation not in source:
        raise RuntimeError("Pinned rLLM validation scheduling changed; re-audit pre/periodic/final validation behavior.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--verl-config-dir", type=Path, default=CACHE / "verl/trainer/config")
    args = parser.parse_args()
    if args.fetch:
        fetch_configs()
    if not (args.verl_config_dir / "ppo_trainer.yaml").is_file():
        parser.error("Missing verl YAML; run once with --fetch, or supply --verl-config-dir.")
    # Fake paths are sufficient for config composition; do not inspect .env.
    os.environ["FINQA_GPU_ROOT"] = str(CACHE / "dry-run")
    os.environ["FINQA_MODEL_PATH"] = str(CACHE / "dry-run/model")
    sync = load_sync_config()
    check_validation_schedule_source()
    CACHE.mkdir(parents=True, exist_ok=True)
    reports = []
    expected_steps = {"a800_80gb": None, "a800_smoke": 1, "a800_baseline_4step": 4, "a800_baseline_full": 251}
    for profile, total_steps in expected_steps.items():
        search = f"hydra.searchpath=[pkg://rllm.trainer.config,file://{args.verl_config_dir.resolve()}]"
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            config = compose(config_name=profile, overrides=[search])
        sync(config, hydra_overrides=[])
        OmegaConf.resolve(config)
        validate_training_config(config)
        assert config.rllm.rollout.n == config.actor_rollout_ref.rollout.n == 4
        assert config.rllm.rollout.train.temperature == config.actor_rollout_ref.rollout.temperature == 0.6
        assert config.rllm.trainer.total_epochs == config.trainer.total_epochs == 1
        assert config.rllm.algorithm.eps_clip_high == config.actor_rollout_ref.actor.clip_ratio_high == 0.28
        assert config.trainer.total_training_steps == total_steps
        if profile == "a800_baseline_4step":
            assert config.rllm.data.train_batch_size == 16
            assert config.rllm.trainer.total_batches == 4
            assert config.finqa.train_sample_size == 64
            assert config.finqa.val_sample_size == 32
        if profile == "a800_baseline_full":
            assert config.rllm.data.train_batch_size == 16
            assert config.rllm.trainer.total_batches == 251
            assert config.rllm.trainer.val_before_train is True
            assert config.rllm.trainer.test_freq == 1000
            assert config.rllm.trainer.save_freq == 25
            assert config.trainer.max_actor_ckpt_to_keep == 2
            assert config.trainer.resume_mode == "disable"
            assert config.finqa.expected_train_size == 4030
            assert config.finqa.expected_val_size == 522
            assert config.finqa.save_final_checkpoint is True
        assert config.rllm.data.val_batch_size == config.data.val_batch_size
        OmegaConf.save(config, CACHE / f"{profile}.resolved.yaml", resolve=True)
        reports.append(
            {
                "profile": profile,
                "lora_rank": config.actor_rollout_ref.model.lora_rank,
                "group_size": config.rllm.rollout.n,
                "train_batch": config.rllm.data.train_batch_size,
                "epochs": config.trainer.total_epochs,
                "total_training_steps": config.trainer.total_training_steps,
            }
        )
    print(
        json.dumps(
            {
                "hydra_and_pinned_sync_config": "passed",
                "pinned_validation_schedule": "pre-validation when enabled; periodic modulo test_freq; final validation when test_freq > 0",
                "profiles": reports,
                "GPU_execution": "not tested",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
