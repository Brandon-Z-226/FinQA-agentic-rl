# Derived from rLLM cookbooks/finqa at 4e828c81680e5a85b638ceedeaf09d1e90109ca8.
# Apache-2.0; see licenses/rllm-LICENSE and THIRD_PARTY_NOTICES.md.
# Modified: package imports and project integration; see docs/migration.md.
"""Train the finqa agent using the Python API.

Usage (from this project root, after selecting a training environment)::

    python -m finqa_agentic_rl.train rllm/backend=tinker

    # Hydra overrides:
    python -m finqa_agentic_rl.train rllm/backend=tinker model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import random
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry

from .finqa_eval import finqa_evaluator
from .finqa_flow import finqa_flow


def _fixed_sample(dataset, size: int | None, seed: int | None, split: str):
    if size is None:
        return dataset, None
    if not isinstance(size, int) or size <= 0 or size > len(dataset):
        raise ValueError(f"finqa.{split}_sample_size must be between 1 and {len(dataset)}")
    if not isinstance(seed, int):
        raise ValueError(f"finqa.{split}_sample_seed must be an integer")
    indices = random.Random(seed).sample(range(len(dataset)), size)
    return dataset.select(indices), indices


def _validate_dataset_size(dataset, expected: int | None, split: str) -> None:
    if expected is not None and len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} FinQA {split} rows, found {len(dataset)}")


def _require_empty_checkpoint_dir(config: DictConfig) -> None:
    if not config.get("finqa", {}).get("require_empty_checkpoint_dir"):
        return
    checkpoint_dir = Path(config.trainer.default_local_dir)
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise RuntimeError(f"Fresh full baseline requires an empty checkpoint directory: {checkpoint_dir}")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    if config.get("finqa", {}).get("resource_profile") == "a800_80gb":
        import json
        import os
        from pathlib import Path

        from omegaconf import OmegaConf
        from rllm.trainer.verl.utils import sync_config

        from .gpu import check_model, check_runtime, require_disk, require_judge, validate_training_config

        if os.environ.get("FINQA_SKIP_PRELOAD"):
            raise RuntimeError("Unset FINQA_SKIP_PRELOAD before real training; FinQA needs its SQL tables.")
        # Check effective values after the SAME synchronization used by rLLM.
        sync_config(config, hydra_overrides=list(HydraConfig.get().overrides.task))
        validate_training_config(config)
        require_judge()
        check_model(Path(config.actor_rollout_ref.model.path))
        require_disk(Path(os.environ["FINQA_GPU_ROOT"]), config.finqa.min_free_disk_gib)
        _require_empty_checkpoint_dir(config)
        report = check_runtime()
        output = Path(HydraConfig.get().runtime.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "gpu-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        OmegaConf.save(config, output / "effective-config.yaml", resolve=True)

    # Backend imports are needed only when actually starting training.
    from rllm.trainer import AgentTrainer

    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("FinQA dataset not found. Run: finqa-prepare")

    limits = config.get("finqa", {})
    _validate_dataset_size(train_dataset, limits.get("expected_train_size"), "train")
    _validate_dataset_size(val_dataset, limits.get("expected_val_size"), "val")
    if limits.get("train_limit") is not None and limits.get("train_sample_size") is not None:
        raise ValueError("Use either finqa.train_limit or finqa.train_sample_size, not both")
    if limits.get("val_limit") is not None and limits.get("val_sample_size") is not None:
        raise ValueError("Use either finqa.val_limit or finqa.val_sample_size, not both")
    selection = {}
    if limits.get("train_limit") is not None:
        train_dataset = train_dataset.select(range(limits["train_limit"]))
        selection["train_indices"] = list(range(limits["train_limit"]))
    else:
        train_dataset, selection["train_indices"] = _fixed_sample(train_dataset, limits.get("train_sample_size"), limits.get("train_sample_seed"), "train")
    if limits.get("val_limit") is not None:
        val_dataset = val_dataset.select(range(limits["val_limit"]))
        selection["val_indices"] = list(range(limits["val_limit"]))
    else:
        val_dataset, selection["val_indices"] = _fixed_sample(val_dataset, limits.get("val_sample_size"), limits.get("val_sample_seed"), "val")
    if any(value is not None for value in selection.values()) and HydraConfig.initialized():
        import json
        from pathlib import Path

        output = Path(HydraConfig.get().runtime.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "dataset-selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=finqa_flow,
        evaluator=finqa_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()
    expected_checkpoint_step = limits.get("expected_checkpoint_step")
    if limits.get("resource_profile") == "a800_80gb" and expected_checkpoint_step is not None:
        from .gpu import verify_lora_checkpoint

        verify_lora_checkpoint(
            Path(config.trainer.default_local_dir) / f"global_step_{expected_checkpoint_step}/actor", config.actor_rollout_ref.model.lora_rank, config.actor_rollout_ref.model.lora_alpha
        )


if __name__ == "__main__":
    main()
