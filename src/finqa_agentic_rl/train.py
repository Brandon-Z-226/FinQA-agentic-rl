# Derived from rLLM cookbooks/finqa at 4e828c81680e5a85b638ceedeaf09d1e90109ca8.
# Apache-2.0; see licenses/rllm-LICENSE and THIRD_PARTY_NOTICES.md.
# Modified: package imports and project integration; see docs/migration.md.
"""Train the finqa agent using the Python API.

Usage (from this project root, after selecting a training environment)::

    python -m finqa_agentic_rl.train rllm/backend=tinker

    # Hydra overrides:
    python -m finqa_agentic_rl.train rllm/backend=tinker model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry

from .finqa_eval import finqa_evaluator
from .finqa_flow import finqa_flow


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
    if limits.get("train_limit") is not None:
        train_dataset = train_dataset.select(range(limits["train_limit"]))
    if limits.get("val_limit") is not None:
        val_dataset = val_dataset.select(range(limits["val_limit"]))

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=finqa_flow,
        evaluator=finqa_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()
    if limits.get("resource_profile") == "a800_80gb" and limits.get("train_limit") is not None and config.trainer.total_training_steps == 1:
        from .gpu import verify_lora_checkpoint

        verify_lora_checkpoint(Path(config.trainer.default_local_dir) / "global_step_1/actor", config.actor_rollout_ref.model.lora_rank, config.actor_rollout_ref.model.lora_alpha)


if __name__ == "__main__":
    main()
