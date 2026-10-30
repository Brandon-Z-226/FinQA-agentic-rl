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
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry

from .finqa_eval import finqa_evaluator
from .finqa_flow import finqa_flow


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    # Backend imports are needed only when actually starting training.
    from rllm.trainer import AgentTrainer

    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("FinQA dataset not found. Run: finqa-prepare")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=finqa_flow,
        evaluator=finqa_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
