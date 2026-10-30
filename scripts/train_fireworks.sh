#!/usr/bin/env bash
# Derived from rLLM FinQA Cookbook; Apache-2.0.
# Modified: module entry point and optional FINQA_PYTHON; all overrides preserved.
set -euo pipefail

"${FINQA_PYTHON:-python}" -u -m finqa_agentic_rl.train \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-35b-a3b \
    model.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
    training.group_size=8 \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=finqa \
    rllm.trainer.experiment_name=qwen3p5-35b-a3b-fireworks \
    rllm.trainer.logger=[console,ui] \
    "$@"
