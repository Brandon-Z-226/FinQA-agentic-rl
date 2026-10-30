#!/usr/bin/env bash
# Derived from rLLM FinQA Cookbook; Apache-2.0.
# Modified: module entry point and optional FINQA_PYTHON; all overrides preserved.
set -euo pipefail

"${FINQA_PYTHON:-python}" -u -m finqa_agentic_rl.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8 \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    rllm.rollout.train.temperature=0.6 \
    rllm.rollout.val.temperature=0.6 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=finqa \
    rllm.trainer.experiment_name=qwen3-30b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"
