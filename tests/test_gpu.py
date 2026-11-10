import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from finqa_agentic_rl.gpu import model_path, require_disk, validate_training_config, verify_lora_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gpu_config(monkeypatch, tmp_path):
    monkeypatch.setenv("FINQA_GPU_ROOT", str(tmp_path))
    monkeypatch.setenv("FINQA_MODEL_PATH", str(tmp_path / "model"))
    return OmegaConf.load(ROOT / "configs/a800_common.yaml")


def test_lora_and_smoke_profile(gpu_config):
    validate_training_config(gpu_config)
    smoke = OmegaConf.merge(gpu_config, OmegaConf.load(ROOT / "configs/a800_smoke.yaml"))
    validate_training_config(smoke)
    assert smoke.rllm.data.train_batch_size == 2
    assert smoke.rllm.data.val_batch_size == 2
    assert smoke.rllm.trainer.save_freq == 1
    assert smoke.rllm.trainer.test_freq == -1
    assert smoke.rllm.trainer.total_batches == 1
    assert smoke.rllm.rollout.n == 4
    baseline = OmegaConf.merge(gpu_config, OmegaConf.load(ROOT / "configs/a800_baseline_4step.yaml"))
    validate_training_config(baseline)
    assert baseline.data.train_batch_size == 16
    assert baseline.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert baseline.rllm.rollout.n == 4
    assert baseline.actor_rollout_ref.model.lora_rank == 32
    assert baseline.actor_rollout_ref.actor.optim.lr == 1e-6
    assert baseline.rllm.trainer.total_batches == 4
    assert baseline.finqa.train_sample_size == 64
    assert baseline.finqa.val_sample_size == 32
    assert baseline.trainer.val_before_train is True
    full = OmegaConf.merge(gpu_config, OmegaConf.load(ROOT / "configs/a800_baseline_full.yaml"))
    validate_training_config(full)
    assert full.data.train_batch_size == 16
    assert full.rllm.rollout.n == 4
    assert full.actor_rollout_ref.model.lora_rank == 32
    assert full.actor_rollout_ref.actor.optim.lr == 1e-6
    assert full.rllm.trainer.total_batches == 251
    assert full.trainer.val_before_train is True
    assert full.trainer.test_freq == 1000
    assert full.trainer.save_freq == 25
    assert full.trainer.max_actor_ckpt_to_keep == 2
    assert full.trainer.resume_mode == "disable"
    assert full.finqa.expected_train_size == 4030
    assert full.finqa.expected_val_size == 522
    assert full.finqa.save_final_checkpoint is True
    recovery = OmegaConf.create(OmegaConf.to_container(full, resolve=False))
    recovery.trainer.resume_mode = "auto"
    recovery.finqa.require_empty_checkpoint_dir = False
    validate_training_config(recovery)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("actor_rollout_ref.model.lora_rank", 0, "FSDP LoRA is OFF"),
        ("actor_rollout_ref.model.lora_rank", 16, "must agree"),
        ("trainer.n_gpus_per_node", 8, "Single GPU"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 2, "Single GPU"),
        ("rllm.async_training.enable", True, "colocated"),
        ("actor_rollout_ref.rollout.max_model_len", 8192, "budget"),
        ("rllm.rollout.n", 1, "GRPO"),
        ("finqa.train_limit", 1, "full train batch"),
    ],
)
def test_invalid_gpu_configs_fail_early(gpu_config, key, value, error):
    OmegaConf.update(gpu_config, key, value)
    with pytest.raises(ValueError, match=error):
        validate_training_config(gpu_config)


def test_model_path_reuses_pinned_snapshot(monkeypatch, tmp_path):
    monkeypatch.delenv("FINQA_MODEL_PATH", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    path = model_path(tmp_path, ROOT)
    assert path == tmp_path / "huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
    monkeypatch.setenv("FINQA_MODEL_PATH", str(tmp_path / "custom"))
    assert model_path(tmp_path, ROOT) == tmp_path / "custom"


def test_disk_guard(tmp_path):
    assert require_disk(tmp_path, 0) >= 0
    with pytest.raises(RuntimeError, match="Expand the data disk"):
        require_disk(tmp_path, 10**12)


def test_smoke_requires_real_lora_checkpoint_metadata(tmp_path):
    with pytest.raises(RuntimeError, match="lacks LoRA metadata"):
        verify_lora_checkpoint(tmp_path, 32, 32)
    metadata = tmp_path / "lora_train_meta.json"
    metadata.write_text('{"r": 16, "lora_alpha": 32}')
    with pytest.raises(RuntimeError, match="do not match"):
        verify_lora_checkpoint(tmp_path, 32, 32)
    metadata.write_text('{"r": 32, "lora_alpha": 32}')
    verify_lora_checkpoint(tmp_path, 32, 32)


def test_gpu_launcher_rejects_mac_before_install(tmp_path):
    # Controlled uname stub tests the platform guard on Linux CI as well.
    uname = tmp_path / "uname"
    uname.write_text('#!/bin/sh\nprintf "Darwin\\n"\n')
    uname.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    result = subprocess.run(["bash", str(ROOT / "scripts/gpu.sh"), "setup"], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Linux x86_64" in result.stderr
    subprocess.run(["bash", "-n", str(ROOT / "scripts/gpu.sh")], check=True)
