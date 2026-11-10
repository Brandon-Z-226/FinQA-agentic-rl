import pytest
from omegaconf import OmegaConf

from finqa_agentic_rl.train import _fixed_sample, _require_empty_checkpoint_dir, _validate_dataset_size


class Dataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return Dataset([self.rows[index] for index in indices])


def test_fixed_sample_is_unique_reproducible_and_not_prefix():
    dataset = Dataset(range(100))
    first, first_indices = _fixed_sample(dataset, 64, 20261109, "train")
    second, second_indices = _fixed_sample(dataset, 64, 20261109, "train")
    assert first_indices == second_indices
    assert first.rows == second.rows
    assert len(first_indices) == len(set(first_indices)) == 64
    assert first_indices != list(range(64))


def test_full_split_size_and_fresh_checkpoint_guards(tmp_path):
    dataset = Dataset(range(4030))
    _validate_dataset_size(dataset, 4030, "train")
    with pytest.raises(RuntimeError, match="Expected 4031"):
        _validate_dataset_size(dataset, 4031, "train")

    checkpoint = tmp_path / "checkpoint"
    config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(checkpoint)},
            "finqa": {"require_empty_checkpoint_dir": True},
        }
    )
    _require_empty_checkpoint_dir(config)
    checkpoint.mkdir()
    _require_empty_checkpoint_dir(config)
    (checkpoint / "stale").write_text("data")
    with pytest.raises(RuntimeError, match="requires an empty checkpoint directory"):
        _require_empty_checkpoint_dir(config)
