from finqa_agentic_rl.train import _fixed_sample


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
