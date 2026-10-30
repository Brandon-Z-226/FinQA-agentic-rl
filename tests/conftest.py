import os
import socket
from pathlib import Path

import pytest

# Configure before test collection imports the evaluator or the tool store.
os.environ["FINQA_SKIP_PRELOAD"] = "1"
os.environ.pop("OPENAI_API_KEY", None)
os.environ["RLLM_HOME"] = str(Path(__file__).resolve().parents[1] / ".cache/test-rllm")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    from rllm.data.dataset import DatasetRegistry

    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(tmp_path / "datasets/registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(tmp_path / "datasets"))
    return DatasetRegistry
