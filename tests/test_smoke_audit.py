import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def audit(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/smoke_audit/sitecustomize.py"
    spec = importlib.util.spec_from_file_location("smoke_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = str(tmp_path)
    return module


def test_one_optimizer_call_and_parameter_change(audit, monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    class Engine:
        def __init__(self):
            self.module = torch.nn.Module()
            self.module.register_parameter("lora_A", torch.nn.Parameter(torch.tensor([1.0, 2.0])))
            self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.01)

        def optimizer_step(self):
            self.optimizer.step()
            return 1.0

    audit.patch_fsdp(SimpleNamespace(FSDPEngine=Engine))
    engine = Engine()
    engine.module.lora_A.sum().backward()
    engine.optimizer_step()
    after = engine.module.lora_A.detach().clone()
    with pytest.raises(FileExistsError):
        engine.optimizer_step()
    assert torch.equal(engine.module.lora_A, after)
    events = [json.loads(line) for p in Path(audit.ROOT).glob("events-*.jsonl") for line in p.read_text().splitlines()]
    assert sum(e["event"] == "optimizer_step_returned" for e in events) == 1
    update = next(e for e in events if e["event"] == "optimizer_after")
    assert update["changed_tensors"] == 1
    assert update["max_abs_delta"] > 0


def test_judge_usage_without_credentials(audit, monkeypatch):
    import httpx._client

    monkeypatch.setenv("FINQA_JUDGE_BASE_URL", "https://judge.invalid/v1")
    original = httpx.Client.send
    monkeypatch.setattr(httpx.Client, "send", original)
    audit.patch_http(httpx._client)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"model": "judge", "usage": {"total_tokens": 42}}))
    with httpx.Client(transport=transport) as client:
        response = client.post("https://judge.invalid/v1/chat/completions", headers={"Authorization": "Bearer DO_NOT_RECORD"}, json={"messages": [{"content": "PRIVATE_PROMPT"}]})
        assert response.json()["usage"]["total_tokens"] == 42
    recorded = "".join(p.read_text() for p in Path(audit.ROOT).glob("events-*.jsonl"))
    assert '"total_tokens": 42' in recorded
    assert "DO_NOT_RECORD" not in recorded
    assert "PRIVATE_PROMPT" not in recorded


def test_cgroup_v1_and_v2_memory(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_smoke_audit.py"
    spec = importlib.util.spec_from_file_location("run_smoke_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.usage_in_bytes").write_text("300")
    (v1 / "memory.stat").write_text("total_rss 100\ntotal_cache 200\n")
    (v1 / "memory.failcnt").write_text("0")
    assert module.cgroup_memory(tmp_path)["anon_bytes"] == 100
    assert module.cgroup_memory(tmp_path)["cgroup_version"] == 1
    (tmp_path / "memory.current").write_text("400")
    (tmp_path / "memory.stat").write_text("anon 150\nfile 250\n")
    (tmp_path / "memory.events").write_text("oom 0\noom_kill 0\n")
    assert module.cgroup_memory(tmp_path)["anon_bytes"] == 150
    assert module.cgroup_memory(tmp_path)["cgroup_version"] == 2
