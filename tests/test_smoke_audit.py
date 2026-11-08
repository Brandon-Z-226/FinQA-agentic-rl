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
    with pytest.raises(RuntimeError, match="limit reached"):
        engine.optimizer_step()
    assert torch.equal(engine.module.lora_A, after)
    events = [json.loads(line) for p in Path(audit.ROOT).glob("events-*.jsonl") for line in p.read_text().splitlines()]
    assert sum(e["event"] == "optimizer_step_returned" for e in events) == 1
    update = next(e for e in events if e["event"] == "optimizer_after")
    assert update["changed_tensors"] == 1
    assert update["max_abs_delta"] > 0


def test_four_step_optimizer_cap(audit, monkeypatch):
    monkeypatch.setenv("FINQA_MAX_OPTIMIZER_STEPS", "4")
    assert [audit.claim_optimizer_step() for _ in range(4)] == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match=r"limit reached \(4\)"):
        audit.claim_optimizer_step()


@pytest.mark.parametrize("endpoint_var", ["FINQA_JUDGE_BASE_URL", "FINQA_SMOKE_JUDGE_ENDPOINT"])
def test_judge_usage_without_credentials(audit, monkeypatch, endpoint_var):
    import httpx

    monkeypatch.delenv("FINQA_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("FINQA_SMOKE_JUDGE_ENDPOINT", raising=False)
    monkeypatch.setenv(endpoint_var, "https://judge.invalid/v1")

    class Client:
        def _send_request(self, request, *args, **kwargs):
            assert request.headers["Authorization"] == "Bearer DO_NOT_RECORD"
            return httpx.Response(200, json={"model": "judge", "usage": {"total_tokens": 42}})

    audit.patch_openai_transport(SimpleNamespace(SyncAPIClient=Client))
    request = httpx.Request(
        "POST",
        "https://judge.invalid/v1/chat/completions?api-version=current",
        headers={"Authorization": "Bearer DO_NOT_RECORD"},
        json={"messages": [{"content": "PRIVATE_PROMPT"}]},
    )
    response = Client()._send_request(request)
    assert response.json()["usage"]["total_tokens"] == 42
    recorded = "".join(p.read_text() for p in Path(audit.ROOT).glob("events-*.jsonl"))
    assert '"judge_http_attempt"' in recorded
    assert '"status": 200' in recorded
    assert "DO_NOT_RECORD" not in recorded
    assert "PRIVATE_PROMPT" not in recorded


def test_judge_sdk_usage_without_prompts(audit):
    class Usage:
        def model_dump(self):
            return {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}

    class Completions:
        def create(self, **kwargs):
            assert kwargs["messages"][0]["content"] == "PRIVATE_PROMPT"
            return SimpleNamespace(model="judge", usage=Usage())

    audit.patch_openai_chat(SimpleNamespace(Completions=Completions))
    response = Completions().create(model="judge", messages=[{"content": "PRIVATE_PROMPT"}])
    assert response.usage.model_dump()["total_tokens"] == 42
    recorded = "".join(p.read_text() for p in Path(audit.ROOT).glob("events-*.jsonl"))
    assert '"total_tokens": 42' in recorded
    assert "PRIVATE_PROMPT" not in recorded


def test_rollouts_saved_before_backend_conversion_failure(audit):
    import asyncio

    step = SimpleNamespace(prompt_ids=[1, 2], response_ids=[3], logprobs=[-0.5])
    episode = SimpleNamespace(id="question:0", is_correct=True, termination_reason="ENV_DONE", trajectories=[SimpleNamespace(reward=1.0, steps=[step])])

    class Backend:
        def __init__(self):
            pass

        async def generate_episodes(self):
            return [episode]

        async def process_backend_batch(self):
            raise AssertionError("conversion failed")

        async def compute_advantages(self):
            pass

        update_policy = on_batch_end = on_policy_updated = compute_advantages

    audit.patch_backend(SimpleNamespace(VerlBackend=Backend))
    backend = Backend()
    assert asyncio.run(backend.generate_episodes()) == [episode]
    with pytest.raises(AssertionError):
        asyncio.run(backend.process_backend_batch())
    saved = json.loads(next(Path(audit.ROOT).glob("rollouts-*.json")).read_text())
    assert saved["is_validation"] is False
    assert saved["episodes"][0]["steps"] == [{"prompt_ids": [1, 2], "response_ids": [3], "logprobs": [-0.5]}]


def test_ray_audit_forwards_only_public_endpoint(audit, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/audit")
    monkeypatch.setenv("FINQA_JUDGE_BASE_URL", "https://judge.invalid/v1")
    monkeypatch.setenv("FINQA_JUDGE_API_KEY", "DO_NOT_FORWARD")
    module = SimpleNamespace(get_ppo_ray_runtime_env=lambda: {"env_vars": {}})
    audit.patch_ray_env(module)
    env = module.get_ppo_ray_runtime_env()["env_vars"]
    assert env["FINQA_SMOKE_JUDGE_ENDPOINT"] == "https://judge.invalid/v1"
    assert "DO_NOT_FORWARD" not in json.dumps(env)
    monkeypatch.setenv("FINQA_JUDGE_BASE_URL", "https://judge.invalid/v1?key=PRIVATE")
    with pytest.raises(ValueError, match="embedded credentials"):
        module.get_ppo_ray_runtime_env()


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
