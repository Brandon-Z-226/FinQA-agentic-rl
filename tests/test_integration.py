import copy
import importlib
from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest
from rllm.eval.agent_loader import load_agent
from rllm.eval.evaluator_loader import load_evaluator
from rllm.types import AgentConfig, Episode, Task

flow = importlib.import_module("finqa_agentic_rl.finqa_flow")
evaluator = importlib.import_module("finqa_agentic_rl.finqa_eval")


def tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def test_plugin_discovery_and_import_paths():
    assert load_agent("finqa") is flow.finqa_flow
    assert load_evaluator("finqa") is evaluator.finqa_evaluator
    for group in ("rllm.agents", "rllm.evaluators"):
        plugins = [ep for ep in entry_points(group=group) if ep.name == "finqa"]
        assert len(plugins) == 1
        assert plugins[0].value.startswith("finqa_agentic_rl.")
    importlib.import_module("finqa_agentic_rl.train")
    importlib.import_module("finqa_agentic_rl.prepare_finqa_data")
    importlib.import_module("rllm_model_gateway")


def test_multiturn_flow_uses_injected_gateway_and_real_tools(monkeypatch):
    requests = []
    constructor = {}
    replies = [
        SimpleNamespace(role="assistant", content="Calculate.", tool_calls=[tool_call("calculator", '{"expression":"2+3"}')]),
        SimpleNamespace(role="assistant", content="FINAL ANSWER: 5", tool_calls=[]),
    ]

    class FakeClient:
        def __init__(self, **kwargs):
            constructor.update(kwargs)
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))])

    monkeypatch.setattr(flow, "AsyncOpenAI", FakeClient)
    episode = flow.finqa_flow.run(Task(id="offline", instruction="2+3?"), AgentConfig(base_url="http://gateway.invalid/v1", model="offline-model", session_uid="test"))
    assert constructor == {"base_url": "http://gateway.invalid/v1", "api_key": "EMPTY"}
    assert episode.artifacts == {"answer": "FINAL ANSWER: 5", "accessed_tables": [], "turns": 2}
    assert requests[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5.0"}
    assert requests[0]["tools"] == flow.TOOL_SPECS
    assert len(episode.trajectories[0].steps) == 2


@pytest.mark.parametrize("failure", [False, True])
def test_flow_turn_limit_and_model_error(monkeypatch, failure):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            if failure:
                raise RuntimeError("offline simulated error")
            msg = SimpleNamespace(role="assistant", content="still thinking", tool_calls=[tool_call("calculator", '{"expression":"1+1"}')])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(flow, "AsyncOpenAI", FakeClient)
    episode = flow.finqa_flow.run(Task(id="limit", instruction="test"), AgentConfig(base_url="http://unused.invalid", model="fake", session_uid="test"))
    assert len(calls) == (1 if failure else 20)
    assert episode.artifacts["turns"] == (0 if failure else 20)
    assert episode.artifacts["answer"] == ("" if failure else "still thinking")


def test_dispatch_errors_and_table_tracking():
    accessed = []
    assert "unknown tool" in flow._exec_tool_call(tool_call("nope", "{}"), accessed)
    assert "parse" in flow._exec_tool_call(tool_call("calculator", "{"), accessed)
    assert "bad arguments" in flow._exec_tool_call(tool_call("calculator", "{}"), accessed)
    flow._exec_tool_call(tool_call("get_table_info", '{"company_name":"unknown","table_name":" MyTable "}'), accessed)
    # Preserve upstream: attempted metadata access counts even if tool returns an error.
    assert accessed == ["mytable"]


def test_missing_judge_key_returns_zero_without_table_bonus(monkeypatch):
    monkeypatch.setattr(evaluator, "_JUDGE_CLIENT", None)
    episode = Episode(artifacts={"answer": "FINAL ANSWER: 5", "accessed_tables": ["t"]})
    result = evaluator.finqa_evaluator({"question": "2+3?", "ground_truth": "5", "table_name": ["t"]}, episode)
    assert result.reward == 0
    assert not result.is_correct
    assert {s.name: s.value for s in result.signals} == {"accuracy": 0, "table_access": 1}


@pytest.mark.parametrize("multi", [False, True])
def test_real_evaluator_rubric_with_fake_judge(monkeypatch, multi):
    import json

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        output = json.dumps(dict.fromkeys(evaluator.CORRECTNESS_WEIGHTS, 95)) if multi else "true"
        return SimpleNamespace(output_text=output)

    monkeypatch.setattr(evaluator, "_JUDGE_CLIENT", SimpleNamespace(responses=SimpleNamespace(create=create)))
    task = Task(id="judge", instruction="test", metadata={"question": "2+3?", "ground_truth": "5", "question_type": "multi_table_test" if multi else "single_table"})
    result = evaluator.finqa_evaluator(task, Episode(artifacts={"answer": "FINAL ANSWER: 5"}))
    assert result.reward == pytest.approx(0.95 if multi else 1.0)
    assert requests[0]["model"] == ("gpt-5-mini" if multi else "gpt-5-nano")
    assert result.is_correct
