import importlib
import json
from types import SimpleNamespace

import pytest

evaluator = importlib.import_module("finqa_agentic_rl.finqa_eval")


@pytest.fixture
def judge_env(monkeypatch):
    # Reload only while the fake SDK constructor is installed; no API requests.
    with monkeypatch.context() as patch:
        yield patch
    importlib.reload(evaluator)


@pytest.mark.parametrize("custom_key", [None, "", "test-judge-key"])
def test_custom_endpoint_never_inherits_unrelated_openai_key(judge_env, custom_key):
    calls = []
    judge_env.setenv("OPENAI_API_KEY", "unrelated-agent-key")
    judge_env.setenv("FINQA_JUDGE_BASE_URL", "https://judge.invalid/v1")
    judge_env.setenv("FINQA_JUDGE_MODEL", "custom-judge")
    judge_env.setenv("FINQA_JUDGE_API_STYLE", "chat_completions")
    if custom_key is not None:
        judge_env.setenv("FINQA_JUDGE_API_KEY", custom_key)
    judge_env.setattr(evaluator.openai, "OpenAI", lambda **kwargs: calls.append(kwargs))
    importlib.reload(evaluator)
    assert evaluator.JUDGE_MODEL == evaluator.MULTI_TABLE_JUDGE_MODEL == "custom-judge"
    assert calls == ([{"api_key": custom_key, "base_url": "https://judge.invalid/v1"}] if custom_key else [])


def test_original_openai_configuration_still_works(judge_env):
    calls = []
    judge_env.setenv("OPENAI_API_KEY", "test-original-key")
    judge_env.setattr(evaluator.openai, "OpenAI", lambda **kwargs: calls.append(kwargs))
    importlib.reload(evaluator)
    assert calls == [{"api_key": "test-original-key"}]
    assert evaluator.JUDGE_MODEL == "gpt-5-nano"
    assert evaluator.MULTI_TABLE_JUDGE_MODEL == "gpt-5-mini"
    assert evaluator.JUDGE_API_STYLE == "responses"


@pytest.mark.parametrize("multi", [False, True])
def test_chat_transport_preserves_prompts_and_scoring(monkeypatch, multi):
    calls = []
    rubric = dict.fromkeys(evaluator.CORRECTNESS_WEIGHTS, 95)

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(rubric) if multi else "true"))])

    monkeypatch.setattr(evaluator, "JUDGE_API_STYLE", "chat_completions")
    monkeypatch.setattr(evaluator, "JUDGE_MODEL", "custom-single")
    monkeypatch.setattr(evaluator, "MULTI_TABLE_JUDGE_MODEL", "custom-multi")
    monkeypatch.setattr(evaluator, "_JUDGE_CLIENT", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    verdict, parsed = evaluator._call_judge("original system prompt", "original question and answer", multi_table=multi)
    assert verdict == pytest.approx(0.95 if multi else 1.0)
    assert parsed == (rubric if multi else {})
    assert calls == [
        {
            "model": "custom-multi" if multi else "custom-single",
            "messages": [{"role": "system", "content": "original system prompt"}, {"role": "user", "content": "original question and answer"}],
            "max_tokens": 5000 if multi else 512,
        }
    ]


def test_unsupported_api_style_fails_before_requests(judge_env):
    judge_env.setenv("FINQA_JUDGE_API_STYLE", "typo")
    with pytest.raises(ValueError, match="FINQA_JUDGE_API_STYLE"):
        importlib.reload(evaluator)


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_chat_multi_table_accepts_complete_json_fence(monkeypatch, fence):
    rubric = {**dict.fromkeys(evaluator.CORRECTNESS_WEIGHTS, 100), "explanation": "Correct."}
    content = f"{fence}\n{json.dumps(rubric)}\n```"
    endpoint = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))
    monkeypatch.setattr(evaluator, "JUDGE_API_STYLE", "chat_completions")
    monkeypatch.setattr(evaluator, "_JUDGE_CLIENT", SimpleNamespace(chat=SimpleNamespace(completions=endpoint)))
    verdict, parsed = evaluator._call_judge("system", "input", multi_table=True)
    assert verdict == 1.0
    assert parsed == rubric


@pytest.mark.parametrize("content", ["```json\nnot JSON\n```", 'Explanation first\n```json\n{"primary_data_score": 100}\n```'])
def test_chat_multi_table_does_not_salvage_invalid_or_embedded_json(monkeypatch, content):
    endpoint = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]))
    monkeypatch.setattr(evaluator, "JUDGE_API_STYLE", "chat_completions")
    monkeypatch.setattr(evaluator, "_JUDGE_CLIENT", SimpleNamespace(chat=SimpleNamespace(completions=endpoint)))
    assert evaluator._call_judge("system", "input", multi_table=True) == (0.0, {})


@pytest.mark.parametrize("style", ["responses", "chat_completions"])
@pytest.mark.parametrize("multi", [False, True])
def test_low_reasoning_override_for_both_protocols_and_rubrics(judge_env, style, multi):
    judge_env.setenv("FINQA_JUDGE_API_STYLE", style)
    judge_env.setenv("FINQA_JUDGE_REASONING_EFFORT", "low")
    importlib.reload(evaluator)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        output = json.dumps(dict.fromkeys(evaluator.CORRECTNESS_WEIGHTS, 100)) if multi else "true"
        return SimpleNamespace(output_text=output, choices=[SimpleNamespace(message=SimpleNamespace(content=output))])

    endpoint = SimpleNamespace(create=create)
    judge_env.setattr(evaluator, "_JUDGE_CLIENT", SimpleNamespace(responses=endpoint, chat=SimpleNamespace(completions=endpoint)))
    verdict, _ = evaluator._call_judge("system", "input", multi_table=multi)
    assert verdict == 1.0
    if style == "responses":
        assert calls[0]["reasoning"] == {"effort": "low"}
        assert calls[0]["max_output_tokens"] == (5000 if multi else 512)
    else:
        assert calls[0]["reasoning_effort"] == "low"
        assert calls[0]["max_completion_tokens"] == (5000 if multi else 512)
        assert "max_tokens" not in calls[0]
