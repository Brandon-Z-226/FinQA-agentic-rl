# Derived from rLLM cookbooks/finqa at 4e828c81680e5a85b638ceedeaf09d1e90109ca8.
# Apache-2.0; see licenses/rllm-LICENSE and THIRD_PARTY_NOTICES.md.
# Modified: package imports and project integration; see docs/migration.md.
"""FinQA evaluator: judge LLM correctness + table-access bonus.

Ports the judge call, single-table vs multi-table rubric, and table-access
scoring from the legacy :func:`projects.finqa.fin_qa_reward.fin_qa_reward_function`
into a self-contained module so the cookbook does not depend on
``projects/finqa``.

Judge connection and model overrides use ``FINQA_JUDGE_*`` variables.
Without overrides, the original OpenAI Responses configuration is retained.
Judge transport/configuration failures raise instead of becoming a false reward.
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx
import openai
import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

from .finqa_constants import CORRECTNESS_PROMPT_PATH, MULTI_TABLE_CORRECTNESS_PROMPT_PATH

# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------

with open(CORRECTNESS_PROMPT_PATH, encoding="utf-8") as f:
    CORRECTNESS_PROMPT = f.read()

with open(MULTI_TABLE_CORRECTNESS_PROMPT_PATH, encoding="utf-8") as f:
    MULTI_TABLE_CORRECTNESS_PROMPT = f.read()

JUDGE_BASE_URL = os.environ.get("FINQA_JUDGE_BASE_URL") or None
# Never send an unrelated OPENAI_API_KEY to a custom judge endpoint.
# An explicitly empty FINQA_JUDGE_API_KEY also disables the legacy fallback.
OPENAI_API_KEY = os.environ.get("FINQA_JUDGE_API_KEY", None if JUDGE_BASE_URL else os.environ.get("OPENAI_API_KEY"))
JUDGE_MODEL = os.environ.get("FINQA_JUDGE_MODEL") or "gpt-5-nano"
MULTI_TABLE_JUDGE_MODEL = os.environ.get("FINQA_JUDGE_MULTI_TABLE_MODEL") or os.environ.get("FINQA_JUDGE_MODEL") or "gpt-5-mini"
JUDGE_API_STYLE = os.environ.get("FINQA_JUDGE_API_STYLE") or "responses"
JUDGE_REASONING_EFFORT = os.environ.get("FINQA_JUDGE_REASONING_EFFORT") or None
JUDGE_SINGLE_MAX_OUTPUT_TOKENS = int(os.environ.get("FINQA_JUDGE_SINGLE_MAX_OUTPUT_TOKENS", "512"))
JUDGE_MAX_ATTEMPTS = int(os.environ.get("FINQA_JUDGE_MAX_ATTEMPTS", "3"))
JUDGE_RETRY_BASE_SECONDS = float(os.environ.get("FINQA_JUDGE_RETRY_BASE_SECONDS", "1"))
if JUDGE_API_STYLE not in {"responses", "chat_completions"}:
    raise ValueError("FINQA_JUDGE_API_STYLE must be responses or chat_completions")
if not 1 <= JUDGE_SINGLE_MAX_OUTPUT_TOKENS <= 5000:
    raise ValueError("FINQA_JUDGE_SINGLE_MAX_OUTPUT_TOKENS must be between 1 and 5000")
if not 1 <= JUDGE_MAX_ATTEMPTS <= 5:
    raise ValueError("FINQA_JUDGE_MAX_ATTEMPTS must be between 1 and 5")
if not 0 <= JUDGE_RETRY_BASE_SECONDS <= 30:
    raise ValueError("FINQA_JUDGE_RETRY_BASE_SECONDS must be between 0 and 30")


class JudgeError(RuntimeError):
    """A judge failure that must never be converted into a training reward."""


class JudgeUnavailableError(JudgeError):
    """Judge credentials or client configuration are unavailable."""


class JudgeAPIError(JudgeError):
    """The judge API failed after the configured finite retry policy."""


class JudgeResponseError(JudgeError):
    """The judge returned a response that cannot be scored safely."""


CORRECTNESS_WEIGHTS = {
    "primary_data_score": 0.30,
    "derived_metrics_score": 0.30,
    "reasoning_score": 0.15,
    "consistency_score": 0.10,
    "completeness_score": 0.10,
    "structure_score": 0.05,
}

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)


def _make_judge_client():
    """Build the judge client without changing the agent's gateway connection."""
    if not OPENAI_API_KEY:
        return None
    kwargs = {"api_key": OPENAI_API_KEY, "max_retries": 0}
    if JUDGE_BASE_URL:
        kwargs["base_url"] = JUDGE_BASE_URL
    return openai.OpenAI(**kwargs)


_JUDGE_CLIENT = _make_judge_client()


def _request_judge_text_once(request: dict) -> str:
    """Send exactly one SDK request; retries are owned by this module."""
    if JUDGE_API_STYLE == "responses":
        if JUDGE_REASONING_EFFORT:
            request = {**request, "reasoning": {**request.get("reasoning", {}), "effort": JUDGE_REASONING_EFFORT}}
        response = _JUDGE_CLIENT.responses.create(**request)
        return getattr(response, "output_text", "") or ""

    # Compatible providers need not support GPT-5 reasoning/verbosity options.
    # The original multi-table prompt already asks for the JSON rubric.
    chat_request = {
        "model": request["model"],
        "messages": [
            {"role": "system", "content": request["instructions"]},
            {"role": "user", "content": request["input"]},
        ],
        "max_tokens": request["max_output_tokens"],
    }
    if JUDGE_REASONING_EFFORT:
        chat_request["reasoning_effort"] = JUDGE_REASONING_EFFORT
        # Reasoning models count hidden reasoning in the completion budget.
        chat_request["max_completion_tokens"] = chat_request.pop("max_tokens")
    response = _JUDGE_CLIENT.chat.completions.create(**chat_request)
    out = response.choices[0].message.content or ""
    # Compatible judges may wrap the multi-table JSON in a Markdown fence.
    # Strip only a complete outer fence; keep scoring and JSON validation intact.
    if request.get("text", {}).get("format", {}).get("type") == "json_schema":
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", out, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            out = fenced.group(1)
    return out


def _retryable_judge_error(exc: Exception) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, httpx.TransportError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def _request_judge_text(request: dict) -> str:
    """Call the judge with finite retries and expose the final failure."""
    if _JUDGE_CLIENT is None:
        raise JudgeUnavailableError("Judge API credentials are not configured")
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            return _request_judge_text_once(request)
        except Exception as exc:
            retryable = _retryable_judge_error(exc)
            if not retryable or attempt == JUDGE_MAX_ATTEMPTS:
                raise JudgeAPIError(f"Judge API failed after {attempt} attempt(s): {type(exc).__name__}") from exc
            time.sleep(JUDGE_RETRY_BASE_SECONDS * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _extract_final_answer(action: str, *, prefer_tail: bool = False) -> str:
    """Pull the FINAL ANSWER section out of a model response."""
    if not action:
        return ""
    m = _FINAL_ANSWER_CODE_BLOCK_RE.search(action)
    if m:
        return m.group(1).strip()
    if not prefer_tail:
        m = _FINAL_ANSWER_PARAGRAPH_RE.search(action)
        if m:
            return m.group(1).strip()
    m = _FINAL_ANSWER_TAIL_RE.search(action)
    if m:
        return m.group(1).strip()
    return action


def _table_access_score(accessed: list[str], expected: str | list[str]) -> float:
    """Fraction of required tables the agent inspected via get_table_info."""
    if not accessed or not expected:
        return 0.0
    seen = {t.lower().strip() for t in accessed if isinstance(t, str) and t.strip()}
    if isinstance(expected, list):
        wanted = [n.lower().strip() for n in expected if isinstance(n, str) and n.strip()]
    else:
        wanted = [expected.lower().strip()] if isinstance(expected, str) else []
    if not wanted:
        return 0.0
    return sum(1 for n in wanted if n in seen) / len(wanted)


def _call_judge(system_prompt: str, user_prompt: str, *, multi_table: bool) -> tuple[bool | float, dict]:
    """Score ``user_prompt`` against ``system_prompt`` via the judge LLM.

    Returns ``(decision, raw_rubric)``. Judge failures and malformed responses
    raise so the trainer cannot consume them as negative rewards.
    """
    request: dict = {
        "model": MULTI_TABLE_JUDGE_MODEL if multi_table else JUDGE_MODEL,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": 5000 if multi_table else JUDGE_SINGLE_MAX_OUTPUT_TOKENS,
    }

    if multi_table:
        schema = {
            "type": "object",
            "properties": {k: {"type": "number"} for k in CORRECTNESS_WEIGHTS} | {"explanation": {"type": "string"}},
            "required": list(CORRECTNESS_WEIGHTS) + ["explanation"],
            "additionalProperties": False,
        }
        request["reasoning"] = {"effort": "medium"}
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "finqa_multi_table_rubric",
                "schema": schema,
                "strict": True,
            }
        }
    else:
        request["reasoning"] = {"effort": "low"}
        request["text"] = {"verbosity": "low"}

    for response_attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        out = _request_judge_text(request)
        try:
            if multi_table:
                parsed = json.loads(out)
                scores = {k: parsed.get(k) for k in CORRECTNESS_WEIGHTS}
                if any(not isinstance(v, int | float) or not 0 <= float(v) <= 100 for v in scores.values()):
                    raise JudgeResponseError("Judge multi-table rubric is incomplete or out of range")
                score = sum((float(scores[k]) / 100.0) * w for k, w in CORRECTNESS_WEIGHTS.items()) / sum(CORRECTNESS_WEIGHTS.values())
                return score, parsed

            # The official prompt requests: {True/False, 'Reason'}. Parse that
            # leading verdict before inspecting explanation text, which may
            # legitimately contain the opposite word (for example "not true").
            leading = re.match(r"\s*[\{\[\(]?\s*(true|false)\b", out, flags=re.IGNORECASE)
            if leading:
                return leading.group(1).lower() == "true", {}
            verdicts = {value.lower() for value in re.findall(r"\b(true|false)\b", out, flags=re.IGNORECASE)}
            if len(verdicts) == 1:
                return verdicts.pop() == "true", {}
            raise JudgeResponseError("Judge single-table verdict is ambiguous or missing")
        except (json.JSONDecodeError, JudgeResponseError) as exc:
            error = exc if isinstance(exc, JudgeResponseError) else JudgeResponseError("Judge returned invalid multi-table JSON")
            if response_attempt == JUDGE_MAX_ATTEMPTS:
                if error is exc:
                    raise
                raise error from exc
            time.sleep(JUDGE_RETRY_BASE_SECONDS * 2 ** (response_attempt - 1))
    raise AssertionError("unreachable")


def _task_meta(task: Task | dict) -> dict:
    """Normalize Runner-supplied Task and trainer-supplied dict to a single shape."""
    if isinstance(task, Task):
        return task.metadata or {}
    return task or {}


@rllm.evaluator
def finqa_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    """Judge-LLM correctness for the agent's FINAL ANSWER, plus table-access bonus."""
    meta = _task_meta(task)
    answer = str(episode.artifacts.get("answer", ""))
    accessed = episode.artifacts.get("accessed_tables", []) or []

    question = meta.get("question")
    core_question = meta.get("core_question") or question
    ground_truth = meta.get("ground_truth")
    qtype = (meta.get("question_type") or "").lower()

    if not answer or not question or not ground_truth:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"reason": "missing answer/question/ground_truth"},
        )

    multi_table = qtype.startswith("multi_table")
    if multi_table:
        prompt_input = f"question : {core_question}\nmodel response : {answer}\nlabel : {ground_truth}"
        verdict, rubric = _call_judge(MULTI_TABLE_CORRECTNESS_PROMPT, prompt_input, multi_table=True)
        correctness = float(verdict)
        is_correct = correctness >= 0.9
    else:
        final_answer = _extract_final_answer(answer)
        prompt_input = f"question : {question}\nmodel response : {final_answer}\nlabel : {ground_truth}"
        verdict, rubric = _call_judge(CORRECTNESS_PROMPT, prompt_input, multi_table=False)
        is_correct = bool(verdict)
        correctness = 1.0 if is_correct else 0.0

    table_bonus = _table_access_score(accessed, meta.get("table_name", ""))

    signals = [
        Signal(name="accuracy", value=correctness),
        Signal(name="table_access", value=table_bonus),
    ]
    metadata: dict = {"right_table_access_reward": table_bonus}
    if multi_table:
        for k in CORRECTNESS_WEIGHTS:
            v = rubric.get(k)
            if isinstance(v, int | float):
                metadata[f"multi_table_{k}"] = float(v)
        metadata["multi_table_overall_score"] = correctness

    return EvalOutput(
        reward=correctness,
        is_correct=is_correct,
        signals=signals,
        metadata=metadata,
    )


# Re-export for tests
__all__ = ["finqa_evaluator", "_extract_final_answer", "_table_access_score"]
