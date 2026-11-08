"""Opt-in, single-run diagnostics; enabled only via PYTHONPATH + audit directory.

Wrap calls without changing rewards, gradients, loss, or optimizer arguments.
No request headers, keys, prompts, or environment dumps are recorded.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = os.environ.get("FINQA_SMOKE_AUDIT_DIR")


def emit(event, **data):
    path = Path(ROOT) / f"events-{os.getpid()}.jsonl"
    with path.open("a") as out:
        out.write(json.dumps({"time": time.time(), "pid": os.getpid(), "event": event, **data}, default=str) + "\n")


def claim_optimizer_step():
    """Shared exclusive files enforce the configured cap across restarts."""
    maximum = int(os.environ.get("FINQA_MAX_OPTIMIZER_STEPS", "1"))
    for step_index in range(1, maximum + 1):
        try:
            fd = os.open(str(Path(ROOT) / f"optimizer-step-{step_index}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as out:
            out.write(str(os.getpid()))
        return step_index
    raise RuntimeError(f"Optimizer step limit reached ({maximum})")


def patch_fsdp(module):
    cls = module.FSDPEngine
    original = cls.optimizer_step

    @functools.wraps(original)
    def checked(self, *args, **kwargs):
        step_index = claim_optimizer_step()
        import torch

        params = [(n, p) for n, p in self.module.named_parameters() if p.requires_grad]
        trainable_numel = sum(p.numel() for _, p in params)
        if not params or trainable_numel > 100_000_000:
            raise RuntimeError("Unexpected trainable size for the verified 66M-parameter LoRA model")
        # FSDP1 may flatten names; keep their original-name metadata as evidence.
        parameter_names = {n: list(getattr(p, "_fqns", [])) for n, p in params}
        emit("trainable_parameters", numel=trainable_numel, names=parameter_names)
        before = {n: p.detach().to("cpu", copy=True).float() for n, p in params}
        grad_finite = all(bool(torch.isfinite(p.grad).all()) for _, p in params if p.grad is not None)
        grad_nonzero = sum(bool(torch.count_nonzero(p.grad)) for _, p in params if p.grad is not None)
        digest = hashlib.sha256()
        for n, p in before.items():
            digest.update(n.encode())
            digest.update(p.contiguous().numpy().tobytes())
        emit(
            "optimizer_before",
            step_index=step_index,
            trainable_numel=sum(p.numel() for _, p in params),
            tensors=len(params),
            gradients_finite=grad_finite,
            tensors_with_nonzero_grad=grad_nonzero,
            sha256=digest.hexdigest(),
        )
        pre = self.optimizer.register_step_pre_hook(lambda *a: emit("optimizer_step_called", step_index=step_index))
        post = self.optimizer.register_step_post_hook(lambda *a: emit("optimizer_step_returned", step_index=step_index))
        try:
            result = original(self, *args, **kwargs)
        finally:
            pre.remove()
            post.remove()
        digest = hashlib.sha256()
        changed, max_delta = 0, 0.0
        for n, p in params:
            after = p.detach().cpu().float()
            changed += int(not torch.equal(before[n], after))
            if p.numel():
                max_delta = max(max_delta, (after - before[n]).abs().max().item())
            digest.update(n.encode())
            digest.update(after.contiguous().numpy().tobytes())
        emit(
            "optimizer_after",
            step_index=step_index,
            grad_norm=result,
            changed_tensors=changed,
            max_abs_delta=max_delta,
            sha256=digest.hexdigest(),
            cuda_peak_allocated=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved=torch.cuda.max_memory_reserved(),
        )
        return result

    cls.optimizer_step = checked
    emit("hook_installed", target="FSDPEngine.optimizer_step")


def patch_backend(module):
    cls = module.VerlBackend
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        emit("phase_start", phase="trainer_init")
        try:
            result = original_init(self, *args, **kwargs)
        except Exception as exc:
            emit("phase_error", phase="trainer_init", error_type=type(exc).__name__)
            raise
        emit("phase_end", phase="trainer_init")
        return result

    cls.__init__ = init

    def wrap(name, original):
        @functools.wraps(original)
        async def observed(self, *args, **kwargs):
            is_validation = bool(kwargs.get("is_validation", False)) if name == "generate_episodes" else None
            emit("phase_start", phase=name, is_validation=is_validation)
            try:
                result = await original(self, *args, **kwargs)
            except Exception as exc:
                emit("phase_error", phase=name, error_type=type(exc).__name__)
                raise
            if name == "generate_episodes":
                # Persist before backend conversion can fail; do not wait for advantages.
                records = []
                for e in result:
                    steps = [s for t in e.trajectories for s in t.steps]
                    record = {
                        "id": e.id,
                        "is_correct": e.is_correct,
                        "termination": str(e.termination_reason),
                        "rewards": [t.reward for t in e.trajectories],
                        "steps": [{"prompt_ids": s.prompt_ids, "response_ids": s.response_ids, "logprobs": s.logprobs} for s in steps],
                    }
                    records.append(record)
                payload = {"is_validation": is_validation, "episodes": records}
                (Path(ROOT) / f"rollouts-{os.getpid()}-{time.time_ns()}.json").write_text(json.dumps(payload, default=str) + "\n")
                emit("rollouts_saved", count=len(records), steps=sum(len(r["steps"]) for r in records), is_validation=is_validation)
            if name == "compute_advantages":
                state = args[0] if args else kwargs["trainer_state"]
                groups = []
                for group in state.trajectory_groups:
                    groups.append({"id": group.group_id, "rewards": [t.reward for t in group.trajectories], "step_advantages": [[s.advantage for s in t.steps] for t in group.trajectories]})
                episodes = []
                for e in state.episodes:
                    steps = [s for t in e.trajectories for s in t.steps]
                    episodes.append(
                        {
                            "id": e.id,
                            "is_correct": e.is_correct,
                            "termination": str(e.termination_reason),
                            "steps": len(steps),
                            "prompt_ids": sum(len(s.prompt_ids) for s in steps),
                            "response_ids": sum(len(s.response_ids) for s in steps),
                            "logprobs": sum(len(s.logprobs) for s in steps),
                            "logprob_lengths_match": all(len(s.response_ids) == len(s.logprobs) for s in steps),
                        }
                    )
                emit("advantages_and_tokens", groups=groups, episodes=episodes, metrics=state.metrics)
            if name == "update_policy":
                state = args[0] if args else kwargs["trainer_state"]
                emit("policy_update_metrics", metrics=state.metrics)
            emit("phase_end", phase=name, is_validation=is_validation)
            return result

        return observed

    for name in ("generate_episodes", "process_backend_batch", "compute_advantages", "update_policy", "on_batch_end", "on_policy_updated"):
        setattr(cls, name, wrap(name, getattr(cls, name)))
    emit("hook_installed", target="VerlBackend")


def patch_http(module):
    original = module.Client.send

    @functools.wraps(original)
    def send(self, request, *args, **kwargs):
        endpoint = (os.environ.get("FINQA_SMOKE_JUDGE_ENDPOINT") or os.environ.get("FINQA_JUDGE_BASE_URL", "")).rstrip("/")
        configured = urlsplit(endpoint)
        actual = urlsplit(str(request.url))
        judge = (
            bool(endpoint)
            and request.method == "POST"
            and (actual.scheme, actual.netloc, actual.path.rstrip("/")) == (configured.scheme, configured.netloc, configured.path.rstrip("/") + "/chat/completions")
        )
        start = time.time()
        if judge:
            emit("judge_http_attempt")
        try:
            response = original(self, request, *args, **kwargs)
        except Exception as exc:
            if judge:
                emit("judge_http_error", error_type=type(exc).__name__, seconds=time.time() - start)
            raise
        if judge:
            # The configured judge uses non-streaming chat completions.
            response.read()
            try:
                body = response.json()
            except ValueError:
                body = {}
            emit("judge_http_response", status=response.status_code, seconds=time.time() - start, model=body.get("model"), usage=body.get("usage"))
        return response

    module.Client.send = send
    emit("hook_installed", target="httpx.Client.send", judge_endpoint_configured=bool(os.environ.get("FINQA_SMOKE_JUDGE_ENDPOINT") or os.environ.get("FINQA_JUDGE_BASE_URL")))


def patch_openai_chat(module):
    """Record logical judge usage above HTTP retries without logging prompts."""
    cls = module.Completions
    original = cls.create

    @functools.wraps(original)
    def create(self, *args, **kwargs):
        start = time.time()
        emit("judge_sdk_attempt")
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            emit("judge_sdk_error", error_type=type(exc).__name__, seconds=time.time() - start)
            raise
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        emit("judge_sdk_response", seconds=time.time() - start, model=getattr(response, "model", None), usage=usage)
        return response

    cls.create = create
    emit("hook_installed", target="openai.Completions.create")


def patch_ray_env(module):
    original = module.get_ppo_ray_runtime_env

    @functools.wraps(original)
    def environment():
        result = original()
        # Only diagnostics paths; never copy credentials into runtime-env logs.
        result["env_vars"]["FINQA_SMOKE_AUDIT_DIR"] = ROOT
        result["env_vars"]["PYTHONPATH"] = os.environ["PYTHONPATH"]
        # Ray may reconstruct evaluator globals while omitting FINQA_* from the
        # worker environment. Forward only the public endpoint for the counter.
        from urllib.parse import urlsplit, urlunsplit

        endpoint = os.environ.get("FINQA_JUDGE_BASE_URL", "")
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Judge audit endpoint must not contain embedded credentials or query parameters")
            result["env_vars"]["FINQA_SMOKE_JUDGE_ENDPOINT"] = urlunsplit(parsed)
        return result

    module.get_ppo_ray_runtime_env = environment


PATCHERS = {
    "verl.workers.engine.fsdp.transformer_impl": patch_fsdp,
    "rllm.trainer.verl.verl_backend": patch_backend,
    "httpx._client": patch_http,
    "openai.resources.chat.completions.completions": patch_openai_chat,
    "rllm.trainer.verl.ray_runtime_env": patch_ray_env,
}


class AuditFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            PATCHERS[fullname](module)

        spec.loader.exec_module = execute
        return spec


if ROOT:
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    sys.meta_path.insert(0, AuditFinder())
    emit("audit_process_started")
