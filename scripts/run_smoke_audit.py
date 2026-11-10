"""Run one bounded GPU experiment, recording resources and diagnostics."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def cgroup_memory(root=Path("/sys/fs/cgroup")):
    """Read v2 or v1; historical cgroup peaks are not per-run peaks."""
    if (root / "memory.current").exists():
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        return {
            "cgroup_version": 2,
            "container_usage_bytes": int((root / "memory.current").read_text()),
            "anon_bytes": int(stats.get("anon", 0)),
            "file_cache_bytes": int(stats.get("file", 0)),
            "oom_counters": (root / "memory.events").read_text(),
        }
    root = root / "memory"
    if (root / "memory.usage_in_bytes").exists():
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        return {
            "cgroup_version": 1,
            "container_usage_bytes": int((root / "memory.usage_in_bytes").read_text()),
            "anon_bytes": int(stats.get("total_rss", stats.get("rss", 0))),
            "file_cache_bytes": int(stats.get("total_cache", stats.get("cache", 0))),
            "oom_counters": (root / "memory.failcnt").read_text(),
        }
    return {"cgroup_version": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--action", choices=["smoke", "baseline-small", "baseline-full"], default="smoke")
    parser.add_argument("--max-optimizer-steps", type=int, default=1)
    parser.add_argument("--gpu-hourly-price", type=float, default=5.59)
    parser.add_argument("--judge-input-price-per-million", type=float)
    parser.add_argument("--judge-output-price-per-million", type=float)
    parser.add_argument("--hydra-override", action="append", default=[])
    args = parser.parse_args()
    if args.max_optimizer_steps < 0:
        parser.error("--max-optimizer-steps must be non-negative")
    if args.gpu_hourly_price < 0:
        parser.error("--gpu-hourly-price must be non-negative")
    if (args.judge_input_price_per_million is None) != (args.judge_output_price_per_million is None):
        parser.error("Provide both judge input and output prices, or neither")
    if args.action == "baseline-full" and args.max_optimizer_steps != 251:
        parser.error("baseline-full requires --max-optimizer-steps 251")
    import psutil

    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    audit = root / "audit"
    audit.mkdir()
    project = Path.cwd()
    for name, command in {
        "git-head.txt": ["git", "rev-parse", "HEAD"],
        "git-status.txt": ["git", "status", "--short"],
        "git-diff.patch": ["git", "diff", "HEAD"],
    }.items():
        (root / name).write_bytes(subprocess.check_output(command))
    shutil.copytree(project / "configs", root / "configs")
    shutil.copy2(project / "environments/cuda/uv.lock", root / "cuda-uv.lock")
    versions = {p: importlib.metadata.version(p) for p in ["rllm", "rllm-model-gateway", "verl", "vllm", "torch", "flash-attn", "numpy", "transformers", "peft", "ray", "tensordict"]}
    (root / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    check = subprocess.run(["uv", "pip", "check", "--python", str(Path(os.environ["FINQA_GPU_ROOT"]) / "venv/bin/python")], capture_output=True, text=True)
    (root / "pip-check.log").write_text(check.stdout + check.stderr)
    env = os.environ.copy()
    env["FINQA_SMOKE_AUDIT_DIR"] = str(audit)
    env["PYTHONPATH"] = str(project / "scripts/smoke_audit") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["FINQA_MAX_OPTIMIZER_STEPS"] = str(args.max_optimizer_steps)
    env["RAY_TMPDIR"] = "/tmp/finqa-smoke-" + str(os.getpid())
    command = ["bash", "scripts/gpu.sh", args.action, *args.hydra_override]
    start = time.time()
    manifest = {
        "start_unix": start,
        "command": command,
        "cwd": str(project),
        "git_commit": (root / "git-head.txt").read_text().strip(),
        "pip_check_returncode": check.returncode,
        "accepted_exception": "verl 0.8.0 NumPy <2 metadata versus pinned rLLM/vLLM NumPy 2.3.5",
        "max_optimizer_steps": args.max_optimizer_steps,
        "attempts": 1,
        "timeout_seconds": args.timeout_seconds,
        "ray_tmpdir": env["RAY_TMPDIR"],
        "pricing_cny": {
            "gpu_per_hour": args.gpu_hourly_price,
            "judge_input_per_million_tokens": args.judge_input_price_per_million,
            "judge_output_per_million_tokens": args.judge_output_price_per_million,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "memory-before.json").write_text(json.dumps(cgroup_memory(), indent=2) + "\n")
    timed_out = False
    with (root / "console.log").open("w") as console, (root / "resources.jsonl").open("w") as resources:
        process = subprocess.Popen(command, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        (root / "driver.pid").write_text(str(process.pid))
        while process.poll() is None:
            now = time.time()
            row = {"time": now, "elapsed_seconds": now - start}
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            if gpu.returncode == 0:
                row["gpu"] = gpu.stdout.strip()
            row.update(cgroup_memory())
            try:
                children = psutil.Process(process.pid).children(recursive=True)
                row["process_rss_sum_bytes"] = sum(p.memory_info().rss for p in children if p.is_running())
            except psutil.Error:
                pass
            resources.write(json.dumps(row) + "\n")
            resources.flush()
            if now - start >= args.timeout_seconds:
                timed_out = True
                break
            time.sleep(1)
    # Only processes carrying this unique audit directory belong to this run.
    owned = []
    for p in psutil.process_iter():
        try:
            if p.pid != os.getpid() and p.environ().get("FINQA_SMOKE_AUDIT_DIR") == str(audit):
                owned.append(p)
        except psutil.Error:
            pass
    cleanup = [{"pid": p.pid, "name": p.name()} for p in owned if p.is_running()]
    for p in owned:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(owned, timeout=8)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    returncode = process.wait(timeout=15)
    wall_seconds = time.time() - start
    result = {
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "gpu_window_cost_cny": wall_seconds / 3600 * args.gpu_hourly_price,
        "cleanup": cleanup,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "memory-after.json").write_text(json.dumps(cgroup_memory(), indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
