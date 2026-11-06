"""Run the existing smoke command ONCE, recording resources and bounded diagnostics."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
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
    versions = {p: importlib.metadata.version(p) for p in ["rllm", "rllm-model-gateway", "verl", "vllm", "torch", "flash-attn", "numpy", "transformers", "peft", "ray"]}
    (root / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    check = subprocess.run(["uv", "pip", "check", "--python", str(Path(os.environ["FINQA_GPU_ROOT"]) / "venv/bin/python")], capture_output=True, text=True)
    (root / "pip-check.log").write_text(check.stdout + check.stderr)
    env = os.environ.copy()
    env["FINQA_SMOKE_AUDIT_DIR"] = str(audit)
    env["PYTHONPATH"] = str(project / "scripts/smoke_audit") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_TMPDIR"] = "/tmp/finqa-smoke-" + str(os.getpid())
    command = ["bash", "scripts/gpu.sh", "smoke"]
    start = time.time()
    manifest = {
        "start_unix": start,
        "command": command,
        "cwd": str(project),
        "git_commit": (root / "git-head.txt").read_text().strip(),
        "pip_check_returncode": check.returncode,
        "accepted_exception": "verl 0.8.0 NumPy <2 metadata versus pinned rLLM/vLLM NumPy 2.3.5",
        "max_optimizer_steps": 1,
        "attempts": 1,
        "timeout_seconds": args.timeout_seconds,
        "ray_tmpdir": env["RAY_TMPDIR"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cgroup = Path("/sys/fs/cgroup")
    before_oom = (cgroup / "memory.events").read_text() if (cgroup / "memory.events").exists() else ""
    (root / "memory-events-before.txt").write_text(before_oom)
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
            for name in ("memory.current", "memory.peak"):
                path = cgroup / name
                if path.exists():
                    row[name] = int(path.read_text())
            if (cgroup / "memory.stat").exists():
                memory = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
                row["anon_bytes"] = int(memory.get("anon", 0))
                row["file_cache_bytes"] = int(memory.get("file", 0))
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
    result = {"returncode": returncode, "timed_out": timed_out, "wall_seconds": time.time() - start, "cleanup": cleanup}
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if (cgroup / "memory.events").exists():
        (root / "memory-events-after.txt").write_text((cgroup / "memory.events").read_text())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
