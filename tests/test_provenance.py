import ast
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_snapshot_hashes_and_prompt_bytes():
    manifest = json.loads((ROOT / "upstream/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = ROOT / "upstream" / name.removeprefix("cookbooks/")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    for prompt in (ROOT / "upstream/finqa/prompts").glob("*.txt"):
        assert (ROOT / "src/finqa_agentic_rl/prompts" / prompt.name).read_bytes() == prompt.read_bytes()


def test_agent_tools_and_evaluator_behavior_code_matches_upstream():
    # An AST comparison permits import relocation and formatting, but catches
    # accidental changes to the baseline's control flow, constants or rubrics.
    for name in ("finqa_flow.py", "finqa_tools.py", "finqa_eval.py"):
        upstream = ast.parse((ROOT / "upstream/finqa" / name).read_text())
        migrated = ast.parse((ROOT / "src/finqa_agentic_rl" / name).read_text())
        for tree in (upstream, migrated):
            tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        assert ast.dump(upstream) == ast.dump(migrated), name


def test_training_overrides_preserved_and_shell_syntax():
    for backend in ("tinker", "verl", "fireworks"):
        original = (ROOT / f"upstream/finqa/train_{backend}.sh").read_text()
        migrated = (ROOT / f"scripts/train_{backend}.sh").read_text()
        assert (
            original[original.index("set -euo pipefail") :].replace("python -u train.py", '"${FINQA_PYTHON:-python}" -u -m finqa_agentic_rl.train') == migrated[migrated.index("set -euo pipefail") :]
        )
        subprocess.run(["bash", "-n", str(ROOT / f"scripts/train_{backend}.sh")], check=True)


def test_rllm_and_gateway_pins_match_for_both_environments():
    expected = json.loads((ROOT / "upstream/manifest.json").read_text())["commit"]
    for path in (ROOT / "pyproject.toml", ROOT / "environments/cuda/pyproject.toml"):
        config = tomllib.loads(path.read_text())
        sources = config["tool"]["uv"]["sources"]
        assert sources["rllm"]["rev"] == sources["rllm-model-gateway"]["rev"] == expected
        assert sources["rllm-model-gateway"]["subdirectory"] == "rllm-model-gateway"
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert not any("[verl]" in dependency for dependency in config["project"]["dependencies"])


def test_launcher_passes_user_overrides_last(tmp_path):
    stub = tmp_path / "python-stub"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    env = dict(os.environ, FINQA_PYTHON=str(stub))
    run = subprocess.run(["bash", str(ROOT / "scripts/train_verl.sh"), "trainer.n_gpus_per_node=1", "--cfg", "job"], env=env, text=True, capture_output=True, check=True)
    args = run.stdout.splitlines()
    assert args[:3] == ["-u", "-m", "finqa_agentic_rl.train"]
    assert "algorithm.adv_estimator=grpo" in args
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507" in args
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes" in args
    assert "trainer.n_gpus_per_node=8" in args
    assert args[-3:] == ["trainer.n_gpus_per_node=1", "--cfg", "job"]
