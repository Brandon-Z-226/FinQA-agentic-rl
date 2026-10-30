# Derived from rLLM cookbooks/finqa at 4e828c81680e5a85b638ceedeaf09d1e90109ca8.
# Apache-2.0; see licenses/rllm-LICENSE and THIRD_PARTY_NOTICES.md.
# Modified: package imports and project integration; see docs/migration.md.
"""Download and register the rLLM/finqa dataset.

Pulls the pinned tarball from Hugging Face, extracts the company-tables tree
(~6,900 tables across 207 companies) into ``FINQA_DATA_DIR`` (default ./data),
and registers the train / val / test question splits with rLLM's
``DatasetRegistry``.

Usage::

    finqa-prepare
    finqa-prepare --offline --archive .cache/data.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from rllm.data.dataset import DatasetRegistry

from . import finqa_constants as C

SOURCE = json.loads((Path(__file__).parent / "data_source.json").read_text())
HF_REPO_ID = SOURCE["repo_id"]
HF_FILENAME = SOURCE["filename"]
MANIFEST_NAME = ".finqa-source.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_data(data_dir: Path) -> dict[str, int]:
    """Validate all splits and company metadata before registering anything."""
    required = {"user_query", "answer", "company", "id", "question_type", "question", "explanation"}
    counts = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_finqa.csv"
        frame = pd.read_csv(path)
        if missing := required - set(frame.columns):
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame.empty or frame[["user_query", "answer", "company", "id"]].isna().any().any():
            raise ValueError(f"{path}: empty split or missing required values")
        counts[split] = len(frame)
    tables = list((data_dir / "company_tables").glob("*/tables_cleaned_all_companies.json"))
    if not tables:
        raise ValueError(f"No company metadata found under {data_dir}")
    return counts


def verify_prepared_data(data_dir: Path) -> None:
    """Reject incomplete or changed cached data instead of silently skipping it."""
    manifest = json.loads((data_dir / MANIFEST_NAME).read_text())
    if manifest.get("source") != SOURCE:
        raise ValueError("Prepared data source differs from the pinned source")
    for name, digest in manifest["files"].items():
        path = data_dir / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Prepared data is missing or changed: {path}")
    if validate_data(data_dir) != SOURCE["splits"]:
        raise ValueError("Prepared data split sizes differ from the pinned source")


def _extract_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
                raise ValueError(f"Unexpected archive member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Links and special files are not supported: {member.name}")
        tar.extractall(destination, filter="data")


def download_data(archive: Path | None = None, *, offline: bool = False) -> None:
    """Fetch the pinned archive, verify its hash, then extract atomically."""
    data_dir = C.DATA_DIR
    if data_dir.exists():
        verify_prepared_data(data_dir)
        print(f"Verified existing data at {data_dir}")
        return

    if archive is None:
        archive = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset", revision=SOURCE["revision"], local_files_only=offline))
    if _sha256(archive) != SOURCE["sha256"]:
        raise ValueError(f"Archive SHA256 does not match the pinned dataset: {archive}")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finqa-extract-", dir=data_dir.parent) as staging:
        staging = Path(staging)
        _extract_archive(archive, staging)
        extracted = staging / "data"
        if validate_data(extracted) != SOURCE["splits"]:
            raise ValueError("Archive split sizes differ from the pinned source")
        files = {str(p.relative_to(extracted)): _sha256(p) for p in sorted(extracted.rglob("*")) if p.is_file()}
        (extracted / MANIFEST_NAME).write_text(json.dumps({"source": SOURCE, "files": files}, indent=2) + "\n")
        extracted.rename(data_dir)
    print(f"Prepared verified data at {data_dir}")


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = s
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


def prepare_finqa_data():
    validate_data(C.DATA_DIR)
    train_df = pd.read_csv(C.TRAIN_QUESTIONS_PATH)
    val_df = pd.read_csv(C.VAL_QUESTIONS_PATH)
    test_df = pd.read_csv(C.TEST_QUESTIONS_PATH)

    def preprocess_fn(example: dict) -> dict:
        return {
            "question": example["user_query"],
            "ground_truth": example["answer"],
            "data_source": "finqa",
            "company": example["company"],
            "question_id": str(example["id"]),
            "question_type": example["question_type"],
            "core_question": example["question"],
            "table_name": _parse_json_list(example.get("table_name")),
            "columns_used": _parse_json_list(example.get("columns_used_json")),
            "rows_used": _parse_json_list(example.get("rows_used_json")),
            "explanation": example["explanation"],
        }

    train = [preprocess_fn(row) for _, row in train_df.iterrows()]
    val = [preprocess_fn(row) for _, row in val_df.iterrows()]
    test = [preprocess_fn(row) for _, row in test_df.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("finqa", train, "train")
    val_dataset = DatasetRegistry.register_dataset("finqa", val, "val")
    test_dataset = DatasetRegistry.register_dataset("finqa", test, "test")
    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare the pinned official rLLM FinQA data; no model or GPU required.")
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded official data.tar.gz")
    parser.add_argument("--offline", action="store_true", help="Use only local data/Hugging Face cache")
    args = parser.parse_args()
    download_data(args.archive, offline=args.offline)
    train, val, test = prepare_finqa_data()
    print(f"Train: {len(train.get_data())} rows")
    print(f"Val:   {len(val.get_data())} rows")
    print(f"Test:  {len(test.get_data())} rows")


if __name__ == "__main__":
    main()
