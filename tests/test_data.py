import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from finqa_agentic_rl import prepare_finqa_data as prep


@pytest.fixture
def sample_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    tables = data / "company_tables/example"
    tables.mkdir(parents=True)
    (tables / "tables_cleaned_all_companies.json").write_text("{}")
    row = {
        "user_query": "For example, revenue?",
        "answer": "5",
        "company": "example",
        "id": 7,
        "question_type": "single_table",
        "question": "revenue?",
        "table_name": '["income"]',
        "columns_used_json": '["2024"]',
        "rows_used_json": '["Revenue"]',
        "explanation": "read table",
    }
    for split in ("train", "val", "test"):
        path = data / f"{split}_finqa.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        monkeypatch.setattr(prep.C, f"{split.upper()}_QUESTIONS_PATH", path)
    monkeypatch.setattr(prep.C, "DATA_DIR", data)
    return data


def test_csv_preprocessing_registry_and_verl_roundtrip(sample_data, isolated_registry):
    splits = prep.prepare_finqa_data()
    assert [len(split) for split in splits] == [1, 1, 1]
    for split in ("train", "val", "test"):
        dataset = isolated_registry.load_dataset("finqa", split)
        row = dataset[0]
        assert row["question"] == "For example, revenue?"
        assert row["question_id"] == "7"
        assert list(row["table_name"]) == ["income"]
        assert list(row["columns_used"]) == ["2024"]
        assert list(row["rows_used"]) == ["Revenue"]
        assert Path(dataset.get_verl_data_path()).is_file()
    prep.prepare_finqa_data()
    assert len(isolated_registry.load_dataset("finqa", "train")) == 1


def test_partial_data_rejected_before_registration(sample_data, isolated_registry):
    (sample_data / "test_finqa.csv").unlink()
    with pytest.raises(FileNotFoundError):
        prep.prepare_finqa_data()
    assert not Path(isolated_registry._REGISTRY_FILE).exists()


def test_archive_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.C, "DATA_DIR", tmp_path / "destination")
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        prep.download_data(archive, offline=True)
    assert not prep.C.DATA_DIR.exists()


@pytest.mark.parametrize("name", ["../escaped", "/escaped", "data/../../escaped"])
def test_archive_path_escape_rejected(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo(name)
        entry.size = 1
        tar.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="archive member"):
        prep._extract_archive(archive, tmp_path / "extract")


def test_prepared_data_detects_changed_files(sample_data):
    files = {str(p.relative_to(sample_data)): prep._sha256(p) for p in sample_data.rglob("*") if p.is_file()}
    (sample_data / prep.MANIFEST_NAME).write_text(json.dumps({"source": prep.SOURCE, "files": files}))
    (sample_data / "train_finqa.csv").write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        prep.verify_prepared_data(sample_data)


@pytest.mark.data
def test_official_data_and_registration(isolated_registry, monkeypatch):
    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / prep.MANIFEST_NAME).exists():
        pytest.skip("Run finqa-prepare to enable the real-data integration test")
    monkeypatch.setattr(prep.C, "DATA_DIR", data)
    for split in ("train", "val", "test"):
        monkeypatch.setattr(prep.C, f"{split.upper()}_QUESTIONS_PATH", data / f"{split}_finqa.csv")
    prep.verify_prepared_data(data)
    datasets = prep.prepare_finqa_data()
    assert [len(d) for d in datasets] == [4030, 522, 558]
    for dataset in datasets:
        assert len(isolated_registry.load_dataset("finqa", dataset.split)) == len(dataset)
        assert Path(dataset.get_verl_data_path()).exists()


@pytest.mark.data
def test_official_table_preload_metadata_and_sql(monkeypatch):
    from finqa_agentic_rl import finqa_tools as tools

    tables = Path(__file__).resolve().parents[1] / "data/company_tables"
    if not tables.exists():
        pytest.skip("Run finqa-prepare to enable real table loading")
    monkeypatch.setattr(tools, "TABLES_ROOT", tables)
    monkeypatch.setattr(tools, "_PRELOADED", False)
    monkeypatch.setattr(tools, "_DB_CONN", None)
    for cache in ("_TABLE_INFO_CACHE", "_COMPANY_TABLES", "_SQL_NAMES", "_TABLE_NAME_MAP"):
        monkeypatch.setattr(tools, cache, {})
    tools._preload_all()
    try:
        assert len(tools._COMPANY_TABLES) == 207
        assert len(tools._TABLE_INFO_CACHE) == 6923
        assert len(tools._SQL_NAMES) == 6110
        # Every question in the pinned archive must still have queryable tables.
        for split in ("train", "val", "test"):
            for row in pd.read_csv(tables.parent / f"{split}_finqa.csv").to_dict("records"):
                company = tools._normalize_company(row["company"])
                for name in prep._parse_json_list(row["table_name"]):
                    assert (company, tools._normalize_table(company, name)) in tools._SQL_NAMES
        company = next(iter(tools._COMPANY_TABLES))
        table = tools.get_table_names(company)[0]
        assert "column_names" in json.loads(tools.get_table_info(company, table))
        result = tools.sql_query(company, table, f'SELECT COUNT(*) AS n FROM "{table}"')
        assert json.loads(result)[0]["n"] > 0
    finally:
        tools._DB_CONN.close()
