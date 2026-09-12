from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.run_physical_conformance import run


ROOT = Path(__file__).resolve().parent.parent


def write_external(path: Path, ntn_value: float = 172.09665) -> None:
    rows = [
        ("38901_uma_los_100m_3p5ghz", 82.881360887, "LLSim5G", "a", "validated", "direct"),
        ("38901_uma_nlos_500m_3p5ghz", 129.897108656, "LLSim5G", "a", "validated", "direct"),
        ("36777_uma_av_los_1km_2ghz", "", "", "", "not_applicable", "not implemented"),
        ("38811_basic_600km_12ghz_plus_clutter", ntn_value, "OpenNTN", "b", "validated", "direct"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "external_db", "source", "source_commit", "status", "notes"])
        writer.writerows(rows)


def test_external_crosscheck_requires_provenanced_statuses(tmp_path):
    external = tmp_path / "external.csv"
    write_external(external)
    result = run(ROOT / "standards" / "reference_vectors.json", tmp_path / "result.csv", external)
    assert result.loc[result["external_status"] == "validated", "external_pass"].all()
    assert result.loc[result["external_status"] == "not_applicable", "external_db"].isna().all()


def test_external_crosscheck_fails_closed_on_disagreement(tmp_path):
    external = tmp_path / "external.csv"
    write_external(external, ntn_value=180.0)
    with pytest.raises(SystemExit, match="cross-check failed"):
        run(ROOT / "standards" / "reference_vectors.json", tmp_path / "result.csv", external)
