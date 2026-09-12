"""Execute deterministic standard-equation vectors and optional external checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from noise_models.channels.pathloss import ntn_basic_path_loss_db, uma_av_path_loss_db, uma_path_loss_db

ROOT = Path(__file__).resolve().parent.parent
IMPLEMENTATIONS = {
    "uma_path_loss_db": uma_path_loss_db,
    "uma_av_path_loss_db": uma_av_path_loss_db,
    "ntn_basic_path_loss_db": ntn_basic_path_loss_db,
}


def run(vectors_path: Path, output_path: Path, external_path: Path | None = None) -> pd.DataFrame:
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    external: dict[str, dict] = {}
    if external_path and external_path.exists():
        external_frame = pd.read_csv(external_path)
        required_columns = {"id", "external_db", "source", "source_commit", "status", "notes"}
        missing_columns = required_columns - set(external_frame.columns)
        if missing_columns:
            raise SystemExit(f"external reference file lacks columns: {sorted(missing_columns)}")
        if external_frame["id"].duplicated().any():
            raise SystemExit("external reference file contains duplicate vector ids")
        expected_ids = {vector["id"] for vector in vectors}
        if set(external_frame["id"]) != expected_ids:
            raise SystemExit("external reference file does not exactly cover the reference vectors")
        external = external_frame.set_index("id").to_dict(orient="index")
    rows = []
    for vector in vectors:
        actual = float(IMPLEMENTATIONS[vector["implementation"]](**vector["arguments"]))
        error = abs(actual - float(vector["expected_db"]))
        external_row = external.get(vector["id"], {})
        external_value = external_row.get("external_db")
        external_status = external_row.get("status")
        if external_status == "validated" and pd.isna(external_value):
            raise SystemExit(f"validated external vector lacks a numeric value: {vector['id']}")
        external_error = (
            abs(actual - float(external_value))
            if external_status == "validated" else None
        )
        rows.append({
            "id": vector["id"], "actual_db": actual,
            "source_expected_db": vector["expected_db"], "source_error_db": error,
            "tolerance_db": vector["tolerance_db"], "source_pass": error <= vector["tolerance_db"],
            "external_db": external_value, "external_error_db": external_error,
            "external_pass": None if external_error is None else external_error <= vector["tolerance_db"],
            "external_status": external_status,
            "external_source": external_row.get("source"),
            "external_commit": external_row.get("source_commit"),
        })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if not result["source_pass"].all():
        raise SystemExit("physical reference-vector conformance failed; standards claim is blocked")
    if external:
        validated = result["external_status"] == "validated"
        if not validated.any() or not result.loc[validated, "external_pass"].all():
            raise SystemExit("external simulator cross-check failed; affected realism claim is blocked")
        unapplied = result["external_status"] == "not_applicable"
        if not (validated | unapplied).all():
            raise SystemExit("external simulator cross-check contains unresolved vectors")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=ROOT / "standards" / "reference_vectors.json")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "revision_v2" / "conformance" / "reference_results.csv")
    args = parser.parse_args()
    print(run(args.vectors, args.output, args.external).to_string(index=False))
