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
    external = {}
    if external_path and external_path.exists():
        external_frame = pd.read_csv(external_path)
        external = external_frame.set_index("id")["external_db"].to_dict()
    rows = []
    for vector in vectors:
        actual = float(IMPLEMENTATIONS[vector["implementation"]](**vector["arguments"]))
        error = abs(actual - float(vector["expected_db"]))
        external_error = (
            abs(actual - float(external[vector["id"]])) if vector["id"] in external else None
        )
        rows.append({
            "id": vector["id"], "actual_db": actual,
            "source_expected_db": vector["expected_db"], "source_error_db": error,
            "tolerance_db": vector["tolerance_db"], "source_pass": error <= vector["tolerance_db"],
            "external_db": external.get(vector["id"]), "external_error_db": external_error,
            "external_pass": None if external_error is None else external_error <= vector["tolerance_db"],
        })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if not result["source_pass"].all():
        raise SystemExit("physical reference-vector conformance failed; standards claim is blocked")
    if external and not result.loc[result["external_db"].notna(), "external_pass"].all():
        raise SystemExit("external simulator cross-check failed; affected realism claim is blocked")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=ROOT / "standards" / "reference_vectors.json")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "revision_v2" / "conformance" / "reference_results.csv")
    args = parser.parse_args()
    print(run(args.vectors, args.output, args.external).to_string(index=False))
