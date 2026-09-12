"""Fail closed unless every required revision-v2 release artifact is present."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_data import load_canonical, validate_canonical
from drl.experiment_manifest import load_manifest

ROOT = Path(__file__).resolve().parent.parent


def require(path: Path, problems: list[str]) -> None:
    if not path.exists():
        problems.append(f"missing {path.relative_to(ROOT)}")


def main(manifest_path: Path, artifacts: Path) -> None:
    manifest = load_manifest(manifest_path)
    problems: list[str] = []
    canonical_root = ROOT / "data" / "canonical_v2"
    dataset_manifest_path = canonical_root / "manifest.json"
    require(dataset_manifest_path, problems)
    if dataset_manifest_path.exists():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest.get("status") != "complete":
            problems.append("canonical dataset manifest is not complete")
        frame = load_canonical(canonical_root / "records")
        report = validate_canonical(frame, complete=True)
        problems.extend(report.errors)

    require(artifacts / "conformance" / "reference_results.csv", problems)
    external = ROOT / "standards" / "external_reference_outputs.csv"
    require(external, problems)
    if external.exists():
        check = pd.read_csv(external)
        required = {"38901_uma_los_100m_3p5ghz", "38901_uma_nlos_500m_3p5ghz", "36777_uma_av_los_1km_2ghz", "38811_basic_600km_12ghz_plus_clutter"}
        required_columns = {"id", "external_db", "source", "source_commit", "status", "notes"}
        if set(check.get("id", [])) != required or not required_columns <= set(check.columns):
            problems.append("external reference output does not cover all required vectors")
        else:
            by_id = check.set_index("id")
            externally_required = required - {"36777_uma_av_los_1km_2ghz"}
            for vector_id in externally_required:
                row = by_id.loc[vector_id]
                if row["status"] != "validated" or pd.isna(row["external_db"]):
                    problems.append(f"external reference vector is not validated: {vector_id}")
                if pd.isna(row["source_commit"]):
                    problems.append(f"external reference vector lacks commit provenance: {vector_id}")
            uav = by_id.loc["36777_uma_av_los_1km_2ghz"]
            if uav["status"] != "not_applicable" or pd.isna(uav["notes"]):
                problems.append("TR 36.777 external non-applicability is not documented")

    require(artifacts / "development" / "ppo_critic" / "selected_configuration.json", problems)
    for band in manifest["bands"]:
        for algorithm in [*manifest["core"]["algorithms"], "cql"]:
            seeds = manifest["core"]["training_seeds"] if algorithm != "cql" else manifest["cql"]["training_seeds"]
            ext = "model.zip" if algorithm != "cql" else "model.d3"
            for seed in seeds:
                root = artifacts / ("core" if algorithm != "cql" else "cql") / band / algorithm / f"seed_{seed}"
                if algorithm == "cql":
                    root = artifacts / "cql" / band / f"seed_{seed}"
                require(root / ext, problems)
                require(root / "metadata.json", problems)
    for name in ("paired_test_results.csv", "seed_level_summary.csv", "seed_aware_statistics.csv"):
        require(artifacts / "evaluation" / name, problems)
    for name in ("paired_geometry_results.csv", "geometry_difficulty.csv"):
        require(artifacts / "geometry" / name, problems)
    require(artifacts / "sensitivity" / "raw_kpi_pareto.csv", problems)
    require(ROOT / "manuscript" / "revision_v2" / "Handover_Aware_DRL_TN_NTN.docx", problems)

    if problems:
        raise SystemExit("REVISION V2 RELEASE BLOCKED\n- " + "\n- ".join(problems))
    print("REVISION V2 RELEASE GATE PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments" / "revision_v2.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "revision_v2")
    args = parser.parse_args()
    main(args.manifest, args.artifacts)
