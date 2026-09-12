"""Build the immutable revision-v2 matched trajectory dataset.

Examples:
  python scripts/build_canonical_dataset.py --total 3 --output-dir tmp/canonical-smoke
  python scripts/build_canonical_dataset.py --total 1200

The default output is separate from every legacy dataset. Existing output is
never overwritten; pass ``--resume`` to retain complete batch files and fill in
missing ones with the same manifest configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_data import validate_canonical
from drl.reference_utility import fit_reference_params, save_reference_params
from drl.scenarios import BANDS, DATASET_VERSION, canonical_specs, generate_scenario, schema_definition

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "canonical_v2"


def _config(total: int, batch_size: int, base_seed: int) -> dict:
    payload = {
        "dataset_version": DATASET_VERSION,
        "total_trajectories": total,
        "batch_size": batch_size,
        "base_seed": base_seed,
        "bands": list(BANDS),
        "decision_interval_s": 10.0,
        "episode_steps": 60,
        "split_policy": {"train": [0, 799], "validation": [800, 999], "test": [1000, 1199]},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build(output_dir: Path, total: int, batch_size: int, base_seed: int, resume: bool) -> None:
    output_dir = output_dir.resolve()
    data_dir = output_dir / "records"
    config = _config(total, batch_size, base_seed)
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output_dir}; use a new path or --resume")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != config["configuration_sha256"]:
            raise RuntimeError("resume manifest does not match requested configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, config)
    _write_json(output_dir / "schema.json", schema_definition())

    specs = canonical_specs(total=total, base_seed=base_seed)
    split_map = {spec.trajectory_id: spec.split for spec in specs}
    _write_json(output_dir / "splits.json", split_map)

    for start in range(0, len(specs), batch_size):
        end = min(start + batch_size, len(specs))
        path = data_dir / f"trajectories_{start:04d}_{end - 1:04d}.parquet"
        if resume and path.exists():
            print(f"skip complete batch {path.name}", flush=True)
            continue
        frames = []
        for index, spec in enumerate(specs[start:end], start=start):
            frames.append(generate_scenario(spec))
            print(f"generated {index + 1}/{len(specs)} {spec.trajectory_id}", flush=True)
        batch = pd.concat(frames, ignore_index=True)
        report = validate_canonical(batch)
        if not report.valid:
            raise AssertionError("; ".join(report.errors))
        tmp_path = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(batch, preserve_index=False), tmp_path, compression="zstd")
        tmp_path.replace(path)

    complete = pd.read_parquet(data_dir)
    report = validate_canonical(complete, complete=total == 1200)
    if not report.valid:
        raise AssertionError("; ".join(report.errors))
    params = fit_reference_params(complete[complete["split"] == "train"])
    save_reference_params(params, output_dir / "reference_params.json")
    manifest = {
        **config,
        "rows": len(complete),
        "trajectories": complete["trajectory_id"].nunique(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "status": "complete",
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260912)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.output_dir, args.total, args.batch_size, args.base_seed, args.resume)
