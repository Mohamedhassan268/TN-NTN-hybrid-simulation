"""Build frozen out-of-distribution geometry banks for revision-v2 testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.canonical_data import validate_canonical
from drl.scenarios import BANDS, GEOMETRY_REGIMES, ScenarioSpec, generate_scenario

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "canonical_v2" / "geometry_suites"


def build(output_root: Path, count: int, base_seed: int, resume: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for regime_index, regime in enumerate(GEOMETRY_REGIMES):
        if regime == "nominal":
            continue
        output = output_root / regime
        if output.exists() and not resume:
            raise FileExistsError(f"refusing to overwrite {output}; use --resume")
        output.mkdir(parents=True, exist_ok=True)
        records = output / "records.parquet"
        if records.exists() and resume:
            print(f"skip complete {regime}")
            continue
        frames = []
        for index in range(count):
            spec = ScenarioSpec(
                trajectory_id=f"{regime}_v2_{index:04d}",
                scenario_id=f"{regime}_scenario_v2_{index:04d}",
                ue_id=f"{regime}_ue_v2_{index:04d}",
                split="test",
                seed=base_seed + regime_index * 10_000 + index,
                geometry_regime=regime,
            )
            frames.append(generate_scenario(spec, bands=BANDS))
        frame = pd.concat(frames, ignore_index=True)
        report = validate_canonical(frame)
        if not report.valid:
            raise AssertionError("; ".join(report.errors))
        frame.to_parquet(records, index=False, compression="zstd")
        (output / "manifest.json").write_text(json.dumps({
            "geometry_regime": regime, "trajectories": count,
            "base_seed": base_seed, "bands": list(BANDS), "rows": len(frame),
            "status": "complete",
        }, indent=2), encoding="utf-8")
        print(f"wrote {records}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20270912)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build(args.output_root, args.count, args.base_seed, args.resume)
