"""Seed-first hierarchical inference for the paired revision-v2 test campaign."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drl.canonical_evaluation import hierarchical_paired_bootstrap

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "artifacts" / "revision_v2" / "evaluation" / "paired_test_results.csv"
COMPARISONS = [
    ("ppo", "dqn"), ("ppo", "cql"), ("ppo", "hysteresis"),
    ("ppo", "oracle"), ("dqn", "hysteresis"), ("cql", "hysteresis"),
]
METRICS = [
    "reference_utility", "raw_qoe", "throughput_mbps", "latency_ms",
    "ber", "bler", "packet_loss_pct", "switch_rate", "invalid_rate",
]


def main(input_path: Path, output_path: Path):
    frame = pd.read_csv(input_path)
    rows = []
    for band, band_frame in frame.groupby("band"):
        for policy_a, policy_b in COMPARISONS:
            for metric in METRICS:
                result = hierarchical_paired_bootstrap(
                    band_frame, policy_a, policy_b, metric, n_boot=10_000, seed=20260912
                )
                rows.append({"band": band, **result})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT.with_name("seed_aware_statistics.csv"))
    args = parser.parse_args()
    main(args.input, args.output)
