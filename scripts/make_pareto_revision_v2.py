"""Derive the raw-KPI non-dominated frontier from completed sensitivity runs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "artifacts" / "revision_v2" / "sensitivity" / "paired_sensitivity_results.csv"
OUTPUT = ROOT / "artifacts" / "revision_v2" / "sensitivity" / "raw_kpi_pareto.csv"


def nondominated(group: pd.DataFrame) -> pd.Series:
    # Convert all objectives to larger-is-better orientation.
    values = group[["throughput_mbps", "latency_ms", "ber", "packet_loss_pct"]].copy()
    values[["latency_ms", "ber", "packet_loss_pct"]] *= -1
    matrix = values.to_numpy()
    keep = []
    for index, candidate in enumerate(matrix):
        dominated = any(
            (other >= candidate).all() and (other > candidate).any()
            for other_index, other in enumerate(matrix) if other_index != index
        )
        keep.append(not dominated)
    return pd.Series(keep, index=group.index)


if __name__ == "__main__":
    frame = pd.read_csv(SOURCE)
    summary = frame.groupby(
        ["band", "condition", "reward_profile", "handover_penalty"], as_index=False
    ).mean(numeric_only=True)
    summary["pareto_optimal"] = summary.groupby("band", group_keys=False).apply(
        nondominated, include_groups=False
    ).sort_index()
    summary.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}")
