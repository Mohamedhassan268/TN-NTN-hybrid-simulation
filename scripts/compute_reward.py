"""Fits reward normalization on Hybrid_Network_TN_NTN_Final.csv, computes the
Phase-1 composite QoE reward, and writes the fitted normalization params for
reuse at inference time.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.reward import RewardWeights, compute_reward, fit_normalization

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("TN_NTN_DATASET", "ku")  # "ka" selects the synthetic Ka-band dataset
if DATASET == "ka":
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Ka.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params_ka.json"
else:
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"


def main():
    df = pd.read_csv(CSV_PATH)

    params = fit_normalization(df)
    PARAMS_PATH.write_text(json.dumps(params, indent=2))

    reward = compute_reward(df, params, RewardWeights())

    assert reward.notna().all(), "reward contains NaN"
    assert reward.between(0.0, 1.0).all(), "reward outside [0, 1]"

    print(f"rows: {len(df)}")
    print(f"reward range: [{reward.min():.4f}, {reward.max():.4f}], mean={reward.mean():.4f}")
    print()
    print("mean reward by network_type:")
    print(reward.groupby(df["network_type"]).mean().sort_values(ascending=False).to_string())
    print()
    print(f"normalization params written to {PARAMS_PATH}")


if __name__ == "__main__":
    main()
