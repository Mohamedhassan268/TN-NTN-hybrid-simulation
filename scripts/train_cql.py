"""Phase-1 training: Discrete CQL offline-RL agent for network selection.

State = Area + Available_Networks context (see drl/features.py) - only what's known
before a network is picked. Action = network_type. Reward = Phase-1 composite QoE
(see drl/reward.py). Each row is treated as its own single-step episode: this is a
contextual-bandit problem (no handover cost / sequential coupling yet - that's Phase 2).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

from drl.features import AREAS, NETWORK_TYPES, build_actions, build_state
from drl.reward import RewardWeights, compute_reward, fit_normalization

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "Hybrid_Network_TN_NTN_Final.csv"
PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"

SEED = 0
N_STEPS = 40_000
ALPHA = 0.1  # lower conservative-penalty weight: our state space is tiny (6 states)
             # and every valid action is well-covered per state, so there's little
             # out-of-distribution risk for CQL's conservatism to guard against;
             # default alpha=1.0 over-penalizes and fights convergence to the true
             # empirical optimum here.
MODEL_PATH = ROOT / "drl" / f"cql_network_selector_alpha{ALPHA}.d3"


def main():
    df = pd.read_csv(CSV_PATH)
    params = json.loads(PARAMS_PATH.read_text())
    reward = compute_reward(df, params, RewardWeights()).to_numpy(dtype=np.float32)

    state = build_state(df)
    action = build_actions(df)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    split = int(0.8 * len(df))
    train_idx, test_idx = idx[:split], idx[split:]

    train_dataset = MDPDataset(
        observations=state[train_idx],
        actions=action[train_idx],
        rewards=reward[train_idx],
        terminals=np.ones(len(train_idx), dtype=np.float32),
    )

    cql = DiscreteCQLConfig(batch_size=256, target_update_interval=1000, alpha=ALPHA).create()
    cql.fit(train_dataset, n_steps=N_STEPS, n_steps_per_epoch=2000)
    cql.save(str(MODEL_PATH))

    # --- evaluation: held-out logged-action reward vs. what the training data says
    #     is the best available action per Area (empirical baseline), vs. the
    #     policy's greedy choice per Area.
    test_df = df.iloc[test_idx]
    test_reward = reward[test_idx]
    print(f"held-out logged-action mean reward: {test_reward.mean():.4f}")

    train_df = df.iloc[train_idx].assign(reward=reward[train_idx])
    empirical_best = (
        train_df.groupby(["Area", "network_type"])["reward"].mean().reset_index()
    )

    print("\nper-Area comparison: empirical best (train) vs CQL greedy policy")
    for a in AREAS:
        area_rows = empirical_best[empirical_best["Area"] == a]
        if area_rows.empty:
            continue
        best_row = area_rows.loc[area_rows["reward"].idxmax()]
        area_state = build_state(pd.DataFrame({
            "Area": [a],
            "Available_Networks": [df.loc[df["Area"] == a, "Available_Networks"].iloc[0]],
        }))
        policy_action = cql.predict(area_state)[0]
        policy_network = NETWORK_TYPES[policy_action]
        print(
            f"  {a:9s} empirical-best={best_row['network_type']:10s} "
            f"(mean_reward={best_row['reward']:.4f})   "
            f"CQL-policy={policy_network:10s}"
        )

    print(f"\nmodel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
