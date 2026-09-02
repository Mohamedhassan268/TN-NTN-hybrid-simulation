"""Phase 5: multi-seed training of PPO and DQN on the redesigned condition-aware
NetworkSelectionEnv (drl/env.py), evaluated against the classical telecom baselines
(drl/baselines.py) via the shared harness (drl/evaluation.py). Replaces the earlier
single-seed/5-episode scripts/train_ppo.py evaluation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.baselines import GreedyPolicy, HysteresisPolicy, RandomPolicy, make_oracle_policy
from drl.env import HANDOVER_PENALTY, NetworkSelectionEnv
from drl.evaluation import SB3Policy, evaluate_policy, rollout_episode, summarize_eval
from drl.reward import RewardWeights

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
FIG_DIR = ROOT / "drl" / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 100_000
EPISODE_STEPS = 60
N_ENVS = 4
N_EVAL_EPISODES = 200


def make_env():
    return NetworkSelectionEnv(CSV_PATH, PARAMS_PATH, episode_steps=EPISODE_STEPS)


def train_ppo(seed: int) -> PPO:
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=seed)
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=1, n_steps=512, batch_size=256)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    return model


def train_dqn(seed: int) -> DQN:
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=seed)
    model = DQN("MlpPolicy", vec_env, seed=seed, verbose=1, learning_starts=1000,
                buffer_size=50_000, train_freq=4, target_update_interval=1000)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    return model


def eval_agent_multi_seed(train_fn, label: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        print(f"  training {label} seed={seed} ...", flush=True)
        model = train_fn(seed)
        print(f"  evaluating {label} seed={seed} ({N_EVAL_EPISODES} episodes)...", flush=True)
        policy = SB3Policy(model)
        eval_df = evaluate_policy(make_env, policy, n_episodes=N_EVAL_EPISODES, base_seed=20_000 + seed * 1000,
                                   progress_label=f"{label} seed={seed} eval")
        eval_df["policy"] = label
        eval_df["train_seed"] = seed
        frames.append(eval_df)
    return pd.concat(frames, ignore_index=True)


def eval_stateless_baseline(policy_factory, label: str) -> pd.DataFrame:
    print(f"  evaluating {label} ({N_EVAL_EPISODES} episodes)...", flush=True)
    policy = policy_factory()
    eval_df = evaluate_policy(make_env, policy, n_episodes=N_EVAL_EPISODES, progress_label=label)
    eval_df["policy"] = label
    eval_df["train_seed"] = -1
    return eval_df


def eval_oracle(reward_params: dict) -> pd.DataFrame:
    print(f"  evaluating oracle ({N_EVAL_EPISODES} episodes, DP-planned)...", flush=True)
    rows = []
    for ep in range(N_EVAL_EPISODES):
        seed = 10_000 + ep
        plan_env = make_env()
        plan_env.reset(seed=seed)
        policy = make_oracle_policy(plan_env, seed, reward_params, RewardWeights(), HANDOVER_PENALTY)
        if policy is None:
            continue
        row = rollout_episode(make_env(), policy, seed=seed)
        row["policy"] = "oracle"
        row["train_seed"] = -1
        rows.append(row)
        if (ep + 1) % 20 == 0:
            print(f"    oracle: {ep + 1}/{N_EVAL_EPISODES} episodes", flush=True)
    return pd.DataFrame(rows)


def main():
    reward_params = json.loads(PARAMS_PATH.read_text())

    print("evaluating baselines...")
    all_frames = [
        eval_stateless_baseline(lambda: RandomPolicy(seed=0), "random"),
        eval_stateless_baseline(GreedyPolicy, "greedy"),
        eval_stateless_baseline(HysteresisPolicy, "hysteresis"),
        eval_oracle(reward_params),
    ]

    print("training + evaluating PPO...")
    all_frames.append(eval_agent_multi_seed(train_ppo, "ppo"))
    print("training + evaluating DQN...")
    all_frames.append(eval_agent_multi_seed(train_dqn, "dqn"))

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.to_csv(FIG_DIR / "phase5_full_eval_results.csv", index=False)

    print("\nsummary (mean [95% CI]) by policy:")
    summary_rows = []
    for policy_name, group in full_df.groupby("policy"):
        s = summarize_eval(group)
        summary_rows.append({
            "policy": policy_name, "n": len(group),
            "reward_mean": s["total_reward"]["mean"],
            "reward_ci_low": s["total_reward"]["ci_low"], "reward_ci_high": s["total_reward"]["ci_high"],
            "switch_rate_mean": s["switch_rate"]["mean"],
            "outage_rate_mean": s["outage_rate"]["mean"],
        })
        print(f"  {policy_name:12s} reward={s['total_reward']['mean']:7.2f} "
              f"[{s['total_reward']['ci_low']:7.2f}, {s['total_reward']['ci_high']:7.2f}]  "
              f"switch_rate={s['switch_rate']['mean']:.3f}  outage_rate={s['outage_rate']['mean']:.3f}")

    pd.DataFrame(summary_rows).to_csv(FIG_DIR / "phase5_summary.csv", index=False)
    print(f"\nfull results: {FIG_DIR / 'phase5_full_eval_results.csv'}")
    print(f"summary: {FIG_DIR / 'phase5_summary.csv'}")


if __name__ == "__main__":
    main()
