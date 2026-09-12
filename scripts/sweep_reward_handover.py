"""Phase 6: reward-weight and handover-cost sensitivity sweep. Two separate 1-D
sweeps (not a full cross product, to keep retraining cost tractable - each point
requires a fresh PPO run since both the reward weights and the handover penalty are
baked into the env): (a) handover penalty at the default balanced reward, (b) reward
weight profile at the default handover penalty. Verifies switch rate responds
monotonically to handover penalty, and reports per-profile behavior honestly
(including any non-monotonic surprises) rather than smoothing over them.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.env import HANDOVER_PENALTY, NetworkSelectionEnv
from drl.evaluation import SB3Policy, evaluate_policy, summarize_eval
from drl.reward import RewardWeights

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("TN_NTN_DATASET", "ku")
if DATASET == "ka":
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Ka.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params_ka.json"
    FIG_DIR = ROOT / "drl" / "figures_ka"
elif DATASET == "s":
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
    FIG_DIR = ROOT / "drl" / "figures_s"
else:
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
    FIG_DIR = ROOT / "drl" / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 0
TOTAL_TIMESTEPS = 50_000  # reduced vs. Phase 5's 100k: this sweep trains 9 separate
                          # PPO models (5 handover values + 4 reward profiles), so the
                          # per-run budget is scoped down to keep total runtime tractable.
EPISODE_STEPS = 60
N_ENVS = 4
N_EVAL_EPISODES = 100

# 0.0 included as a true zero-penalty ablation (ties the switch-avoidance claim directly
# to the handover-cost term, rather than inferring it from the lowest tested nonzero value).
HANDOVER_PENALTIES = [0.0, 0.05, 0.15, 0.30, 0.50]
REWARD_PROFILES = {
    "balanced": RewardWeights(throughput=1 / 3, latency=1 / 3, reliability=1 / 3),
    "throughput-leaning": RewardWeights(throughput=0.6, latency=0.2, reliability=0.2),
    "latency-leaning": RewardWeights(throughput=0.2, latency=0.6, reliability=0.2),
    "reliability-leaning": RewardWeights(throughput=0.2, latency=0.2, reliability=0.6),
}


def train_and_eval(handover_penalty: float, reward_weights: RewardWeights) -> dict:
    def make_env():
        return NetworkSelectionEnv(CSV_PATH, PARAMS_PATH, episode_steps=EPISODE_STEPS,
                                    handover_penalty=handover_penalty, reward_weights=reward_weights)

    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)
    model = PPO("MlpPolicy", vec_env, seed=SEED, verbose=1, n_steps=512, batch_size=256)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    print(f"    evaluating ({N_EVAL_EPISODES} episodes)...", flush=True)
    eval_df = evaluate_policy(make_env, SB3Policy(model), n_episodes=N_EVAL_EPISODES, progress_label="eval")
    s = summarize_eval(eval_df)
    return {
        "reward_mean": s["total_reward"]["mean"],
        "switch_rate_mean": s["switch_rate"]["mean"],
        "outage_rate_mean": s["outage_rate"]["mean"],
    }


def run_handover_sweep() -> pd.DataFrame:
    rows = []
    for penalty in HANDOVER_PENALTIES:
        print(f"  handover_penalty={penalty} ...", flush=True)
        result = train_and_eval(penalty, REWARD_PROFILES["balanced"])
        rows.append({"handover_penalty": penalty, **result})
    return pd.DataFrame(rows)


def run_reward_profile_sweep() -> pd.DataFrame:
    rows = []
    for name, weights in REWARD_PROFILES.items():
        print(f"  reward_profile={name} ...", flush=True)
        result = train_and_eval(HANDOVER_PENALTY, weights)
        rows.append({"reward_profile": name, **result})
    return pd.DataFrame(rows)


def plot_handover_sweep(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["handover_penalty"], df["switch_rate_mean"], marker="o", color="#C44E52")
    axes[0].set_xlabel("handover penalty")
    axes[0].set_ylabel("mean switch rate")
    axes[0].set_title("Switch rate vs. handover penalty")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["handover_penalty"], df["reward_mean"], marker="o", color="#4C72B0")
    axes[1].set_xlabel("handover penalty")
    axes[1].set_ylabel("mean episode reward")
    axes[1].set_title("Reward vs. handover penalty")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Phase 6: PPO sensitivity to handover penalty (balanced reward)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase6_handover_sweep.png", dpi=150)
    plt.close(fig)


def plot_reward_profile_sweep(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(df))
    ax.bar(x, df["switch_rate_mean"], color="#DD8452", label="switch rate")
    ax2 = ax.twinx()
    ax2.plot(x, df["reward_mean"], marker="o", color="#4C72B0", label="reward")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["reward_profile"], rotation=15)
    ax.set_ylabel("mean switch rate")
    ax2.set_ylabel("mean episode reward")
    ax.set_title(f"Phase 6: PPO sensitivity to reward-weight profile (handover_penalty={HANDOVER_PENALTY})")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase6_reward_profile_sweep.png", dpi=150)
    plt.close(fig)


def main():
    print("running handover-penalty sweep (balanced reward)...")
    handover_df = run_handover_sweep()
    handover_df.to_csv(FIG_DIR / "phase6_handover_sweep.csv", index=False)
    plot_handover_sweep(handover_df)
    print(handover_df.to_string(index=False))

    is_monotonic = handover_df["switch_rate_mean"].is_monotonic_decreasing
    print(f"\nswitch rate monotonically non-increasing with handover penalty: {is_monotonic}")

    print("\nrunning reward-profile sweep (handover_penalty={})...".format(HANDOVER_PENALTY))
    profile_df = run_reward_profile_sweep()
    profile_df.to_csv(FIG_DIR / "phase6_reward_profile_sweep.csv", index=False)
    plot_reward_profile_sweep(profile_df)
    print(profile_df.to_string(index=False))

    print(f"\nfigures + csv written to {FIG_DIR}")


if __name__ == "__main__":
    main()
