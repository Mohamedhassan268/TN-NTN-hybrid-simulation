"""Generates the figures + metrics summary for the Phase-1 (CQL) and Phase-2 (PPO)
DRL work, saved to drl/figures/. Mirrors the plotting convention already used in
validation/ for the noise_models layer.
"""
import json
import sys
from pathlib import Path

import d3rlpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.features import AREAS, NETWORK_TYPES, build_state
from drl.reward import RewardWeights, compute_reward

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "Hybrid_Network_TN_NTN_Final.csv"
PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
FIG_DIR = ROOT / "drl" / "figures"
FIG_DIR.mkdir(exist_ok=True)

CQL_RUN_ALPHA1 = ROOT / "d3rlpy_logs" / "DiscreteCQL_20260713213317"   # alpha=1.0 (default)
CQL_RUN_ALPHA01 = ROOT / "d3rlpy_logs" / "DiscreteCQL_20260713214717"  # alpha=0.1 (tuned)
CQL_MODEL_ALPHA01 = ROOT / "drl" / "cql_network_selector_alpha0.1.d3"

# PPO ep_rew_mean vs total_timesteps, transcribed from the training log
# (scripts/train_ppo.py stdout; see status.md for the run).
PPO_TIMESTEPS = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
                 22528, 24576, 26624, 28672, 30720, 32768, 34816, 36864, 38912, 40960,
                 43008, 45056, 47104, 49152, 51200, 53248, 55296, 57344, 59392, 61440,
                 63488, 65536, 67584, 69632, 71680, 73728, 75776, 77824, 79872, 81920,
                 83968, 86016, 88064, 90112, 92160, 94208, 96256, 98304, 100352]
PPO_EP_REW_MEAN = [3.87, 6.3, 8.68, 14.2, 19.4, 23.2, 25.8, 28.3, 30.3, 31.4,
                    32.4, 33.1, 33.2, 33.8, 35.1, 36.0, 37.1, 37.2, 37.4, 38.4,
                    38.5, 39.4, 39.5, 40.2, 41.0, 41.5, 42.1, 42.1, 42.4, 42.0,
                    41.8, 42.0, 42.2, 42.3, 42.4, 42.5, 42.7, 42.7, 42.7, 42.8,
                    42.9, 43.0, 42.8, 42.7, 42.7, 42.8, 43.3, 43.3, 43.5]

PPO_EVAL_RESULTS = [
    {"area": "Rural", "total_reward": 43.51, "switches": 0, "network": "NR_5G"},
    {"area": "Maritime", "total_reward": 45.85, "switches": 0, "network": "UAV"},
    {"area": "Highway", "total_reward": 45.88, "switches": 0, "network": "UAV"},
    {"area": "Rural", "total_reward": 49.68, "switches": 0, "network": "NR_5G"},
    {"area": "Desert", "total_reward": 45.30, "switches": 0, "network": "UAV"},
]


def fig_phase1_reward_by_network(df, reward):
    fig, ax = plt.subplots(figsize=(7, 4))
    means = reward.groupby(df["network_type"]).mean().sort_values(ascending=False)
    ax.bar(means.index, means.values, color="#4C72B0")
    ax.set_ylabel("mean composite QoE reward")
    ax.set_title("Phase 1: mean reward by network_type (Hybrid_Network_TN_NTN_Final.csv)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase1_reward_by_network.png", dpi=150)
    plt.close(fig)


def fig_phase1_cql_training_loss():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run_dir, label in [(CQL_RUN_ALPHA1, "alpha=1.0 (default)"), (CQL_RUN_ALPHA01, "alpha=0.1 (tuned)")]:
        td = pd.read_csv(run_dir / "td_loss.csv", header=None, names=["epoch", "step", "value"])
        cons = pd.read_csv(run_dir / "conservative_loss.csv", header=None, names=["epoch", "step", "value"])
        axes[0].plot(td["step"], td["value"], label=label)
        axes[1].plot(cons["step"], cons["value"], label=label)
    axes[0].set_title("td_loss")
    axes[1].set_title("conservative_loss")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend()
    fig.suptitle("Phase 1: Discrete CQL training loss, alpha=1.0 vs alpha=0.1")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase1_cql_training_loss.png", dpi=150)
    plt.close(fig)


def fig_phase1_policy_vs_empirical(df, reward):
    cql = d3rlpy.load_learnable(str(CQL_MODEL_ALPHA01))
    area_networks = df[["Area", "Available_Networks"]].drop_duplicates().set_index("Area")["Available_Networks"]
    train_df = df.assign(reward=reward)

    rows = []
    for area in AREAS:
        empirical = train_df[train_df["Area"] == area].groupby("network_type")["reward"].mean()
        best_network = empirical.idxmax()
        state = build_state(pd.DataFrame({"Area": [area], "Available_Networks": [area_networks[area]]}))
        policy_action = cql.predict(state)[0]
        policy_network = NETWORK_TYPES[policy_action]
        rows.append({
            "area": area,
            "empirical_best_network": best_network,
            "empirical_best_reward": empirical.max(),
            "cql_policy_network": policy_network,
            "cql_policy_reward": empirical.get(policy_network, np.nan),
            "match": best_network == policy_network,
        })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(FIG_DIR / "phase1_policy_vs_empirical.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(result_df))
    width = 0.35
    ax.bar(x - width / 2, result_df["empirical_best_reward"], width, label="empirical best (train)", color="#55A868")
    ax.bar(x + width / 2, result_df["cql_policy_reward"], width, label="CQL policy choice", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(result_df["area"])
    for i, row in result_df.iterrows():
        marker = "✓" if row["match"] else "✗"
        ax.text(i, max(row["empirical_best_reward"], row["cql_policy_reward"]) + 0.02, marker, ha="center")
    ax.set_ylabel("mean reward")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 1: CQL policy vs. empirical-best network reward, per Area\n(alpha=0.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase1_policy_vs_empirical.png", dpi=150)
    plt.close(fig)
    return result_df


def fig_phase2_ppo_training_curve():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(PPO_TIMESTEPS, PPO_EP_REW_MEAN, color="#C44E52")
    ax.set_xlabel("total timesteps")
    ax.set_ylabel("mean episode reward (ep_rew_mean)")
    ax.set_title("Phase 2: PPO training curve (NetworkSelectionEnv, 60-step episodes)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2_ppo_training_curve.png", dpi=150)
    plt.close(fig)


def fig_phase2_eval_summary():
    eval_df = pd.DataFrame(PPO_EVAL_RESULTS)
    eval_df.to_csv(FIG_DIR / "phase2_eval_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"NR_5G": "#4C72B0", "UAV": "#DD8452", "HAPS": "#55A868", "WiFi": "#C44E52", "SAT (LEO)": "#8172B2"}
    bar_colors = [colors[n] for n in eval_df["network"]]
    labels = [f"{a}\n(ep {i+1})" for i, a in enumerate(eval_df["area"])]
    ax.bar(labels, eval_df["total_reward"], color=bar_colors)
    for i, row in eval_df.iterrows():
        ax.text(i, row["total_reward"] + 0.5, row["network"], ha="center", fontsize=9)
    ax.set_ylabel("episode total reward (60 steps)")
    ax.set_title("Phase 2: PPO held-out evaluation episodes (0 switches in all 5)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2_eval_summary.png", dpi=150)
    plt.close(fig)
    return eval_df


def write_metrics_summary(reward, phase1_result_df, phase2_eval_df):
    lines = [
        "# DRL metrics summary\n",
        f"Generated by `scripts/make_drl_figures.py`. Figures/data in `drl/figures/`.\n",
        "## Phase 1 — Discrete CQL (bandit, Hybrid_Network_TN_NTN_Final.csv)\n",
        f"- Reward range: [{reward.min():.4f}, {reward.max():.4f}], mean {reward.mean():.4f} (55,503 rows)",
        f"- Policy (alpha=0.1) matches empirical-best network in "
        f"{phase1_result_df['match'].sum()}/{len(phase1_result_df)} Areas "
        f"(alpha=1.0 default matched only 3/6 — see phase1_cql_training_loss.png)",
        "- See `phase1_reward_by_network.png`, `phase1_cql_training_loss.png`, "
        "`phase1_policy_vs_empirical.png` (+ `.csv`)\n",
        "## Phase 2 — PPO (sequential, NetworkSelectionEnv with handover cost)\n",
        f"- Training: 100,352 timesteps, ep_rew_mean {PPO_EP_REW_MEAN[0]:.2f} -> {PPO_EP_REW_MEAN[-1]:.2f}",
        f"- Held-out eval (5 episodes x 60 steps): 0/5 episodes had any network switch "
        f"(mean total reward {phase2_eval_df['total_reward'].mean():.2f})",
        "- See `phase2_ppo_training_curve.png`, `phase2_eval_summary.png` (+ `.csv`)\n",
    ]
    (FIG_DIR / "METRICS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = pd.read_csv(CSV_PATH)
    params = json.loads(PARAMS_PATH.read_text())
    reward = compute_reward(df, params, RewardWeights())

    fig_phase1_reward_by_network(df, reward)
    fig_phase1_cql_training_loss()
    phase1_result_df = fig_phase1_policy_vs_empirical(df, reward)
    fig_phase2_ppo_training_curve()
    phase2_eval_df = fig_phase2_eval_summary()
    write_metrics_summary(reward, phase1_result_df, phase2_eval_df)

    print(f"figures + metrics written to {FIG_DIR}")
    for f in sorted(FIG_DIR.iterdir()):
        print(" ", f.name)


if __name__ == "__main__":
    main()
