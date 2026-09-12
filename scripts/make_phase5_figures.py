"""Figures for the Phase 5 policy comparison (RL agents vs. classical telecom
baselines vs. oracle), from the results written by scripts/train_agents.py."""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("TN_NTN_DATASET", "ku")
FIG_DIR = ROOT / "drl" / {"ka": "figures_ka", "s": "figures_s"}.get(DATASET, "figures")

POLICY_ORDER = ["random", "greedy", "hysteresis", "dqn", "ppo", "oracle"]
COLORS = {
    "random": "#8C8C8C", "greedy": "#DD8452", "hysteresis": "#55A868",
    "dqn": "#8172B2", "ppo": "#4C72B0", "oracle": "#C44E52",
}


def fig_reward_comparison(summary: pd.DataFrame):
    summary = summary.set_index("policy").loc[POLICY_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    errs = [summary["reward_mean"] - summary["reward_ci_low"], summary["reward_ci_high"] - summary["reward_mean"]]
    ax.bar(summary["policy"], summary["reward_mean"], yerr=errs, capsize=4,
           color=[COLORS[p] for p in summary["policy"]])
    ax.set_ylabel("mean episode total reward (95% CI)")
    ax.set_title("Phase 5: RL agents vs. classical handover baselines vs. oracle")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_reward_comparison.png", dpi=150)
    plt.close(fig)


def fig_switch_outage_comparison(summary: pd.DataFrame):
    summary = summary.set_index("policy").loc[POLICY_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(summary["policy"], summary["switch_rate_mean"], color=[COLORS[p] for p in summary["policy"]])
    axes[0].set_ylabel("mean switch rate")
    axes[0].set_title("Handover frequency (lower = less ping-pong)")
    axes[1].bar(summary["policy"], summary["outage_rate_mean"], color=[COLORS[p] for p in summary["policy"]])
    axes[1].set_ylabel("mean outage/invalid-action rate")
    axes[1].set_title("Invalid selections (picking an unavailable network)")
    fig.suptitle("Phase 5: switch rate and outage rate by policy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_switch_outage_comparison.png", dpi=150)
    plt.close(fig)


def fig_reward_vs_switch_scatter(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for _, row in summary.iterrows():
        ax.scatter(row["switch_rate_mean"], row["reward_mean"], s=140, color=COLORS[row["policy"]], label=row["policy"])
        ax.annotate(row["policy"], (row["switch_rate_mean"], row["reward_mean"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.set_xlabel("mean switch rate")
    ax.set_ylabel("mean episode total reward")
    ax.set_title("Phase 5: reward vs. switch-rate trade-off\n(top-left is best: high reward, low switching)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_reward_vs_switch_tradeoff.png", dpi=150)
    plt.close(fig)


def main():
    summary = pd.read_csv(FIG_DIR / "phase5_summary.csv")
    fig_reward_comparison(summary)
    fig_switch_outage_comparison(summary)
    fig_reward_vs_switch_scatter(summary)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
