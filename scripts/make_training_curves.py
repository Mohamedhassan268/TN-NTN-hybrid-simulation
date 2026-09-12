"""Phase B: PPO vs DQN sample-efficiency figure from the SB3 Monitor logs in drl/monitor/
(Ku-band, the paper's primary results). Previously impossible -- Monitor logging was added to
train_agents.py partway through this project; this is the first campaign that produced them.

Each seed's 4 parallel envs are pooled by cumulative timestep (env steps = episode index * 60,
since EPISODE_STEPS=60), then averaged across the 3 seeds with a rolling window for readability.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MONITOR_DIR = ROOT / "drl" / "monitor"
FIG_DIR = ROOT / "drl" / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPISODE_STEPS = 60
SEEDS = [0, 1, 2]
ROLLING_WINDOW = 20


def load_seed_curve(label: str, seed: int) -> pd.DataFrame:
    """Pools all 4 parallel envs for one seed, sorted by wall-clock time, with cumulative
    env-steps computed from episode order (each episode is exactly EPISODE_STEPS steps)."""
    seed_dir = MONITOR_DIR / f"{label}_seed{seed}"
    frames = []
    for csv_path in sorted(seed_dir.glob("*.monitor.csv")):
        df = pd.read_csv(csv_path, skiprows=1)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    combined["timesteps"] = (combined.index + 1) * EPISODE_STEPS
    combined["reward_smoothed"] = combined["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return combined


def averaged_curve(label: str, grid: np.ndarray) -> np.ndarray:
    """Interpolates each seed's smoothed curve onto a common timestep grid, then averages."""
    seed_curves = []
    for seed in SEEDS:
        df = load_seed_curve(label, seed)
        interp = np.interp(grid, df["timesteps"], df["reward_smoothed"])
        seed_curves.append(interp)
    return np.mean(seed_curves, axis=0)


def main():
    grid = np.linspace(EPISODE_STEPS, 100_000, 500)
    ppo_curve = averaged_curve("ppo", grid)
    dqn_curve = averaged_curve("dqn", grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, ppo_curve, label="PPO", color="#4C72B0", linewidth=2)
    ax.plot(grid, dqn_curve, label="DQN", color="#8172B2", linewidth=2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode reward ({ROLLING_WINDOW}-episode rolling mean, 3-seed average)")
    ax.set_title("Phase 5: PPO vs DQN sample efficiency (Ku-band)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "phase5_training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"PPO final smoothed reward: {ppo_curve[-1]:.2f}")
    print(f"DQN final smoothed reward: {dqn_curve[-1]:.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
