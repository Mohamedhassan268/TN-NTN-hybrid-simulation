"""Create Figure 1: independent offline CQL and online PPO/DQN branches."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent


def box(ax, xy, text, color):
    patch = FancyBboxPatch(xy, 2.25, 0.64, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#1f2937")
    ax.add_patch(patch)
    ax.text(xy[0] + 1.125, xy[1] + 0.32, text, ha="center", va="center", fontsize=9)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#374151"})


if __name__ == "__main__":
    out = ROOT / "manuscript" / "revision_v2" / "figures" / "figure1_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.3, 4.6), layout="constrained")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    box(ax, (0.5, 3.2), "Frozen matched\nscenario bank", "#dbeafe")
    box(ax, (3.8, 5.3), "Offline logged\ntrajectory groups", "#fef3c7")
    box(ax, (7.0, 5.3), "CQL\n21 features", "#fef3c7")
    box(ax, (3.8, 1.1), "Online Gym\nenvironment", "#dcfce7")
    box(ax, (7.0, 1.9), "PPO\n26 features", "#dcfce7")
    box(ax, (7.0, 0.2), "DQN\n26 features", "#dcfce7")
    ax.text(7.9, 4.1, "Independent training\nNo checkpoint transfer", ha="center", va="center", fontsize=9, fontweight="bold")
    arrow(ax, (2.75, 3.7), (3.8, 5.65)); arrow(ax, (6.05, 5.65), (7.0, 5.65))
    arrow(ax, (2.75, 3.35), (3.8, 1.42)); arrow(ax, (6.05, 1.42), (7.0, 2.22)); arrow(ax, (6.05, 1.42), (7.0, 0.52))
    fig.savefig(out, dpi=300, bbox_inches="tight")
