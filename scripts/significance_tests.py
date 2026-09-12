"""Post-hoc significance tests over the existing Phase 5 evaluation episodes -- no retraining.
Welch's t-test (unequal variance) and Mann-Whitney U (distribution-free) on episode reward and
switch rate, for the comparisons the paper's Results section makes claims about, plus Cohen's d
as an effect-size measure alongside the p-value.
"""
import os
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("TN_NTN_DATASET", "ku")
FIG_DIR = ROOT / "drl" / {"ka": "figures_ka", "s": "figures_s"}.get(DATASET, "figures")
IN_PATH = FIG_DIR / "phase5_full_eval_results.csv"
OUT_PATH = FIG_DIR / "phase7_significance.csv"

COMPARISONS = [
    ("ppo", "dqn"),
    ("ppo", "hysteresis"),
    ("ppo", "oracle"),
    ("dqn", "hysteresis"),
]
METRICS = ["total_reward", "switch_rate"]


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    pooled_std = (((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2)) ** 0.5
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else float("nan")


def main():
    df = pd.read_csv(IN_PATH)
    rows = []
    for pol_a, pol_b in COMPARISONS:
        a_df = df[df.policy == pol_a]
        b_df = df[df.policy == pol_b]
        for metric in METRICS:
            a, b = a_df[metric], b_df[metric]
            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a, b)
            rows.append({
                "policy_a": pol_a, "policy_b": pol_b, "metric": metric,
                "n_a": len(a), "n_b": len(b),
                "mean_a": a.mean(), "mean_b": b.mean(),
                "welch_t": t_stat, "welch_p": t_p,
                "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
                "cohens_d": d,
            })
            print(f"{pol_a} vs {pol_b} [{metric}]: "
                  f"mean {a.mean():.4f} vs {b.mean():.4f}, "
                  f"Welch p={t_p:.2e}, MWU p={u_p:.2e}, d={d:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
