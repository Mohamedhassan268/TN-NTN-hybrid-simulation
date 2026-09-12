"""Held-out geometry generalization test (Phase 1.4). Trains happen once under
drl.mobility._PASS_PARAMS (nominal LEO/HAPS pass geometry); this evaluates the SAVED
PPO models plus the hysteresis baseline under a deliberately DIFFERENT, harder orbital-geometry
regime -- lower peak elevations, shorter passes, more frequent/longer out-of-view gaps -- as a
stand-in for "a different orbital shell." Answers: does the learned policy generalize, or did it
memorize the one fixed analytic profile it trained on? Requires drl/models_ka/ppo_seed*.zip to
already exist (produced by `TN_NTN_DATASET=ka python scripts/train_agents.py`).
"""
import os
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.baselines import HysteresisPolicy
from drl.env import NetworkSelectionEnv
from drl.evaluation import SB3Policy, evaluate_policy, summarize_eval

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("TN_NTN_DATASET", "ku")
if DATASET == "ka":
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Ka.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params_ka.json"
    FIG_DIR = ROOT / "drl" / "figures_ka"
    MODEL_DIR = ROOT / "drl" / "models_ka"
elif DATASET == "s":
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
    FIG_DIR = ROOT / "drl" / "figures_s"
    MODEL_DIR = ROOT / "drl" / "models_s"
else:
    CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
    PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
    FIG_DIR = ROOT / "drl" / "figures"
    MODEL_DIR = ROOT / "drl" / "models"
FIG_DIR.mkdir(exist_ok=True)

EPISODE_STEPS = 60
N_EVAL_EPISODES = 200
SEEDS = [0, 1, 2]

# Deliberately harder than the nominal training regime -- lower peak elevations (weaker links),
# shorter passes (less time to exploit good conditions), longer/more frequent gaps (more forced
# handovers). Not an arbitrary perturbation: this is the direction that stresses the
# switch-avoidance policy the most.
SHIFTED_PASS_PARAMS = {
    "SAT (LEO)": {"peak_range": (10.0, 60.0), "pass_len_range": (5, 15), "gap_len_range": (10, 45)},
    "HAPS": {"peak_range": (15.0, 60.0), "pass_len_range": (20, 50), "gap_len_range": (2, 10)},
}


def make_env(pass_params=None):
    return lambda: NetworkSelectionEnv(CSV_PATH, PARAMS_PATH, episode_steps=EPISODE_STEPS,
                                        pass_params=pass_params)


def main():
    rows = []

    for geometry_label, pass_params in [("nominal", None), ("shifted", SHIFTED_PASS_PARAMS)]:
        env_factory = make_env(pass_params)

        print(f"evaluating hysteresis under {geometry_label} geometry...", flush=True)
        eval_df = evaluate_policy(env_factory, HysteresisPolicy(), n_episodes=N_EVAL_EPISODES,
                                   progress_label=f"hysteresis/{geometry_label}")
        s = summarize_eval(eval_df)
        rows.append({"policy": "hysteresis", "geometry": geometry_label,
                     "reward_mean": s["total_reward"]["mean"],
                     "switch_rate_mean": s["switch_rate"]["mean"],
                     "outage_rate_mean": s["outage_rate"]["mean"]})

        for seed in SEEDS:
            model_path = MODEL_DIR / f"ppo_seed{seed}.zip"
            if not model_path.exists():
                print(f"  SKIP ppo seed={seed}: {model_path} not found "
                      f"(run TN_NTN_DATASET={DATASET} python scripts/train_agents.py first)")
                continue
            model = PPO.load(str(model_path))
            print(f"evaluating ppo seed={seed} under {geometry_label} geometry...", flush=True)
            eval_df = evaluate_policy(env_factory, SB3Policy(model), n_episodes=N_EVAL_EPISODES,
                                       progress_label=f"ppo_seed{seed}/{geometry_label}")
            s = summarize_eval(eval_df)
            rows.append({"policy": f"ppo_seed{seed}", "geometry": geometry_label,
                         "reward_mean": s["total_reward"]["mean"],
                         "switch_rate_mean": s["switch_rate"]["mean"],
                         "outage_rate_mean": s["outage_rate"]["mean"]})

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    out_path = FIG_DIR / "phase8_geometry_generalization.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
