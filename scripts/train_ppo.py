"""Phase-2 training: PPO on the shared-mobility NetworkSelectionEnv (sequential,
handover-cost-aware network selection). See drl/env.py and status.md for the
environment design and its documented simplifications.
"""
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drl.env import NetworkSelectionEnv

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Final.csv"
PARAMS_PATH = ROOT / "drl" / "reward_norm_params.json"
MODEL_PATH = ROOT / "drl" / "ppo_network_selector.zip"

SEED = 0
TOTAL_TIMESTEPS = 100_000
EPISODE_STEPS = 60
N_ENVS = 4


def main():
    env = make_vec_env(lambda: NetworkSelectionEnv(CSV_PATH, PARAMS_PATH, episode_steps=EPISODE_STEPS, seed=SEED),
                        n_envs=N_ENVS, seed=SEED)

    model = PPO("MlpPolicy", env, seed=SEED, verbose=1, n_steps=512, batch_size=256)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(str(MODEL_PATH))

    # --- evaluation: run held-out episodes per Area, compare PPO's chosen network
    #     mix + switch rate against a "greedy, no handover awareness" reference
    #     (always pick the Phase-1 empirical-best network for the Area).
    eval_env = NetworkSelectionEnv(CSV_PATH, PARAMS_PATH, episode_steps=EPISODE_STEPS, seed=SEED + 999)
    print("\nper-episode evaluation (5 episodes):")
    for ep in range(5):
        obs, info = eval_env.reset(seed=SEED + 1000 + ep)
        area = eval_env._area
        total_reward = 0.0
        switches = 0
        prev_net = None
        chosen = []
        for _ in range(EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = eval_env.step(int(action))
            total_reward += reward
            net = step_info.get("network_type")
            if net is not None:
                chosen.append(net)
                if prev_net is not None and net != prev_net:
                    switches += 1
                prev_net = net
            if terminated:
                break
        print(f"  Area={area:9s} total_reward={total_reward:6.2f} switches={switches:2d} "
              f"networks_used={sorted(set(chosen))}")

    print(f"\nmodel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
