# TN+NTN Hybrid Network Simulation & Network Selection

A simulation-and-learning stack for hybrid **Terrestrial (TN)** and **Non-Terrestrial (NTN)**
networks — 5G NR, WiFi 6, HAPS, LEO satellite, and UAV links. It pairs a physics/ITU-R-grounded
channel simulator with a deep reinforcement learning agent that learns *which network a mobile user
should connect to at each moment*, benchmarked against classical telecom handover algorithms.

**See [REPORT.md](REPORT.md) for the full record** — methodology, plan, phase-by-phase progress,
results, and known gaps.

## Layout

| Path | What it is |
|---|---|
| `data/` | Datasets: merged 5-network CSV, `raw/` scenario snapshots, `regenerated/` comparison run |
| `sim/` | Layer 1 scenario simulator (`python_sim_all.py`) and layer 2 NS-3 scripts (`ns3/`, planned) |
| `noise_models/` | Layer 3 — physics-grounded channel model (the simulation core) |
| `drl/` | Reinforcement learning package: env, reward, mobility, baselines, evaluation, `figures/` |
| `scripts/` | Runnable drivers for generation, training, evaluation, and sweeps |
| `validation/` | Physical invariant checks and diagnostic plots |
| `docs/` | `noise_model_research.md` — full layer-3 methodology |
| `reports/` | Formal write-ups (papers, FL assessment, bug report, specs) |
| `notebooks/` | Original EDA/merge notebook |

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyarrow d3rlpy stable-baselines3

# Layer 1 — regenerate baseline scenario datasets
python sim/python_sim_all.py

# Layer 3 — regenerate fine-grained noisy time series
python scripts/run_all.py sample   # fast: samples + validation/report.md
python scripts/run_all.py full     # complete generation, ~33M rows

# DRL pipeline
python scripts/compute_reward.py         # fit + persist reward normalization
python scripts/train_agents.py           # main experiment (long-running)
python scripts/make_phase5_figures.py
```

> **Note:** on Windows this project targets Python 3.14. Install with `python -m pip install`,
> not bare `pip install` — see the environment notes in [REPORT.md](REPORT.md).

## Headline result

PPO reaches ~98% of an oracle upper bound while achieving the **lowest handover rate of any policy
tested**, including a hand-tuned 3GPP A3-style hysteresis baseline — learning anti-ping-pong
behaviour purely from the handover-cost reward term. Full table in [REPORT.md](REPORT.md).
