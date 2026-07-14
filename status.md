# Project Status

## Current phase
Phases 1-5 complete (paper-grade redesign). Phase 6 (reward/handover-cost sensitivity sweep) running.

## Plan
Active plan file: `C:\Users\HP OMEN\.claude\plans\i-added-three-new-sprightly-flute.md`

## Decisions made
- Feature approach: **Option A** — feed engineered physical-layer KPIs directly as state; do not have the model reconstruct KPIs from raw physics.
- Roadmap: **phased**.
  - **Phase 1 (done)**: contextual-bandit framing. State = `Area` + `Available_Networks` only (the KPI columns are outcomes of the network choice, not pre-decision context — confirmed e.g. `distance_km` is ~800-1900km for every SAT(LEO) row vs <2km for every NR_5G row). Action = `network_type` (5-way). Reward = composite QoE (`drl/reward.py`, balanced weights over throughput/latency/reliability, using `Latency_ms` which already includes propagation delay per `noise_models/link.py:135`). Trained Discrete CQL (`d3rlpy`) on `Hybrid_Network_TN_NTN_Final.csv`; `alpha=0.1` (down from default 1.0) matches the empirical best network in 5/6 Areas — default alpha over-penalized on this small, fully-covered 6-state action table. Model: `drl/cql_network_selector_alpha0.1.d3`.
  - **Phase 2 (next)**: sequential DRL with handover-cost reward, trained on ordered time-series traces from `noise_models/` wrapped as a Gym environment. PPO primary, Double+Dueling DQN baseline.

## Phase 2 — built
Gap identified: `noise_models/generate.py` only produces one technology's trace per source row, drawn from an independent Monte-Carlo scenario per technology (confirmed via `Script python/python_sim_all.py` — `sim_5g`/`sim_haps`/`sim_leo`/etc. each simulate their own unrelated 20 UEs). There was no shared UE trajectory across networks, so handover/comparison had no physical basis. Resolved by building:
- `drl/mobility.py` — shared-mobility model: per-episode Area is fixed, and each candidate network's `distance_km`/`Rain_Rate_mmhr`(/`altitude_m`,`speed_ms` for UAV) is a bounded random walk within 1st-99th-percentile ranges fit from the CSV per `network_type` (`fit_geometry_ranges`), so all candidates are evaluated against the same simulated moment. Doppler held at each tech's median (documented simplification — no orbital/positional mechanics modeled).
- `drl/env.py` — `NetworkSelectionEnv(gymnasium.Env)`: obs = Area one-hot + Available_Networks multi-hot + previous-action one-hot (16-dim); action = network_type (5-way); reward = Phase-1 composite QoE for that instant's KPIs (computed live via `noise_models/link.py:simulate_trace`, single-step calls) minus a flat `HANDOVER_PENALTY=0.15` if the network changed since last step; picking an unavailable network is invalid (`-1.0`, no state advance). Passes `gymnasium.utils.env_checker.check_env`.
- `scripts/train_ppo.py` — trains PPO (`stable_baselines3`) on 4 vectorized envs, evaluates 5 held-out episodes (total reward, switch count, networks used per Area).

Packages installed this session: `d3rlpy`, `stable-baselines3` (brings `gymnasium`) — via `python -m pip install` (note: `pip` on PATH targets Python 3.13, but the active interpreter is 3.14 at `AppData\Local\Python\pythoncore-3.14-64`; always use `python -m pip install`, not bare `pip install`).

## Known issues (see `TN_NTN_methodology_bugs.pdf`)
- `TN_NTN_Deep_RL_MODEL.ipynb` contains no RL code — EDA/merging only.
- No feature extraction/selection has been implemented; `feature_TN_NTN.pdf` is a design spec only.
- Final CSV has nonzero rain features for `NR_5G` despite the spec marking rain as not applicable to 5G (still unresolved, low priority).

## Phase 3-5 (paper-grade redesign) — done
Re-evaluating the original Phase-2 env against the "academic paper" goal surfaced a foundational flaw: its observation (`Area + Available_Networks + prev-action`) carried no live radio measurements, so the policy could only be a static per-Area lookup + stickiness, blind to actual conditions — undermining the case for deep RL at all. Fixed via a full redesign, confirmed by the user (state-enrichment: yes; baselines: telecom handover algorithms, not an offline-RL zoo):

- **Phase 3 — condition-aware env** (`drl/env.py`, `drl/mobility.py`): observation now includes per-candidate-network `[available, RSSI_norm, SINR_norm]` (26-dim total with Area one-hot + prev-action one-hot), computed live via `noise_models/link.py:simulate_trace` each step — pre-decision measurables only, outcomes (throughput/latency/BER) stay reward-only (no leakage, same discipline as Phase 1). `drl/mobility.py` now models: (a) realistic LEO/HAPS pass geometry — elevation rises/peaks/sets, deriving `distance_km` from elevation via the inverse of `noise_models/channels/pathloss.elevation_deg`; below `MIN_VISIBLE_ELEV_DEG=10°` the platform is out of view, forcing a real handover (confirmed: LEO toggles visibility ~12 times per 200 steps; HAPS calibrated as a persistent regional platform, visible ~79% of the time); (b) Area now evolves within an episode via a hand-specified Markov chain (`AREA_TRANSITION`), not fixed per episode. Passes `gymnasium.utils.env_checker.check_env`.
- **Phase 4 — evaluation harness** (`drl/evaluation.py`): `evaluate_policy`/`rollout_episode` (200+ episodes, bootstrap 95% CIs via `bootstrap_ci`/`summarize_eval`), `multi_seed_evaluate`. Any policy exposing `.predict(obs) -> int` works — `SB3Policy`/`D3RLPolicy` adapters included.
- **Phase 5 — telecom baselines + RL comparison** (`drl/baselines.py`, `scripts/train_agents.py`): built `RandomPolicy`, `GreedyPolicy` (myopic/ping-pong strawman), `HysteresisPolicy` (3GPP A3-style margin+time-to-trigger), and an `oracle` (DP over a reward matrix built with independent-fading planning noise, then scored via the env's real per-seed-deterministic stepping — see docstring in `drl/baselines.py` for why this is a fair comparison despite the separate planning-noise draw). Trained PPO and DQN (`stable_baselines3`), 3 seeds each, 100k steps/seed, evaluated 200 episodes/seed (600 total per algorithm).

**Result** (`drl/figures/phase5_summary.csv`, mean reward [95% CI] / switch rate / outage rate over n episodes):

| policy | reward | switch rate | outage rate |
|---|---|---|---|
| random | 3.17 [2.08, 4.22] | 0.413 | 0.332 |
| greedy | 39.88 [39.60, 40.17] | 0.170 | 0.000 |
| hysteresis | 40.88 [40.59, 41.19] | 0.062 | 0.000 |
| dqn | 43.12 [42.92, 43.33] | 0.159 | 0.000 |
| **ppo** | **44.19 [43.97, 44.42]** | **0.043** | 0.0001 |
| oracle | 45.20 [44.83, 45.56] | 0.051 | 0.000 |

Headline finding: PPO reaches ~98% of the oracle's reward while having the **lowest switch rate of any policy, including the hand-tuned hysteresis baseline** — it learned implicit anti-ping-pong behavior purely from the handover-cost reward term, beating an explicit rule-based algorithm on both reward and switch rate. DQN clearly beats the classical baselines on reward but not on switch rate (0.159, close to greedy's 0.170) — an honest, reportable asymmetry between the two RL algorithms. Figures: `drl/figures/phase5_reward_comparison.png`, `phase5_switch_outage_comparison.png`, `phase5_reward_vs_switch_tradeoff.png` (the last one is the cleanest single figure for the paper).

## Phase 6 — reward/handover-cost sensitivity sweep
`scripts/sweep_reward_handover.py`: two 1-D sweeps (not a full grid, to keep retraining cost bounded) — handover penalty ∈ {0.05, 0.15, 0.30, 0.50} at balanced reward weights, and reward-weight profile ∈ {balanced, throughput-leaning, latency-leaning, reliability-leaning} at the default handover penalty (0.15). Each point retrains a fresh PPO (50k steps, reduced from Phase 5's 100k to keep the 8-point sweep tractable).

**Status: done.** Both sweeps completed and saved.

**Handover-penalty sweep result** (`drl/figures/phase6_handover_sweep.csv`/`.png`, balanced reward):

| handover_penalty | reward | switch_rate |
|---|---|---|
| 0.05 | 44.31 | 0.082 |
| 0.15 | 43.83 | 0.057 |
| 0.30 | 43.38 | 0.047 |
| 0.50 | 43.15 | 0.030 |

Switch rate is monotonically non-increasing as the penalty rises (confirmed `True` by the script's own check) — the mechanism works as designed. Small, expected reward cost as the penalty tightens (44.31 → 43.15), a legitimate trade-off to report, not a bug.

**Reward-profile sweep result** (`drl/figures/phase6_reward_profile_sweep.csv`/`.png`, handover_penalty=0.15):

| reward_profile | reward | switch_rate |
|---|---|---|
| balanced | 43.83 | 0.0565 |
| throughput-leaning | 43.48 | 0.0528 |
| **latency-leaning** | **44.86** | 0.0507 |
| reliability-leaning | 43.18 | 0.0465 |

Latency-leaning stands out with clearly the highest reward — consistent with the reward-design finding that `Latency_ms` (which already includes propagation delay) is the single KPI that most sharply separates TN (fast) from NTN (slow) networks, giving the agent the clearest signal to act on. Consistency check passed: the `balanced` row here (43.83, 0.0565) exactly matches the `handover_penalty=0.15` row in the other sweep — same config evaluated twice, identical result.

**This completes the Phases 3-6 plan** (`C:\Users\HP OMEN\.claude\plans\i-added-three-new-sprightly-flute.md`): condition-aware RL agents benchmarked against classical telecom handover baselines with multi-seed 95% CIs (Phase 5), plus a documented reward/handover-cost sensitivity analysis (Phase 6).

## Operational notes from this session
- Windows has two Python installs on PATH: `pip` resolves to Python 3.13's, but `python` resolves to 3.14 (`AppData\Local\Python\pythoncore-3.14-64`). Always install with `python -m pip install`, never bare `pip install`, or packages land in the wrong interpreter.
- When running training scripts as background tasks, stdout is block-buffered (not line-buffered) once redirected to a file — `print()` calls (including inside long loops with no per-iteration flush) can sit invisible for many minutes even though the process is actively working. Fix: launch with `python -u`, AND add explicit progress prints inside long loops (every ~20 episodes) — `-u` alone isn't enough if the loop itself never prints anything until it's done. Both `drl/evaluation.py:evaluate_policy` (via `progress_label`/`progress_every`) and the baseline/oracle loops in `scripts/train_agents.py` now do this.
