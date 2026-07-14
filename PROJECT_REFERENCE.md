# TN+NTN DRL — output & metrics reference

Generated as a lookup index for everything produced during the Phase 1-6 DRL work.
Paths are relative to the repo root (`d:\Projects\TN+NTN\`).

## 1. Source data (pre-existing, untouched)

| Path | What it is |
|---|---|
| `Hybrid_Network_TN_NTN_Final.csv` | 55,503-row merged dataset (5 networks) — Phase 1 training data, and the source for fitting Phase 2+ mobility/reward normalization ranges |
| `TN_NTN_Deep_RL_MODEL.ipynb` | Original EDA/merge notebook (no RL code — see bug report) |
| `feature_TN_NTN.pdf` | Original feature design spec |
| `TN_NTN_methodology_bugs.pdf` | Bug report on the above two |
| `noise_models/` | Physics/ITU-R-grounded channel simulator (thermal noise → path loss → fading → SINR → BER → throughput → latency). Reused live by the Phase 3+ environment via `noise_models/link.py:simulate_trace` |

## 2. `drl/` — source package

| Path | What it is |
|---|---|
| `drl/features.py` | `NETWORK_TYPES`, `AREAS` constants; `build_state`/`build_actions` (Phase 1 encoding); `parse_observation` (decodes the Phase 3 env's observation vector) |
| `drl/reward.py` | Composite-QoE reward: `RewardWeights`, `fit_normalization`, `compute_reward` |
| `drl/reward_norm_params.json` | Fitted reward normalization bounds (1st-99th percentile per KPI), persisted for reproducibility |
| `drl/mobility.py` | Shared-mobility model: bounded-walk geometry for TN/UAV, analytic pass geometry (rise/peak/set) for LEO/HAPS, Area Markov-chain transitions (`AREA_TRANSITION`) |
| `drl/env.py` | `NetworkSelectionEnv` (Gymnasium env) — condition-aware observation, handover penalty, invalid-action handling |
| `drl/baselines.py` | `RandomPolicy`, `GreedyPolicy`, `HysteresisPolicy`, oracle (`make_oracle_policy` + DP solver) |
| `drl/evaluation.py` | `rollout_episode`, `evaluate_policy`, `bootstrap_ci`, `summarize_eval`, `multi_seed_evaluate`, `SB3Policy`/`D3RLPolicy` adapters |

## 3. Trained models

| Path | What it is |
|---|---|
| `drl/cql_network_selector.d3` | Phase 1 Discrete CQL, default `alpha=1.0` (superseded — 3/6 Area match rate) |
| `drl/cql_network_selector_alpha0.1.d3` | Phase 1 Discrete CQL, tuned `alpha=0.1` (5/6 Area match rate) — load via `d3rlpy.load_learnable(path)` |
| `drl/ppo_network_selector.zip` | Phase 2 (early, thin-state) PPO — superseded by Phase 5's condition-aware agents; kept for history only |
| **Phase 5 PPO/DQN models** | **Not saved to disk** — `scripts/train_agents.py` trains 3 seeds each in memory and evaluates them, but only the *results* (`phase5_full_eval_results.csv`) are persisted, not the model weights. Rerun the script to reproduce, or see §6 below to save weights if you want them. |

## 4. `drl/figures/` — all figures, CSVs, metrics summaries

| File | Phase | Content |
|---|---|---|
| `METRICS.md` | 1-2 | Plain-text headline numbers |
| `phase1_reward_by_network.png` | 1 | Mean reward per `network_type` |
| `phase1_cql_training_loss.png` | 1 | CQL `td_loss`/`conservative_loss`, alpha=1.0 vs 0.1 |
| `phase1_policy_vs_empirical.png` / `.csv` | 1 | Per-Area: CQL policy choice vs. empirical-best network |
| `phase2_ppo_training_curve.png` | 2 | Superseded thin-state PPO training curve |
| `phase2_eval_summary.png` / `.csv` | 2 | Superseded thin-state PPO 5-episode eval |
| `phase5_full_eval_results.csv` | 5 | **Raw per-episode results**, all 6 policies (random/greedy/hysteresis/oracle/ppo/dqn), 200 episodes × up to 3 seeds each — the source-of-truth table |
| `phase5_summary.csv` | 5 | Aggregated mean + 95% CI per policy (the table reported in `status.md`) |
| `phase5_reward_comparison.png` | 5 | Bar chart, reward with CI error bars, all 6 policies |
| `phase5_switch_outage_comparison.png` | 5 | Bar charts, switch rate & outage rate, all 6 policies |
| `phase5_reward_vs_switch_tradeoff.png` | 5 | Scatter, reward vs. switch rate — the single best summary figure |
| `phase6_handover_sweep.png` / `.csv` | 6 | Switch rate & reward vs. handover penalty (monotonic decrease confirmed) |
| `phase6_reward_profile_sweep.png` / `.csv` | 6 | Switch rate & reward vs. reward-weight profile (latency-leaning wins on reward) |

## 5. `scripts/` — runnable drivers

| Script | Produces |
|---|---|
| `compute_reward.py` | `drl/reward_norm_params.json` + reward sanity summary |
| `train_cql.py` | Phase 1 CQL model + `d3rlpy_logs/DiscreteCQL_*/` (raw per-step loss CSVs) |
| `train_ppo.py` | Superseded Phase 2 PPO (thin state) |
| `make_drl_figures.py` | Phase 1-2 figures |
| `train_agents.py` | Phase 5: baselines + multi-seed PPO/DQN training & evaluation → `phase5_*.csv` |
| `make_phase5_figures.py` | Phase 5 comparison figures from `phase5_summary.csv` |
| `sweep_reward_handover.py` | Phase 6: sensitivity sweep → `phase6_*.csv`/`.png` |

## 6. Raw d3rlpy training logs (auto-generated, Phase 1 only)

`d3rlpy_logs/DiscreteCQL_<timestamp>/` — one directory per CQL run (alpha=1.0 and alpha=0.1), containing per-training-step CSVs (`td_loss.csv`, `conservative_loss.csv`, plus internal gradient-norm CSVs for each network layer). This is d3rlpy's own logging, not something we wrote.

**Gap worth knowing about**: SB3 (PPO/DQN, Phases 2/5/6) does **not** get this automatic structured logging in our current scripts — its live progress tables only exist in whatever console/task-output log you were watching during the run; nothing is persisted to CSV. If you want persisted, replayable training curves for PPO/DQN (e.g. to plot `ep_rew_mean` vs. `total_timesteps` without re-transcribing console output, like I had to do manually for Phase 2), the fix is one line: pass `tensorboard_log="./tb_logs"` to `PPO(...)`/`DQN(...)` in `scripts/train_agents.py` and `sweep_reward_handover.py`, or wrap envs in SB3's `Monitor` with a `filename=` to get a CSV. Not done yet — flagging as a known gap, not a silent omission.

---

## 7. Deep Learning / Deep RL metrics — full catalog

Organized by category. ✅ = we have this, with where to find it. ⚠️ = partially/only visible in raw console logs, not persisted as data. ❌ = not computed, listed for completeness/future work.

### A. Neural-network training diagnostics (generic DL)
| Metric | Status | Where |
|---|---|---|
| Training loss (total) | ✅ (CQL) / ⚠️ (PPO/DQN) | `d3rlpy_logs/*/loss.csv`; PPO/DQN only in console logs (see §6 gap) |
| Gradient norms per layer | ✅ (CQL only) | `d3rlpy_logs/*/q_funcs.*_grad.csv` |
| Learning rate schedule | ⚠️ | printed in SB3 console tables (`learning_rate`), constant in our runs (not scheduled) |

### B. PPO-specific training diagnostics
| Metric | Meaning | Status |
|---|---|---|
| `approx_kl` | KL divergence between old/new policy per update — should stay small (<0.02ish) or PPO's trust-region assumption is violated | ⚠️ console only |
| `clip_fraction` | Fraction of samples where PPO's probability-ratio clipping activated | ⚠️ console only |
| `entropy_loss` | Policy entropy (negative = more deterministic policy) — tracks exploration→exploitation over training | ⚠️ console only |
| `explained_variance` | How well the value function predicts actual returns (1.0 = perfect, 0 = no better than predicting the mean) — ours stayed low (~0.01-0.15), a real limitation worth reporting | ⚠️ console only |
| `value_loss` | Critic's MSE loss | ⚠️ console only |
| `policy_gradient_loss` | Actor's loss term | ⚠️ console only |

### C. DQN-specific training diagnostics
| Metric | Meaning | Status |
|---|---|---|
| `exploration_rate` | Current epsilon in epsilon-greedy | ⚠️ console only |
| TD loss | Bellman-error loss on sampled transitions | ⚠️ console only |
| `n_updates` | Gradient steps taken on the Q-network | ⚠️ console only |

### D. CQL-specific (offline RL) diagnostics
| Metric | Status | Where |
|---|---|---|
| `td_loss` | ✅ | `d3rlpy_logs/*/td_loss.csv`, plotted in `phase1_cql_training_loss.png` |
| `conservative_loss` | ✅ | `d3rlpy_logs/*/conservative_loss.csv`, plotted in `phase1_cql_training_loss.png` |

### E. Episodic / policy-performance metrics (what actually matters for the paper)
| Metric | Status | Where |
|---|---|---|
| Mean episode return (reward) | ✅ | `phase5_summary.csv:reward_mean`, `phase1`/`phase2` equivalents |
| 95% confidence interval (bootstrap) | ✅ | `phase5_summary.csv:reward_ci_low/high`; computed via `drl/evaluation.py:bootstrap_ci` (2000 resamples) |
| Switch/handover rate | ✅ | `phase5_summary.csv:switch_rate_mean` — our domain-specific ping-pong metric |
| Outage/invalid-action rate | ✅ | `phase5_summary.csv:outage_rate_mean` — picking an unavailable network |
| Episode length | ✅ (fixed at 60, not a free variable in our setup) | `drl/evaluation.py:rollout_episode` returns `steps` |
| Per-seed variance across training runs | ✅ (raw) / ❌ (not summarized as std) | `phase5_full_eval_results.csv:train_seed` column has the raw per-seed data; the summary table pools all seeds into one CI rather than also reporting inter-seed std separately — easy follow-up if you want seed-to-seed variance isolated from within-policy episode variance |
| Sample efficiency (reward vs. training steps, cross-algorithm) | ❌ | Would need the persisted training-curve fix in §6 to compare PPO vs DQN convergence speed directly, not just final performance |
| Wall-clock / compute cost per algorithm | ❌ | Not logged; could extract from task timestamps post-hoc but not systematically recorded |
| Action distribution / policy entropy at convergence | ❌ | Not computed — would show e.g. how "confident"/deterministic the final policy is per Area |
| Regret vs. oracle (reward gap) | ✅ (derivable) | Not a named column, but directly computable from `phase5_summary.csv`: oracle − ppo = 45.20 − 44.19 = 1.01 (≈2% gap) |
| Success/failure rate on a task-specific criterion | N/A | No pass/fail criterion defined for this problem (continuous reward instead) |

### F. Reward/handover sensitivity metrics (Phase 6 — done)
| Metric | Status | Where |
|---|---|---|
| Reward & switch rate vs. handover penalty (4 points) | ✅ | `phase6_handover_sweep.csv`/`.png` |
| Reward & switch rate vs. reward-weight profile (4 points) | ✅ | `phase6_reward_profile_sweep.csv`/`.png` |
| Monotonicity check (switch rate ↓ as penalty ↑) | ✅ confirmed `True` | `status.md` |

---

**Bottom line on gaps**: everything needed for the paper's headline comparison (Phase 5 table) is solid and reproducible. The one real gap is unpersisted PPO/DQN training curves (§6) — fixable in ~5 minutes if you want cross-algorithm sample-efficiency plots later, just wasn't part of the original Phase 3-6 plan's scope.
