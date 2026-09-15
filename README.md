# TN+NTN Hybrid Network Simulation & Network Selection

## Revision v2 status

The authoritative revision is **Handover-Aware Deep Reinforcement Learning for Network Selection in Simulated Hybrid TN-NTN Systems**. Its immutable matched-scenario dataset is complete at `data/canonical_v2/`: 1,200 trajectories split 800/200/200 only by trajectory ID, 60 steps at 10 seconds, five candidates per scenario-step-band, and Ku/Ka/S views.

The v2 experiment protocol is in [docs/experiment_protocol_v2.md](docs/experiment_protocol_v2.md). Run the campaign through the separate v2 scripts; all older CSVs, models, figures, and narrative claims below are preserved only as **legacy artifacts** and must not be cited as revision-v2 evidence.

```powershell
python -m pytest tests -q
python scripts/run_physical_conformance.py --external standards/external_reference_outputs.csv
python scripts/develop_ppo_critic.py
python scripts/train_revision_v2.py
python scripts/train_cql_revision_v2.py
python scripts/evaluate_revision_v2.py
python scripts/statistics_revision_v2.py
python scripts/build_geometry_suites.py --resume
python scripts/evaluate_geometry_revision_v2.py
python scripts/run_sensitivity_revision_v2.py
python scripts/make_pareto_revision_v2.py
python scripts/release_gate_revision_v2.py
```

After the CPU-only critic selection is frozen, benchmark GPU adoption before core PPO/DQN training:

```powershell
python scripts/benchmark_ppo_device.py
```

CUDA is adopted only when the frozen benchmark reports `cuda_adopted`, which requires at least a 4x matched end-to-end speed-up. Otherwise run core training with the default `--device cpu`.

The pinned external OpenNTN/LLSim5G cross-check passed on 2026-09-12. The release gate intentionally continues to fail until the PPO critic selection, full 10-seed core campaign, five-seed sweeps, frozen results, clean-clone verification, and final DOCX render have been completed.

A simulation-and-learning stack for hybrid **Terrestrial (TN)** and **Non-Terrestrial (NTN)**
networks — 5G NR, WiFi 6, HAPS, LEO satellite, and UAV links. It pairs a physics/ITU-R-grounded
channel simulator with a deep reinforcement learning agent that learns *which network a mobile
user should connect to at each moment*, benchmarked against classical telecom handover
algorithms — and it is also the working repository for a submittable research paper built on
that work.

**Two things live here:**
1. **The simulation + RL codebase** — three-layer channel simulator, Gymnasium environment,
   CQL/PPO/DQN training, evaluation harness.
2. **A manuscript in progress** — `FRIEND_AI_Native_Framework_final.docx`, a paper co-authored
   with a collaborator who wrote Sections 1–4 (introduction, related work, system model, dataset,
   methodology); this repository's results fill its Results/Discussion/Conclusion (Sections 5–7)
   and a Data and Code Availability statement. `FRIEND_AI_Native_Framework.docx` alongside it is
   the untouched original as received; earlier editing checkpoints are archived in
   `manuscript/history/`.

**See [REPORT.md](REPORT.md) for the full project record** — methodology, phase-by-phase
history, every verified number, and known gaps. This README is the map; REPORT.md is the detail.

---

## Repository layout

| Path | What it is |
|---|---|
| `data/` | Datasets: merged 5-network CSV, `raw/` scenario snapshots, `regenerated/` comparison run, `Hybrid_Network_TN_NTN_Ka.csv` (synthetic Ka-band variant, see below) |
| `sim/` | Layer 1 scenario simulator (`python_sim_all.py`) and layer 2 NS-3 scripts (`ns3/`, planned) |
| `noise_models/` | Layer 3 — physics/ITU-R channel model (the simulation core); `channels/atmosphere.py` implements ITU-R P.838-3 rain and P.676-13 gaseous absorption from the actual recommendations, not approximations |
| `drl/` | Reinforcement learning package: `env.py`, `reward.py`, `mobility.py`, `baselines.py`, `evaluation.py`; `figures/`/`figures_ka/`/`figures_s/` (Ku/Ka/S results); `models/`/`models_ka/`/`models_s/` (saved PPO/DQN checkpoints) |
| `scripts/` | Runnable drivers — generation, training, evaluation, sweeps, significance tests (see below) |
| `validation/` | Physical invariant checks and diagnostic plots for the noise model |
| `docs/` | `noise_model_research.md` — full layer-3 methodology, standards referenced, calibration |
| `reports/` | `CITATIONS_NEEDED.md`, `SECTIONS_5_7_DRAFT.md` (manuscript work), plus prior papers/assessments |
| `notebooks/` | Original EDA/merge notebook |
| `manuscript/history/` | Superseded manuscript editing checkpoints, kept for provenance |
| `FRIEND_AI_Native_Framework.docx` / `_final.docx` / `_final.pdf` | The manuscript — untouched original, current working version, and its PDF export (root level, deliberately visible) |
| `LICENSE` / `data/LICENSE` | MIT (code) and CC BY 4.0 (datasets) |
| `TN_NTN_DRL_Report.md` / `.pdf` | An earlier internal DRL report, kept alongside the manuscript for comparison |

---

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyarrow d3rlpy stable-baselines3

# Layer 1 — regenerate baseline scenario datasets
python sim/python_sim_all.py

# Layer 3 — regenerate fine-grained noisy time series
python scripts/run_all.py sample   # fast: samples + validation/report.md
python scripts/run_all.py full     # complete generation, ~33M rows

# DRL pipeline (Ku-band, the paper's primary band)
python scripts/compute_reward.py         # fit + persist reward normalization
python scripts/train_agents.py           # Phase 5: baselines + PPO/DQN, 3 seeds each (~2h)
python scripts/make_phase5_figures.py
python scripts/sweep_reward_handover.py  # Phase 6: handover-penalty + reward-profile sweeps
python scripts/eval_geometry_generalization.py
python scripts/significance_tests.py     # Welch/Mann-Whitney + Cohen's d on the Phase 5 results
```

### Running the Ka-band (20 GHz) comparison

Every pipeline script recognizes `TN_NTN_DATASET`, defaulting to `ku`:

```bash
TN_NTN_DATASET=ka python scripts/train_agents.py   # writes to drl/figures_ka/, drl/models_ka/
```

This drives the *same* generator, seeds, and reward weights through a Ka-band LEO link
(`noise_models/config.py`'s `"LEO_Ka"` `TechConfig` — only `carrier_freq_hz` and the
frequency-derived `antenna_gain_db` differ from the Ku config), so any difference in results is
attributable to the band, not to a different setup.

> **Note:** on Windows this project targets Python 3.14. Install with `python -m pip install`,
> not bare `pip install` — see the environment notes in [REPORT.md](REPORT.md).

---

## Methodology at a glance

**Three simulation layers**, coarse to fine:
1. **Scenario simulator** (`sim/python_sim_all.py`) — UE mobility, per-technology channel
   heuristics, produces baseline scenario datasets.
2. **NS-3 protocol layer** (`sim/ns3/`) — packet/protocol-level simulation, planned, not yet
   integrated.
3. **Physics/ITU-R channel model** (`noise_models/`) — regenerates each scenario row into a
   realistic time series: thermal noise → path loss → shadowing → fading → Doppler → rain/gaseous
   attenuation → interference → hardware limits → SNR → SINR → BER → throughput → packet loss →
   latency → link-quality index. This is what actually drives the RL environment's reward at
   every step — see `docs/noise_model_research.md` for the full derivation.

**The learning problem.** A Gymnasium environment (`drl/env.py`) exposes a 5-way network-choice
action at each step, with an observation built only from pre-decision measurables (never the
outcome KPIs the reward is computed from — no leakage). Reward is a composite QoE score minus a
flat handover penalty. Three approaches are compared:
- **CQL** (offline, contextual-bandit formulation) — `scripts/train_cql.py`
- **PPO** and **DQN** (online, sequential, condition-aware) — `scripts/train_agents.py`
- **Classical baselines** — Random, Greedy, a 3GPP A3-style Hysteresis policy, and a
  dynamic-programming Oracle upper bound — `drl/baselines.py`

**Evaluation** uses multi-seed training (3 seeds × 200 episodes), 95% bootstrap confidence
intervals, Welch's t-test / Mann-Whitney U / Cohen's d for significance, a handover-penalty
sensitivity sweep (including a true zero-penalty ablation), a reward-weight-profile sweep, and a
held-out geometry generalization test (evaluating trained policies under orbital pass geometry
they never trained on, to check they didn't just memorize one fixed profile).

---

## Physics grounding

Both atmospheric impairment models in `noise_models/channels/atmosphere.py` are implemented
directly from the ITU-R recommendations, not hand-fit approximations:

- **Rain attenuation** — ITU-R **P.838-3**, the closed-form regression (not a lookup table), so it
  is continuous across frequency with no clamping. Verified against the recommendation's own
  published reference table to within 0.03% across 1–100 GHz.
- **Gaseous absorption** — ITU-R **P.676-13** Annex 1 line-by-line calculation (44 oxygen +
  35 water-vapour spectral lines), integrated through the **P.835-7** Annex 1 reference standard
  atmosphere from the surface to 100 km. Verified: reproduces the well-known ~15 dB/km peak of the
  60 GHz oxygen absorption band, and the water-vapour resonance correctly peaks at 22.235 GHz —
  material at Ka-band (20 GHz), which sits on that resonance's shoulder.

The per-frequency computation is cached (`functools.lru_cache`), so this rigor costs a few hundred
milliseconds once per distinct carrier frequency, not per simulation step.

---

## Legacy results not valid for revision v2

**Ku-band (12 GHz LEO) — the paper's primary band.** Headline: a PPO agent reaches within a few
percent of the dynamic-programming oracle's reward while switching networks *less often* than a
hand-tuned 3GPP A3-style hysteresis baseline — anti-ping-pong behaviour that emerges purely from
the handover-cost term in the reward, with no explicit switching rule. Full table and statistics
in [REPORT.md](REPORT.md).

**Ka-band (20 GHz LEO) — a controlled band-sensitivity comparison.** Same generator, seeds, and
reward design; only the LEO link's frequency (and its physically-derived antenna gain) differ.
PPO reaches ~99% of the oracle's reward under Ka; the switch-rate advantage PPO holds over the
oracle under Ku narrows to a near-tie under Ka (p≈0.04), and the best reward-weight profile shifts
from latency-leaning (Ku) to reliability-leaning (Ka) — consistent with rain fade becoming the
more dominant impairment at the higher band. Results in `drl/figures_ka/`.

Both bands passed a held-out geometry generalization test without meaningful reward degradation.

---

## Legacy manuscript material

`FRIEND_AI_Native_Framework_final.docx` is a paper in progress, currently complete through a
Data and Code Availability statement. A collaborator wrote Sections 1–4; this repository's DRL
results now fill Sections 5 (Results, including a three-band Ku/Ka/S comparison), 6
(Discussion), and 7 (Conclusion). Supporting documents:

- **`reports/SECTIONS_5_7_DRAFT.md`** — the reviewable Results/Discussion/Conclusion source text,
  built directly from the verified CSVs in `drl/figures/`, `drl/figures_ka/`, and `drl/figures_s/`
  (nothing hand-transcribed).
- **`reports/CITATIONS_NEEDED.md`** — a full map of the manuscript's 24 unresolved citation sites
  (29 individual references), graded by how confidently each can be identified from context.

Known outstanding items: the 29 citations need final confirmation/sourcing, a dataset-availability
statement is promised in the manuscript's contributions but not yet written, and the document's
formatting template needs replacing before submission.

---

## Reproducibility notes

- Every stochastic component is seeded; `sim/python_sim_all.py`'s five generators use `seed=42`.
- `scripts/train_agents.py` skips retraining any seed whose model file already exists in
  `drl/models/` (or `drl/models_ka/`) — safe to resume after an interruption without losing
  completed work.
- Training runs write SB3 `Monitor` logs to `drl/monitor/` / `drl/monitor_ka/` for
  sample-efficiency analysis.
- Nothing here survives a machine power-off — background training is a plain OS process, not a
  checkpointed job. If you're running a long campaign, don't sleep or shut down the machine.
