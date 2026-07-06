# TN+NTN Hybrid Network Simulation System

A multi-layer simulation system for hybrid **Terrestrial (TN)** and **Non-Terrestrial (NTN)**
networks — 5G NR, WiFi 6, HAPS, LEO satellite, and UAV links. The system is organized as three
complementary layers that go from coarse system-level geometry down to fine-grained, standards-grounded
channel realism:

1. **System-level mobility & scenario simulator** (`Script python/`) — a lightweight Python
   simulator that models UE mobility, beam/cell association, and simplified per-technology channel
   formulas to produce baseline scenario datasets (distance, Doppler, rain rate, SNR/SINR, etc.)
   at coarse time resolution (10 s steps over a ~6000 s trace, 20 UEs per technology).
2. **NS-3 protocol-level simulation** (`Scripts on Ns3/`) — *planned/in progress*: packet- and
   protocol-level network simulation (scheduling, retransmission, handover) layered on top of the
   same scenarios, using NS-3.
3. **Fine-grained physics/standards-grounded noise model** (`noise_models/`) — the core of this
   repo: takes each coarse scenario row as a **geometry descriptor** and regenerates it into a
   realistic 6-second, 10 ms-resolution time series, propagating channel impairments coherently
   through the full link chain (thermal noise, path loss, shadowing, Rayleigh/Rician fading,
   Doppler, ITU-R rain/gaseous/scintillation attenuation, interference, hardware limits):
   SNR → SINR → BER → throughput → packet loss → latency → link-quality index.

Together these layers form a hybrid simulation pipeline: coarse mobility/scenario generation →
(planned) protocol-level network simulation → fine-grained physical-layer noise realism.

## Layout

- `Script python/python_sim_all.py` — the coarse system-level mobility & scenario simulator (layer 1); regenerates the baseline CSVs from scratch
- `Scripts on Ns3/` — NS-3 protocol-level simulation scripts (layer 2, planned)
- `noise_models/` — the fine-grained noise-model package (layer 3): config, channel primitives, link chain, generation driver, validation
- `scripts/run_all.py` — driver for layer 3: `python scripts/run_all.py sample` (small per-tech samples + validation report) or `full` (complete generation)
- `docs/noise_model_research.md` — noise sources modeled, standards referenced, per-tech RF assumptions, chain equations, calibration results, and documented simplifications
- `validation/` — physical invariant checks and diagnostic plots (CDF overlays, fading autocorrelation vs. theory, rain curves, time traces)
- `output/` — regenerated fine-grained noisy datasets (`*.parquet`, gitignored due to size — regenerate via `scripts/run_all.py full`; small `*_sample.csv` previews are included)
- `Dataset/` — the original clean scenario snapshots (same as the CSVs at repo root)
- `Dataset Python/` — a freshly regenerated run of the layer-1 simulator, for comparison against `Dataset/`
- `*.csv` (repo root) — original clean source datasets (layer-1 output, used as scenario input to layer 3)

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyarrow

# Layer 1: regenerate baseline scenario datasets
python "Script python/python_sim_all.py"

# Layer 3: regenerate fine-grained noisy time series from scenario datasets
python scripts/run_all.py sample   # fast: builds samples + validation/report.md
python scripts/run_all.py full     # full generation, all UEs, ~33M rows total
```

See `docs/noise_model_research.md` for the full layer-3 methodology.
