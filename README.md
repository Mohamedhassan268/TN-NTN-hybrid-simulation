# TN+NTN Noise Models

Physics/standards-grounded noise modeling for terrestrial (5G NR, WiFi 6) and non-terrestrial
(HAPS, LEO, UAV) communication links. Regenerates each static clean-data snapshot into a
realistic 6-second, 10 ms-resolution time series that propagates channel impairments coherently
through the full link chain: SNR → SINR → BER → throughput → packet loss → latency → link-quality index.

## Layout

- `noise_models/` — the reusable noise-model package (config, channel primitives, link chain, generation driver, validation)
- `scripts/run_all.py` — driver: `python scripts/run_all.py sample` (small per-tech samples + validation report) or `full` (complete generation)
- `docs/noise_model_research.md` — noise sources modeled, standards referenced, per-tech RF assumptions, chain equations, calibration results, and documented simplifications
- `validation/` — physical invariant checks and diagnostic plots (CDF overlays, fading autocorrelation vs. theory, rain curves, time traces)
- `output/` — regenerated noisy datasets (`*.parquet`, gitignored due to size — regenerate via `scripts/run_all.py full`; small `*_sample.csv` previews are included)
- `*.csv` (repo root) — original clean source datasets

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyarrow
python scripts/run_all.py sample   # fast: builds samples + validation/report.md
python scripts/run_all.py full     # full generation, all UEs, ~33M rows total
```

See `docs/noise_model_research.md` for the full methodology.
