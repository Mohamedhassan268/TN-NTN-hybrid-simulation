# TN+NTN Noise Model — Research & Implementation Notes

## 1. Purpose

The source CSVs (`TN_Data_5G.csv`, `TN_Data_WIFI_6.csv`, `NTNData_HAPS.csv`, `NTN_Data_LEO.csv`,
`Data_UAV.csv`) are static, idealized per-UE snapshots. This project regenerates each dataset as a
**6-second, 10 ms-resolution time series per UE**, synthesized from a physics/standards-grounded
channel model so the data reflects the impairments a real link actually experiences: thermal noise,
path loss, shadowing, multipath fading, Doppler, rain/gaseous/scintillation attenuation,
interference, congestion, and hardware limits — propagated coherently through the full link chain
(SNR → SINR → BER → throughput → packet loss → latency → link-quality index).

Each source row is treated as a **scenario descriptor**, not a value to copy: its geometry/environment
columns (`distance_km`, `Doppler_Hz`, `Rain_Rate_mmhr`, `altitude_m`, `speed_ms`, band/`sat_type`)
seed the physical model; every other column (SNR, SINR, RSSI, BER, throughput, latency, packet loss,
LQI, spectral efficiency) is recomputed from first principles.

## 2. Noise/impairment sources modeled, per technology

| Source | Standard / model | Applies to |
|---|---|---|
| Thermal noise (kTB) | Johnson–Nyquist | all |
| Free-space + log-distance path loss | Friis / FSPL | all |
| Log-normal shadowing (time-correlated) | Gudmundson AR(1), decorrelation distance | all |
| Small-scale multipath fading | Rayleigh (NLOS, 5G urban) / Rician (LOS-dominated: WiFi indoor, HAPS, LEO, UAV), Clarke/Jakes-shaped time correlation via `J0(2π·fd·Δt)` | all |
| Doppler shift (+ Doppler rate for LEO passes) | Geometry/velocity-derived, smoothed jitter, linear rate drift toward zero for LEO | all, most pronounced UAV/LEO |
| Rain attenuation | ITU-R P.618 (simplified k, α specific-attenuation coefficients, effective path length) | 5G, HAPS, LEO, UAV (rain-exposed outdoor links) |
| Gaseous absorption | ITU-R P.676 (simplified zenith-attenuation trend, elevation-scaled) | same as rain-capable |
| Tropospheric/ionospheric scintillation | AR(1) fast fluctuation | HAPS, LEO |
| Co-/adjacent-channel interference | Time-varying SNR→SINR gap | all |
| CSMA contention/collision bursts | Bernoulli burst-loss process | WiFi only |
| LEO handover/outage | Rare Bernoulli outage events | LEO only |
| Hardware EVM/ADC/phase-noise ceiling | SINR ceiling combined via parallel-noise (harmonic) formula | all |
| MAC/congestion throughput efficiency | Time-varying multiplicative derate on Shannon-capped throughput | WiFi (heavy), UAV (moderate), 5G (light) |

## 3. Per-technology assumed RF parameters (documented, configurable in `noise_models/config.py`)

| Tech | Carrier | Bandwidth | Fading | Rain? | Notes |
|---|---|---|---|---|---|
| 5G NR | 3.5 GHz (FR1) | 100 MHz | Rayleigh (urban NLOS) | Yes (light) | antenna_gain −8 dB models clutter/indoor penetration margin |
| WiFi 6 (2.4/5/6 GHz) | per `network_type` sub-band | 40/80/160 MHz | Rician K=6–8 dB | No | heavy contention-driven throughput derate (0.05×) — see §5 |
| HAPS | S-band 2.1 GHz | 30 MHz | Rician K=12 dB | Yes + scintillation | stratospheric platform altitude 20 km |
| LEO | Ku-band 12 GHz | 250 MHz | Rician K=10 dB | Yes + scintillation | 550 km representative shell altitude; 36 dB combined VSAT+satellite antenna gain; large Doppler + Doppler-rate drift |
| UAV | 2 GHz (A2G) | 40 MHz | Rician K=9 dB | Yes (light) | altitude-dependent shadowing variance (more multipath at low altitude) |

## 4. Link chain (how noise propagates)

```
Prx(t) = Ptx + Gain − PathLoss − Shadowing(t) − Rain(t) − Gas − Scintillation(t) + FadingGain(t)
SNR(t) = Prx(t) − ThermalNoise
SINR(t) = min(SNR(t) − InterferenceGap(t), HW_EVM_ceiling ⊕ SNR(t))   [SINR ≤ SNR always]
MCS(t)  = highest adaptive-modulation entry whose min-SINR threshold ≤ SINR(t)
BER(t)  = closed-form M-QAM AWGN bit-error rate at SINR(t), MCS(t)          (diagnostic, pre-FEC)
BLER(t) = logistic waterfall referenced to MCS(t) threshold (BLER=10% at threshold, 1.5 dB steepness)
SE(t)   = min(log2(M)·coderate, Shannon(SINR(t)))
Throughput(t) = SE(t) · Bandwidth · (1−BLER(t)) · Efficiency(t)     [Efficiency: MAC/congestion derate]
PacketLoss(t) = BLER(t) blended with tech-specific extra loss (WiFi contention bursts, LEO outage)
Latency(t) = PropagationDelay(distance) + ProcessingDelay + QueueJitter(∝ PacketLoss)
LQI(t) = composite of normalized SINR and packet-loss penalty
```

This guarantees the chain-consistency requirement: every downstream metric is derived from the same
SINR trajectory, so a deep fade simultaneously depresses SNR/SINR, raises BER/BLER/packet-loss,
lowers throughput/SE, and raises latency/lowers LQI — as it would on a real link.

## 5. Known simplifications / honest caveats

- **Two-ray ground reflection** (UAV) is approximated via altitude-dependent shadowing variance
  rather than an explicit geometric two-ray interference pattern — a scope-driven simplification.
- **Interference** is modeled as a direct time-varying dB gap subtracted from SNR (guaranteeing
  SINR ≤ SNR) rather than an explicit interference-power sum; this is a standard link-abstraction
  simplification, not a full multi-cell interference simulation.
- **Original `Throughput_Mbps` in the HAPS/LEO source files exceeds the Shannon bound** for their
  stated bandwidths (e.g., HAPS max 1438 Mbps at ~30 MHz implies ~48 bps/Hz spectral efficiency,
  physically unreachable even with advanced MIMO). We treat those original values as non-physical
  placeholders and instead derive throughput self-consistently from the regenerated SINR via
  Shannon-capped adaptive modulation — this is the physically correct choice, at the cost of not
  numerically matching the original column for those two files. SNR, packet loss, and latency were
  still calibrated to closely track the original ranges.
- **WiFi/UAV throughput efficiency factors** (0.05 / 0.55 mean) are a documented modeling choice
  representing real-world MAC overhead and channel contention/congestion, calibrated so the
  regenerated throughput matches the observed order of magnitude — not derived from a first-principles
  contention simulation.
- Antenna gain terms (`antenna_gain_db` per tech) lump Tx/Rx antenna gain, cabling/implementation
  loss, and (for terrestrial techs) indoor/clutter penetration margin into one calibrated constant.

## 6. Calibration results (30–40 UE sample, 600 steps × 10 ms)

| Tech | SNR mean (ours / observed) | Packet loss % (ours / observed) |
|---|---|---|
| 5G NR | 22.5 / 23.5 | 3.76 / 3.09 |
| WiFi 6 | 28.7 / 25.4 | 7.17 / 6.89 |
| HAPS | 14.2 / 12.9 | 3.18 / 3.59 |
| LEO | 5.7 / 7.3 | 4.62 / 11.12 |
| UAV | 28.6 / 29.8 | 3.15 / 1.56 |

All within the same order of magnitude / a few dB of the profiled clean-data ranges, with full
internal physical consistency (SINR ≤ SNR always; BER/BLER monotonically decrease with SINR;
throughput ≤ Shannon bound).
