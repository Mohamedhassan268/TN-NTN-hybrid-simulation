# Standards conformance boundary

The link simulator implements selected channel-model components from 3GPP TR 38.901, TR 36.777, and TR 38.811 within their stated parameter ranges. ITU-R models provide atmospheric attenuation, while custom mobility, interference, and network-selection components are documented separately.

The system-level description is **standards-informed simulator**. Neither the complete simulator nor the generated dataset is described as “3GPP-compliant.” The machine-readable source of this matrix is `standards/conformance_matrix.json`.

| Source | Selected equation/component | Implementation and domain | Disclosed boundary/deviation | Deterministic vectors | External cross-check |
|---|---|---|---|---|---|
| 3GPP TR 38.901 v17.1.0, Table 7.4.1-1 | UMa LOS and NLOS path loss: `PL_NLOS = max(PL_LOS, PL_NLOS-prime)` | `uma_path_loss_db`; 3.5 GHz, 10 m distance floor, hBS 25 m, hUT 1.5 m | LOS state is generated separately; no claim for full spatially consistent cluster generation | 100 m LOS and 500 m NLOS, 0.1 dB tolerance | LLSim5G frozen-output comparison required before release |
| 3GPP TR 36.777 v15.0.0, Table B-2 | UMa-AV LOS and high-altitude NLOS terms | `uma_av_path_loss_db`; 2 GHz, UAV altitude 50–300 m | Elevation-dependent LOS probability is a disclosed scenario-generator choice | 1 km LOS at 100 m altitude, 0.1 dB tolerance | LLSim5G frozen-output comparison required before release |
| 3GPP TR 38.811 v15.4.0, Section 6.6 | Basic NTN path-loss component | `ntn_basic_path_loss_db`; S/Ku/Ka configurations, spherical-Earth slant range | Only FSPL plus an explicit clutter input is attributed to the report; fading, MCS, interference, and outage are custom | 600 km Ku plus 2.5 dB clutter, 0.1 dB tolerance | OpenNTN frozen-output comparison required before release |
| ITU-R P.838-3 | Horizontal-polarization rain specific attenuation | `_rain_k_alpha` and `rain_attenuation_db_series` | Effective path length and temporal jitter are disclosed simulator choices | Existing coefficient regression tests | Not a 3GPP claim |
| ITU-R P.676-13 and P.835-7 | Line-by-line gaseous attenuation through reference atmosphere | `_gaseous_specific_attenuation_db_km` and `_zenith_gaseous_attenuation_db` | Thin-layer flat-atmosphere slant approximation | Spectral-line/reference-atmosphere unit tests required | Not a 3GPP claim |
| IEEE/indoor propagation literature | Wi-Fi log-distance indoor loss | `wifi_indoor_path_loss_db`; one-meter reference, exponent 3, explicit wall loss | Deliberately outside every 3GPP claim | Decade-distance 30 dB vector | Independent Wi-Fi model |

The internal equation-vector test is run with:

```powershell
python scripts/run_physical_conformance.py
```

The optional `--external standards/external_reference_outputs.csv` input activates comparison against frozen OpenNTN/LLSim5G outputs. Missing external outputs remain an explicit release blocker rather than being silently treated as validation.

Authoritative sources: [3GPP specification archive](https://www.3gpp.org/ftp/Specs/archive/), [OpenNTN](https://github.com/ant-uni-bremen/OpenNTN), and [LLSim5G](https://github.com/EFontesP90/LLSim5G).
