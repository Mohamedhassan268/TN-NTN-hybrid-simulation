"""The link chain: composes all channel primitives and propagates noise coherently
from received power through to SNR -> SINR -> BER -> throughput -> packet loss ->
latency -> link-quality-index -> spectral efficiency, for a single UE's 6 s trace.
"""
import numpy as np
from scipy.special import erfc

from .constants import C_LIGHT, shannon_capacity_bps_hz
from .channels import thermal, pathloss, fading, doppler, atmosphere, interference, impairments

# Simplified adaptive-modulation table: (M, code_rate, min_required_SINR_dB)
# Ordered ascending; the link picks the highest-throughput entry it can sustain.
_MCS_TABLE = [
    (4, 0.40, -3.0),
    (4, 0.80, 2.0),
    (16, 0.60, 8.0),
    (16, 0.90, 12.0),
    (64, 0.70, 16.0),
    (64, 0.90, 20.0),
    (256, 0.70, 24.0),
    (256, 0.90, 28.0),
    (1024, 0.80, 32.0),
]


def _select_mcs(sinr_db: np.ndarray, order_max: int):
    table = [e for e in _MCS_TABLE if e[0] <= order_max]
    m = np.full(sinr_db.shape, table[0][0], dtype=float)
    rate = np.full(sinr_db.shape, table[0][1], dtype=float)
    thresh = np.full(sinr_db.shape, table[0][2], dtype=float)
    for order, code_rate, th in table:
        mask = sinr_db >= th
        m[mask] = order
        rate[mask] = code_rate
        thresh[mask] = th
    return m, rate, thresh


_BLER_STEEPNESS_DB = 1.5
_BLER_LN9 = np.log(9)


def _bler_waterfall(sinr_db: np.ndarray, mcs_thresh_db: np.ndarray) -> np.ndarray:
    """Logistic BLER-vs-SINR waterfall referenced to each MCS's operating threshold
    (BLER=10% at threshold), matching typical coded-system link curves far better than
    raw uncoded-BER-per-packet-bit, which is unrealistically steep without an FEC model."""
    exponent = (sinr_db - mcs_thresh_db + _BLER_STEEPNESS_DB * _BLER_LN9) / _BLER_STEEPNESS_DB
    return np.clip(1.0 / (1.0 + np.exp(exponent)), 0.0, 1.0)


def _qam_ber(sinr_db: np.ndarray, m: np.ndarray) -> np.ndarray:
    sinr_lin = 10 ** (sinr_db / 10)
    log2m = np.log2(m)
    log2m = np.maximum(log2m, 1e-6)
    arg = np.sqrt(np.maximum(3 * log2m / (m - 1) * sinr_lin, 0.0))
    q = 0.5 * erfc(arg / np.sqrt(2))
    ber = (4.0 / log2m) * (1 - 1 / np.sqrt(m)) * q
    return np.clip(ber, 0.0, 0.5)


def simulate_trace(scenario: dict, cfg, rng: np.random.Generator) -> dict:
    """scenario keys: distance_km, doppler_char_hz, rain_rate_mmhr, altitude_m (optional),
    speed_ms (optional), platform_altitude_km. The trace duration is defined by
    the supplied versioned technology configuration."""
    n, dt = cfg.n_steps, cfg.step_s

    def series(value):
        return np.broadcast_to(np.asarray(value, dtype=float), (n,))

    distance_km = series(scenario["distance_km"])
    velocity_ms = series(scenario.get("speed_ms", 1.0))

    # --- geometry / large-scale ---
    elev_deg = series(scenario.get(
        "elevation_deg",
        pathloss.elevation_deg(distance_km, scenario.get("platform_altitude_km", 0.0)),
    ))
    alt_m = scenario.get("altitude_m")
    pl_db = pathloss.path_loss_db(
        distance_km,
        cfg.carrier_freq_hz,
        tech_name=cfg.name,
        los=scenario.get("los", True),
        altitude_m=alt_m,
        clutter_loss_db=scenario.get("clutter_loss_db", 0.0),
        wall_loss_db=scenario.get("wall_loss_db", 0.0),
    )
    shadow_sigma = cfg.shadowing_sigma_db
    if alt_m is not None:  # low-altitude UAV sees richer multipath -> larger shadowing variance
        shadow_sigma = cfg.shadowing_sigma_db * (
            1.0 + np.exp(-float(np.asarray(alt_m, dtype=float).mean()) / 300.0)
        )
    shadow_db = pathloss.shadowing_db_series(
        n, dt, shadow_sigma, float(velocity_ms.mean()), decorr_distance_m=20.0, rng=rng
    )

    # --- atmosphere ---
    rain_db = np.zeros(n)
    if cfg.rain_capable:
        rain_db = atmosphere.rain_attenuation_db_series(
            n, dt, scenario.get("rain_rate_mmhr", 0.0), cfg.carrier_freq_hz, elev_deg, rng)
    gas_db = atmosphere.gaseous_attenuation_db(cfg.carrier_freq_hz, elev_deg) if cfg.rain_capable else 0.0
    scint_db = np.zeros(n)
    if cfg.scintillation_capable:
        scint_db = atmosphere.scintillation_db_series(n, sigma_db=0.4, rng=rng)

    # --- small-scale fading ---
    fd_input = np.asarray(scenario.get("doppler_char_hz", 1.0), dtype=float)
    if fd_input.ndim == 0:
        fd_hz_series = doppler.doppler_series_hz(n, dt, float(fd_input), cfg.name, rng)
    else:
        fd_hz_series = series(fd_input) + rng.normal(
            0.0, np.maximum(np.abs(series(fd_input)) * 0.02, 0.5), n
        )
    fade_db = fading.fading_gain_db(n, dt, float(np.mean(np.abs(fd_hz_series)) + 1e-3),
                                     cfg.fading_type, cfg.rician_k_db, rng)

    # --- received power / SNR ---
    prx_dbm = (cfg.tx_power_dbm + cfg.antenna_gain_db - pl_db - shadow_db
               - rain_db - gas_db - scint_db + fade_db)
    noise_dbm = thermal.noise_floor_dbm(cfg.bandwidth_hz, cfg.noise_figure_db)
    snr_db = prx_dbm - noise_dbm

    # --- interference -> SINR, then hardware ceiling ---
    gap_db = interference.sinr_gap_db(n, cfg.interference_margin_db, rng)
    sinr_db = snr_db - gap_db
    sinr_db = impairments.apply_hw_ceiling(sinr_db, cfg.hw_sinr_ceiling_db)
    sinr_db = np.minimum(sinr_db, snr_db)  # SINR can never exceed SNR

    # --- adaptive modulation -> BER (diagnostic, pre-FEC) and BLER (post-FEC, drives loss/throughput) ---
    m_order, code_rate, mcs_thresh_db = _select_mcs(sinr_db, cfg.modulation_order_max)
    ber = _qam_ber(sinr_db, m_order)
    bler = _bler_waterfall(sinr_db, mcs_thresh_db)

    # --- throughput (Shannon-capped adaptive modulation) ---
    se_mod = np.log2(m_order) * code_rate
    se_shannon = shannon_capacity_bps_hz(sinr_db)
    se_bps_hz = np.minimum(se_mod, se_shannon)
    eff_mean_db = 10 * np.log10(max(cfg.throughput_efficiency, 1e-6))
    eff_jitter = rng.normal(0, cfg.throughput_efficiency_sigma_db, n)
    if n > 15:
        kernel = np.ones(15) / 15
        eff_jitter = np.convolve(eff_jitter, kernel, mode="same")
    efficiency = np.clip(10 ** ((eff_mean_db + eff_jitter) / 10), 0.0, 1.0)
    throughput_mbps = se_bps_hz * cfg.bandwidth_hz / 1e6 * (1 - bler) * efficiency

    # --- packet loss (PHY BLER + technology-specific extra loss) ---
    extra_loss_frac = np.zeros(n)
    if cfg.name.startswith("WiFi"):
        burst_db = interference.contention_burst_loss_db(n, rng)
        extra_loss_frac = np.clip(burst_db / 40.0, 0.0, 1.0)
    elif cfg.name.startswith("LEO"):
        outage = (rng.random(n) < 0.003).astype(float)  # rare handover/outage events
        extra_loss_frac = outage
    packet_loss_pct = np.clip((bler + extra_loss_frac - bler * extra_loss_frac) * 100.0, 0.0, 100.0)

    # --- latency ---
    prop_delay_ms = (distance_km * 1000.0 / C_LIGHT) * 1000.0
    proc_delay_ms = 0.3 if cfg.name.startswith("WiFi") else (1.0 if cfg.name == "5G_NR" else 2.0)
    queue_jitter_ms = rng.exponential(scale=0.05 + packet_loss_pct / 100.0, size=n)
    latency_ms = prop_delay_ms + proc_delay_ms + queue_jitter_ms

    # --- link quality index (0-100 composite) ---
    sinr_norm = np.clip((sinr_db + 10) / 40.0, 0.0, 1.0) * 100.0
    loss_penalty = np.clip(packet_loss_pct * 1.5, 0.0, 100.0)
    lqi = np.clip(sinr_norm - 0.5 * loss_penalty, 0.0, 100.0)

    return {
        "distance_km": distance_km,
        "SNR_dB": snr_db,
        "SINR_dB": sinr_db,
        "RSSI_dBm": prx_dbm,
        "BER": ber,
        "BLER": bler,
        "Throughput_Mbps": throughput_mbps,
        "Latency_ms": latency_ms,
        "Packet_Loss_pct": packet_loss_pct,
        "Doppler_Hz": fd_hz_series,
        "Propagation_Delay_ms": prop_delay_ms,
        "Rain_Rate_mmhr": series(scenario.get("rain_rate_mmhr", 0.0)),
        "Rain_Fade_dB": rain_db,
        "Link_Quality_Index": lqi,
        "Spectral_Efficiency_bps_hz": se_bps_hz,
        "MCS_Order": m_order,
        "Code_Rate": code_rate,
        "Spectral_Efficiency_Saturated": np.isclose(se_bps_hz, se_mod),
    }
