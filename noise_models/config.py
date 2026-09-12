"""Per-technology physical configuration.

Each clean CSV row already encodes a *scenario* through a handful of geometry /
environment columns (distance, a characteristic Doppler value, rain rate,
altitude, speed, band). These are treated as the ground-truth scenario
parameters that drive the physics; the *other* metric columns (SNR, BER,
throughput, ...) are recomputed from scratch by the link chain and only used
afterwards as calibration/validation references.
"""
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TechConfig:
    name: str
    carrier_freq_hz: float
    bandwidth_hz: float
    noise_figure_db: float
    tx_power_dbm: float
    fading_type: str          # "rayleigh" or "rician"
    rician_k_db: float        # ignored if rayleigh
    shadowing_sigma_db: float
    rain_capable: bool
    scintillation_capable: bool
    interference_margin_db: float   # typical SNR->SINR gap from co/adjacent-channel interference
    modulation_order_max: int       # cap for max-order M-QAM used in BER model
    antenna_gain_db: float          # combined Tx+Rx antenna gain minus implementation/clutter losses
    hw_sinr_ceiling_db: float       # hardware EVM-limited SINR ceiling
    throughput_efficiency: float = 1.0       # mean fraction of Shannon-capped PHY throughput realized
    throughput_efficiency_sigma_db: float = 1.0  # time-varying congestion/MAC-overhead variability
    step_s: float = 0.010
    n_steps: int = 600


# Base configs. tx_power_dbm is calibrated (in the sample-validation phase)
# so regenerated SNR overlaps the observed clean-data ranges per file.
TECH_CONFIGS = {
    "5G_NR": TechConfig(
        name="5G_NR", carrier_freq_hz=3.5e9, bandwidth_hz=100e6,
        noise_figure_db=7.0, tx_power_dbm=43.0,
        fading_type="rayleigh", rician_k_db=0.0,
        shadowing_sigma_db=4.0, rain_capable=True, scintillation_capable=False,
        interference_margin_db=4.0, modulation_order_max=256,
        antenna_gain_db=-8.0, hw_sinr_ceiling_db=70.0,
        throughput_efficiency=0.90, throughput_efficiency_sigma_db=1.0,
    ),
    "WiFi6_2.4GHz": TechConfig(
        name="WiFi6_2.4GHz", carrier_freq_hz=2.437e9, bandwidth_hz=40e6,
        noise_figure_db=8.0, tx_power_dbm=20.0,
        fading_type="rician", rician_k_db=6.0,
        shadowing_sigma_db=3.0, rain_capable=False, scintillation_capable=False,
        interference_margin_db=6.0, modulation_order_max=1024,
        antenna_gain_db=-15.0, hw_sinr_ceiling_db=85.0,
        throughput_efficiency=0.05, throughput_efficiency_sigma_db=3.0,
    ),
    "WiFi6_5GHz": TechConfig(
        name="WiFi6_5GHz", carrier_freq_hz=5.18e9, bandwidth_hz=80e6,
        noise_figure_db=8.0, tx_power_dbm=20.0,
        fading_type="rician", rician_k_db=6.0,
        shadowing_sigma_db=3.0, rain_capable=False, scintillation_capable=False,
        interference_margin_db=5.0, modulation_order_max=1024,
        antenna_gain_db=-15.0, hw_sinr_ceiling_db=85.0,
        throughput_efficiency=0.05, throughput_efficiency_sigma_db=3.0,
    ),
    "WiFi6E_6GHz": TechConfig(
        name="WiFi6E_6GHz", carrier_freq_hz=6.425e9, bandwidth_hz=160e6,
        noise_figure_db=8.0, tx_power_dbm=20.0,
        fading_type="rician", rician_k_db=8.0,
        shadowing_sigma_db=3.0, rain_capable=False, scintillation_capable=False,
        interference_margin_db=3.0, modulation_order_max=1024,
        antenna_gain_db=-15.0, hw_sinr_ceiling_db=85.0,
        throughput_efficiency=0.05, throughput_efficiency_sigma_db=3.0,
    ),
    "HAPS": TechConfig(
        name="HAPS", carrier_freq_hz=2.1e9, bandwidth_hz=30e6,
        noise_figure_db=3.0, tx_power_dbm=53.0,
        fading_type="rician", rician_k_db=12.0,
        shadowing_sigma_db=3.5, rain_capable=True, scintillation_capable=True,
        interference_margin_db=3.0, modulation_order_max=256,
        antenna_gain_db=4.0, hw_sinr_ceiling_db=38.0,
    ),
    "LEO": TechConfig(
        name="LEO", carrier_freq_hz=12.0e9, bandwidth_hz=250e6,
        noise_figure_db=2.0, tx_power_dbm=58.0,
        fading_type="rician", rician_k_db=10.0,
        shadowing_sigma_db=2.5, rain_capable=True, scintillation_capable=True,
        interference_margin_db=2.0, modulation_order_max=256,
        antenna_gain_db=36.0, hw_sinr_ceiling_db=16.0,
    ),
    "UAV": TechConfig(
        name="UAV", carrier_freq_hz=2.0e9, bandwidth_hz=40e6,
        noise_figure_db=5.0, tx_power_dbm=30.0,
        fading_type="rician", rician_k_db=9.0,
        shadowing_sigma_db=3.0, rain_capable=True, scintillation_capable=False,
        interference_margin_db=4.0, modulation_order_max=256,
        antenna_gain_db=8.0, hw_sinr_ceiling_db=55.0,
        throughput_efficiency=0.55, throughput_efficiency_sigma_db=2.0,
    ),
}

# Ka-band (20 GHz) LEO variant for the controlled band-sensitivity study. Clone of "LEO"
# above, changing ONLY what is physically frequency-dependent:
#   - carrier_freq_hz: 12.0 -> 20.0 GHz
#   - antenna_gain_db: +20*log10(20/12) dB, the fixed-aperture gain scaling with frequency
# Bandwidth, noise figure, Tx power, Rician K, shadowing, interference margin, modulation
# order, and SINR ceiling are held constant -- these are design/component choices, not
# derivable from frequency alone, so inventing new values for them would reintroduce exactly
# the kind of unverified "generic numbers" this rerun exists to eliminate. Holding them fixed
# is also what makes this a controlled comparison: only frequency-dependent physics
# (path loss in pathloss.py, rain/gaseous attenuation in atmosphere.py, and this antenna
# gain term) differs between "LEO" and "LEO_Ka".
_LEO_KA_ANTENNA_GAIN_DELTA_DB = 20.0 * math.log10(20.0e9 / TECH_CONFIGS["LEO"].carrier_freq_hz)
TECH_CONFIGS["LEO_Ka"] = replace(
    TECH_CONFIGS["LEO"],
    name="LEO_Ka",
    carrier_freq_hz=20.0e9,
    antenna_gain_db=TECH_CONFIGS["LEO"].antenna_gain_db + _LEO_KA_ANTENNA_GAIN_DELTA_DB,
)

# S-band (2.2 GHz) LEO variant, same clone-and-derive pattern as "LEO_Ka" above. 2.2 GHz is the
# ITU space-operation/space-research downlink allocation (2200-2290 MHz), the band actually used
# for LEO TT&C and narrowband constellations (e.g. Orbcomm) -- NOT representative of the
# high-throughput broadband links this dataset's Throughput_Mbps values model; that caveat must
# accompany any S-band result. Only carrier_freq_hz and the fixed-aperture antenna_gain_db differ
# from "LEO"; everything else held constant for the same controlled-comparison reason as Ka.
_LEO_S_ANTENNA_GAIN_DELTA_DB = 20.0 * math.log10(2.2e9 / TECH_CONFIGS["LEO"].carrier_freq_hz)
TECH_CONFIGS["LEO_S"] = replace(
    TECH_CONFIGS["LEO"],
    name="LEO_S",
    carrier_freq_hz=2.2e9,
    antenna_gain_db=TECH_CONFIGS["LEO"].antenna_gain_db + _LEO_S_ANTENNA_GAIN_DELTA_DB,
)


def get_config(tech_name: str) -> TechConfig:
    return TECH_CONFIGS[tech_name]


def with_deltas(tech_name: str, tx_power_delta_db: float = 0.0, antenna_gain_delta_db: float = 0.0) -> TechConfig:
    """Return a copy of a tech config with link-budget knobs shifted (used for calibration)."""
    base = TECH_CONFIGS[tech_name]
    return replace(
        base,
        tx_power_dbm=base.tx_power_dbm + tx_power_delta_db,
        antenna_gain_db=base.antenna_gain_db + antenna_gain_delta_db,
    )
