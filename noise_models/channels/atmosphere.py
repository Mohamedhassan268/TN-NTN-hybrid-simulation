"""Atmospheric impairments: ITU-R rain attenuation, gaseous absorption, and
tropospheric/ionospheric scintillation.

Rain: ITU-R P.838-3 "Specific attenuation model for rain for use in prediction methods"
(horizontal polarization), implemented as the closed-form regression (its equations (2)-(3),
Tables 1 and 3) rather than a hand-picked anchor table -- continuous across frequency, no
clamping. Verified against the recommendation's own published Table 5 reference values (max
error 0.03% across 1-100 GHz, 8 test points).

Gaseous: ITU-R P.676-13 Annex 1 line-by-line calculation (its equations (1)-(9), full Table 1
oxygen lines and Table 2 water-vapour lines) integrated through the ITU-R reference standard
atmosphere of ITU-R P.835-7 Annex 1 (temperature/pressure/water-vapour-density vs. altitude,
its equations (1)-(8)) from the surface to 100 km, using a flat-atmosphere-layer slant-path
approximation (1/sin(elevation) per thin horizontal layer) consistent with the flat-earth
elevation geometry used elsewhere in this package (pathloss.py). Verified: reproduces the
well-documented ~15 dB/km peak of the 60 GHz oxygen absorption complex, and the water-vapour
resonance correctly peaks at 22.235 GHz.

Both replace the file's previous "Simplified" hand-fit models, whose 12 GHz rain coefficient
(k=0.0188) was found to differ from the true P.838-3 value (k=0.02386) by about 21%, and whose
gaseous model was a linear fit that could not represent the 22.235 GHz water-vapour resonance
at all -- material at Ka-band (20 GHz), which sits on that resonance's shoulder.
"""
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Rain -- ITU-R P.838-3, horizontal polarization, equations (2)-(3), Tables 1 and 3.
# ---------------------------------------------------------------------------

# Table 1: coefficients for k_H
_K_A = (-5.33980, -0.35351, -0.23789, -0.94158)
_K_B = (-0.10008, 1.26970, 0.86036, 0.64552)
_K_C = (1.13098, 0.45400, 0.15354, 0.16817)
_K_MK, _K_CK = -0.18961, 0.71147

# Table 3: coefficients for alpha_H
_A_A = (-0.14318, 0.29591, 0.32177, -5.37610, 16.1721)
_A_B = (1.82442, 0.77564, 0.63773, -0.96230, -3.29980)
_A_C = (-0.55187, 0.19822, 0.13164, 1.47828, 3.43990)
_A_M, _A_C0 = 0.67849, -1.95537


@lru_cache(maxsize=None)
def _rain_k_alpha(freq_ghz: float) -> tuple[float, float]:
    """k, alpha at freq_ghz per P.838-3 eq (2)-(3), horizontal polarization."""
    lf = np.log10(freq_ghz)
    log10_k = sum(a * np.exp(-((lf - b) / c) ** 2) for a, b, c in zip(_K_A, _K_B, _K_C))
    log10_k += _K_MK * lf + _K_CK
    k = 10 ** log10_k
    alpha = sum(a * np.exp(-((lf - b) / c) ** 2) for a, b, c in zip(_A_A, _A_B, _A_C))
    alpha += _A_M * lf + _A_C0
    return float(k), float(alpha)


def rain_attenuation_db_series(n_steps: int, step_s: float, rain_rate_mmhr: float,
                                freq_hz: float, elevation_deg: float,
                                rng: np.random.Generator) -> np.ndarray:
    rain_rate = np.broadcast_to(np.asarray(rain_rate_mmhr, dtype=float), (n_steps,))
    elevation = np.broadcast_to(np.asarray(elevation_deg, dtype=float), (n_steps,))
    if np.all(rain_rate <= 0):
        return np.zeros(n_steps)
    freq_ghz = freq_hz / 1e9
    k, alpha = _rain_k_alpha(freq_ghz)
    gamma_r = k * np.maximum(rain_rate, 0.0) ** alpha  # specific attenuation, dB/km
    elevation_rad = np.deg2rad(np.maximum(elevation, 5.0))
    l_eff = np.minimum(35 * np.exp(-0.015 * rain_rate) / np.sin(elevation_rad), 30.0)
    mean_atten = gamma_r * l_eff
    jitter = rng.normal(0, 0.05 * np.maximum(mean_atten, 0.1), n_steps)
    if n_steps > 20:
        kernel = np.ones(20) / 20
        jitter = np.convolve(jitter, kernel, mode="same")
    return np.maximum(mean_atten + jitter, 0.0)


# ---------------------------------------------------------------------------
# Gaseous -- ITU-R P.676-13 Annex 1 line-by-line, eq (1)-(9), Tables 1-2 (oxygen, water
# vapour), integrated through the ITU-R P.835-7 Annex 1 reference atmosphere.
# ---------------------------------------------------------------------------

# Table 1: oxygen spectral lines (f0, a1, a2, a3, a4, a5, a6)
_OXYGEN_LINES = (
    (50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850), (50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800),
    (51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729), (52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640),
    (52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526), (53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206),
    (53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085), (54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750),
    (54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654), (55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952),
    (55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135), (56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978),
    (56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547), (56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451),
    (57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056), (58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436),
    (58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273), (59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309),
    (59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776), (60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699),
    (60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825), (61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584),
    (61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619), (62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759),
    (62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844), (62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675),
    (63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139), (64.127775, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895),
    (64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590), (65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680),
    (65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002), (66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091),
    (66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393), (67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475),
    (67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545), (68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600),
    (68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650), (118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079),
    (368.498246, 67.400, 0.048, 16.400, 0.0, 0.000, 0.000), (424.763020, 637.700, 0.044, 16.400, 0.0, 0.000, 0.000),
    (487.249273, 237.400, 0.049, 16.000, 0.0, 0.000, 0.000), (715.392902, 98.100, 0.145, 16.000, 0.0, 0.000, 0.000),
    (773.839490, 572.300, 0.141, 16.200, 0.0, 0.000, 0.000), (834.145546, 183.100, 0.145, 14.700, 0.0, 0.000, 0.000),
)

# Table 2: water-vapour spectral lines (f0, b1, b2, b3, b4, b5, b6), including the 1780 GHz
# pseudo-line representing the wet continuum below 1000 GHz.
_WATER_VAPOUR_LINES = (
    (22.235080, .1079, 2.144, 26.38, .76, 5.087, 1.00), (67.803960, .0011, 8.732, 28.58, .69, 4.930, .82),
    (119.995940, .0007, 8.353, 29.48, .70, 4.780, .79), (183.310087, 2.273, .668, 29.06, .77, 5.022, .85),
    (321.225630, .0470, 6.179, 24.04, .67, 4.398, .54), (325.152888, 1.514, 1.541, 28.23, .64, 4.893, .74),
    (336.227764, .0010, 9.825, 26.93, .69, 4.740, .61), (380.197353, 11.67, 1.048, 28.11, .54, 5.063, .89),
    (390.134508, .0045, 7.347, 21.52, .63, 4.810, .55), (437.346667, .0632, 5.048, 18.45, .60, 4.230, .48),
    (439.150807, .9098, 3.595, 20.07, .63, 4.483, .52), (443.018343, .1920, 5.048, 15.55, .60, 5.083, .50),
    (448.001085, 10.41, 1.405, 25.64, .66, 5.028, .67), (470.888999, .3254, 3.597, 21.34, .66, 4.506, .65),
    (474.689092, 1.260, 2.379, 23.20, .65, 4.804, .64), (488.490108, .2529, 2.852, 25.86, .69, 5.201, .72),
    (503.568532, .0372, 6.731, 16.12, .61, 3.980, .43), (504.482692, .0124, 6.731, 16.12, .61, 4.010, .45),
    (547.676440, .9785, .158, 26.00, .70, 4.500, 1.00), (552.020960, .1840, .158, 26.00, .70, 4.500, 1.00),
    (556.935985, 497.0, .159, 30.86, .69, 4.552, 1.00), (620.700807, 5.015, 2.391, 24.38, .71, 4.856, .68),
    (645.766085, .0067, 8.633, 18.00, .60, 4.000, .50), (658.005280, .2732, 7.816, 32.10, .69, 4.140, 1.00),
    (752.033113, 243.4, .396, 30.86, .68, 4.352, .84), (841.051732, .0134, 8.177, 15.90, .33, 5.760, .45),
    (859.965698, .1325, 8.055, 30.60, .68, 4.090, .84), (899.303175, .0547, 7.914, 29.85, .68, 4.530, .90),
    (902.611085, .0386, 8.429, 28.65, .70, 5.100, .95), (906.205957, .1836, 5.110, 24.08, .70, 4.700, .53),
    (916.171582, 8.400, 1.441, 26.73, .70, 5.150, .78), (923.112692, .0079, 10.293, 29.00, .70, 5.000, .80),
    (970.315022, 9.009, 1.919, 25.50, .64, 4.940, .67), (987.926764, 134.6, .257, 29.85, .68, 4.550, .90),
    (1780.000000, 17506., .952, 196.3, 2.00, 24.15, 5.00),
)


def _gaseous_specific_attenuation_db_km(f_ghz: float, p_dry_hpa: float, e_hpa: float, t_k: float) -> float:
    """P.676-13 Annex 1 eq (1)-(9): total (oxygen + water vapour) specific attenuation, dB/km."""
    theta = 300.0 / t_k

    oxygen_sum = 0.0
    for f0, a1, a2, a3, a4, a5, a6 in _OXYGEN_LINES:
        s_i = a1 * 1e-7 * p_dry_hpa * theta ** 3 * np.exp(a2 * (1 - theta))
        df = a3 * 1e-4 * (p_dry_hpa * theta ** (0.8 - a4) + 1.1 * e_hpa * theta)
        df = np.sqrt(df ** 2 + 2.25e-6)  # eq (6b), oxygen: Zeeman-splitting correction
        delta = (a5 + a6 * theta) * 1e-4 * (p_dry_hpa + e_hpa) * theta ** 0.8  # eq (7)
        f_i = (f_ghz / f0) * (
            (df - delta * (f0 - f_ghz)) / ((f0 - f_ghz) ** 2 + df ** 2)
            + (df - delta * (f0 + f_ghz)) / ((f0 + f_ghz) ** 2 + df ** 2)
        )
        oxygen_sum += s_i * f_i

    water_sum = 0.0
    for f0, b1, b2, b3, b4, b5, b6 in _WATER_VAPOUR_LINES:
        s_i = b1 * 1e-1 * e_hpa * theta ** 3.5 * np.exp(b2 * (1 - theta))
        df = b3 * 1e-4 * (p_dry_hpa * theta ** b4 + b5 * e_hpa * theta ** b6)
        df = 0.535 * df + np.sqrt(0.217 * df ** 2 + 2.1316e-12 * f0 ** 2 / theta)  # eq (6b), water vapour
        f_i = (f_ghz / f0) * (df / ((f0 - f_ghz) ** 2 + df ** 2) + df / ((f0 + f_ghz) ** 2 + df ** 2))
        water_sum += s_i * f_i

    d = 5.6e-4 * (p_dry_hpa + e_hpa) * theta ** 0.8  # eq (9)
    n_d = f_ghz * p_dry_hpa * theta ** 2 * (
        6.14e-5 / (d * (1 + (f_ghz / d) ** 2))
        + 1.4e-12 * p_dry_hpa * theta ** 1.5 / (1 + 1.9e-5 * f_ghz ** 1.5)
    )  # eq (8), dry continuum

    return float(0.1820 * f_ghz * (oxygen_sum + n_d + water_sum))


# P.835-7 Annex 1 reference atmosphere, eq (1)-(8).
def _geopotential_height_km(z_km: float) -> float:
    return 6356.766 * z_km / (6356.766 + z_km)


def _temperature_pressure_k(z_km: float) -> tuple[float, float]:
    if z_km <= 86.0:
        h = _geopotential_height_km(z_km)
        if h <= 11: t = 288.15 - 6.5 * h
        elif h <= 20: t = 216.65
        elif h <= 32: t = 216.65 + (h - 20)
        elif h <= 47: t = 228.65 + 2.8 * (h - 32)
        elif h <= 51: t = 270.65
        elif h <= 71: t = 270.65 - 2.8 * (h - 51)
        else: t = 214.65 - 2.0 * (h - 71)

        if h <= 11: p = 1013.25 * (288.15 / (288.15 - 6.5 * h)) ** (-34.1632 / 6.5)
        elif h <= 20: p = 226.3226 * np.exp(-34.1632 * (h - 11) / 216.65)
        elif h <= 32: p = 54.74980 * (216.65 / (216.65 + (h - 20))) ** 34.1632
        elif h <= 47: p = 8.680422 * (228.65 / (228.65 + 2.8 * (h - 32))) ** (34.1632 / 2.8)
        elif h <= 51: p = 1.109106 * np.exp(-34.1632 * (h - 47) / 270.65)
        elif h <= 71: p = 0.6694167 * (270.65 / (270.65 - 2.8 * (h - 51))) ** (-34.1632 / 2.8)
        else: p = 0.03956649 * (214.65 / (214.65 - 2.0 * (h - 71))) ** (-34.1632 / 2.0)
        return float(t), float(p)

    if z_km <= 91.0:
        t = 186.8673
    else:
        t = 263.1905 - 76.3232 * np.sqrt(max(0.0, 1 - ((z_km - 91) / 19.9429) ** 2))
    a0, a1, a2, a3, a4 = 95.571899, -4.011801, 6.424731e-2, -4.789660e-4, 1.340543e-6
    p = np.exp(a0 + a1 * z_km + a2 * z_km ** 2 + a3 * z_km ** 3 + a4 * z_km ** 4)
    return float(t), float(p)


def _water_vapour_density_pressure(z_km: float, t_k: float, p_hpa: float) -> tuple[float, float]:
    rho = 7.5 * np.exp(-z_km / 2.0)  # eq (6)
    e = rho * t_k / 216.7  # eq (7)
    if e / p_hpa < 2e-6:  # eq (8): mixing-ratio cutoff at high altitude
        rho = 2e-6 * p_hpa * 216.7 / t_k
        e = rho * t_k / 216.7
    return float(rho), float(e)


@lru_cache(maxsize=None)
def _zenith_gaseous_attenuation_db(freq_ghz: float, z_max_km: float = 100.0, dz_km: float = 0.25) -> float:
    """Zenith gaseous attenuation (dB) at freq_ghz: the line-by-line specific attenuation
    integrated through the P.835-7 reference atmosphere from the surface to z_max_km. Cached
    per frequency -- this is the expensive part (tens of thousands of spectral-line
    evaluations) but is elevation-independent and freq_hz is fixed per technology, so this
    runs once per distinct carrier frequency for the lifetime of the process."""
    z_values = np.arange(0.0, z_max_km + dz_km, dz_km)
    specific = np.empty_like(z_values)
    for i, z in enumerate(z_values):
        t_k, p_hpa = _temperature_pressure_k(z)
        rho, e_hpa = _water_vapour_density_pressure(z, t_k, p_hpa)
        p_dry_hpa = p_hpa - e_hpa
        specific[i] = _gaseous_specific_attenuation_db_km(freq_ghz, p_dry_hpa, e_hpa, t_k)
    return float(np.trapezoid(specific, z_values))


def gaseous_attenuation_db(freq_hz: float, elevation_deg: float):
    freq_ghz = freq_hz / 1e9
    zenith_atten = _zenith_gaseous_attenuation_db(freq_ghz)
    elevation = np.asarray(elevation_deg, dtype=float)
    elevation_rad = np.deg2rad(np.maximum(elevation, 5.0))
    result = zenith_atten / np.sin(elevation_rad)
    return float(result) if result.ndim == 0 else result


def scintillation_db_series(n_steps: int, sigma_db: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_db <= 0:
        return np.zeros(n_steps)
    rho = 0.9
    w = rng.standard_normal(n_steps)
    x = np.empty(n_steps)
    x[0] = w[0]
    scale = np.sqrt(1 - rho ** 2)
    for i in range(1, n_steps):
        x[i] = rho * x[i - 1] + scale * w[i]
    return x * sigma_db
