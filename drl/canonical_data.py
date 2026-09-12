"""Loading and validation helpers for the canonical revision-v2 dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import AREAS, NETWORK_TYPES
from .scenarios import BANDS, DATASET_VERSION, EPISODE_STEPS

KEY = ["trajectory_id", "step_index", "band", "network_type"]
REQUIRED_COLUMNS = {
    "dataset_version", "scenario_id", "trajectory_id", "ue_id", "split",
    "step_index", "time_s", "decision_interval_s", "band", "geometry_regime",
    "area", "network_type", "available_mask", "area_allowed", "visible", "available",
    "distance_km", "elevation_deg", "altitude_m", "speed_mps", "doppler_hz",
    "rain_rate_mm_h", "rssi_dbm", "snr_db", "sinr_db", "ber", "bler",
    "throughput_mbps", "propagation_delay_ms", "latency_ms", "packet_loss_pct",
    "spectral_efficiency_bps_hz", "link_quality_index", "mcs_order", "code_rate",
    "spectral_efficiency_saturated", "log_throughput_mbps", "log_ber",
    "log_packet_loss_pct",
}


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    trajectories: int
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def load_canonical(path: str | Path, *, split: str | None = None, band: str | None = None):
    frame = pd.read_parquet(path, filters=[("split", "==", split)] if split else None)
    if band:
        frame = frame[frame["band"] == band]
    return frame.reset_index(drop=True)


def validate_canonical(frame: pd.DataFrame, *, complete: bool = False) -> ValidationReport:
    errors = []
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
        return ValidationReport(len(frame), frame.get("trajectory_id", pd.Series(dtype=str)).nunique(), tuple(errors))
    if not frame["dataset_version"].eq(DATASET_VERSION).all():
        errors.append("unexpected dataset_version")
    if frame.duplicated(KEY).any():
        errors.append("duplicate primary keys")
    counts = frame.groupby(["trajectory_id", "step_index", "band"])["network_type"].agg(
        ["size", "nunique"]
    )
    if not ((counts["size"] == 5) & (counts["nunique"] == 5)).all():
        errors.append("each trajectory-step-band must contain five distinct candidates")
    if set(frame["network_type"].unique()) - set(NETWORK_TYPES):
        errors.append("unknown network_type")
    if set(frame["area"].unique()) - set(AREAS):
        errors.append("unknown area")
    if set(frame["band"].unique()) - set(BANDS):
        errors.append("unknown band")
    available_metrics = frame.loc[frame["available"], [
        "rssi_dbm", "snr_db", "sinr_db", "ber", "bler", "throughput_mbps",
        "latency_ms", "packet_loss_pct", "spectral_efficiency_bps_hz",
    ]]
    if available_metrics.isna().any().any():
        errors.append("available links contain null outcomes")
    unavailable_metrics = frame.loc[~frame["available"], ["rssi_dbm", "sinr_db", "throughput_mbps"]]
    if unavailable_metrics.notna().any().any():
        errors.append("unavailable links must expose null outcomes")
    if not frame["decision_interval_s"].eq(10.0).all():
        errors.append("decision interval is not uniformly 10 seconds")
    if not np.allclose(
        frame["time_s"].to_numpy(dtype=float),
        frame["step_index"].to_numpy(dtype=float) * frame["decision_interval_s"].to_numpy(dtype=float),
    ):
        errors.append("time_s is inconsistent with the decision interval")
    if not frame["available_mask"].between(0, (1 << len(NETWORK_TYPES)) - 1).all():
        errors.append("availability mask is outside the canonical five-bit range")
    logical_availability = frame["area_allowed"] & frame["visible"]
    if not frame["available"].eq(logical_availability).all():
        errors.append("availability must equal area_allowed AND visible")
    action_index = frame["network_type"].map({name: index for index, name in enumerate(NETWORK_TYPES)})
    availability_bits = (frame["available"].to_numpy(dtype=np.int16) << action_index.to_numpy(dtype=np.int16))
    expected_masks = pd.Series(availability_bits, index=frame.index).groupby(
        [frame["trajectory_id"], frame["step_index"], frame["band"]], sort=False
    ).transform("sum")
    if not frame["available_mask"].eq(expected_masks).all():
        errors.append("availability mask does not match candidate flags")
    geometry_columns = ["distance_km", "elevation_deg", "altitude_m", "speed_mps", "doppler_hz", "rain_rate_mm_h"]
    if not np.isfinite(frame[geometry_columns].to_numpy(dtype=float)).all():
        errors.append("geometry contains non-finite values")
    if (frame["distance_km"] <= 0).any() or (frame["altitude_m"] < 0).any() or (frame["speed_mps"] < 0).any():
        errors.append("geometry contains impossible distance, altitude, or speed")
    if not frame["elevation_deg"].between(-90.0, 90.0).all():
        errors.append("elevation is outside physical angular bounds")
    if not frame["rain_rate_mm_h"].between(0.0, 50.0).all():
        errors.append("rain rate is outside configured range")
    if (available_metrics["sinr_db"] > available_metrics["snr_db"] + 1e-9).any():
        errors.append("SINR exceeds SNR")
    if not available_metrics["ber"].between(0.0, 0.5).all():
        errors.append("BER outside [0, 0.5]")
    if not available_metrics["bler"].between(0.0, 1.0).all():
        errors.append("BLER outside [0, 1]")
    if (available_metrics[["throughput_mbps", "latency_ms", "packet_loss_pct"]] < 0).any().any():
        errors.append("negative KPI")
    if (available_metrics["spectral_efficiency_bps_hz"] > 8.0 + 1e-9).any():
        errors.append("spectral efficiency exceeds configured MCS cap")
    split_members = frame.groupby("trajectory_id")["split"].nunique()
    if (split_members != 1).any():
        errors.append("trajectory appears in multiple splits")
    if complete:
        expected = {"train": 800, "validation": 200, "test": 200}
        actual = frame[["trajectory_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
        if actual != expected:
            errors.append(f"unexpected complete split counts: {actual}")
        steps = frame.groupby("trajectory_id")["step_index"].nunique()
        if not steps.eq(EPISODE_STEPS).all():
            errors.append("complete trajectories must contain 60 steps")
    return ValidationReport(len(frame), frame["trajectory_id"].nunique(), tuple(errors))


def assert_no_split_overlap(frame: pd.DataFrame) -> None:
    mapping = frame[["trajectory_id", "split"]].drop_duplicates()
    if mapping["trajectory_id"].duplicated().any():
        raise AssertionError("trajectory leakage across splits")
