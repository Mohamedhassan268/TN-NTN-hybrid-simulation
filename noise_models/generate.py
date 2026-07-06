"""Driver: reads a clean CSV, synthesizes a per-UE 6 s @ 10 ms noisy trace using the
link chain, and writes the result to Parquet (+ a small CSV sample for eyeballing)."""
import numpy as np
import pandas as pd
from dataclasses import replace

from .technologies import resolve_tech_name, build_scenario, get_tech_config
from .link import simulate_trace

TRACE_COLUMNS = [
    "distance_km", "SNR_dB", "SINR_dB", "RSSI_dBm", "BER", "Throughput_Mbps",
    "Latency_ms", "Packet_Loss_pct", "Doppler_Hz", "Propagation_Delay_ms",
    "Rain_Rate_mmhr", "Rain_Fade_dB", "Link_Quality_Index", "Spectral_Efficiency_bps_hz",
]


def generate_noisy_dataframe(clean_df: pd.DataFrame, file_name: str, seed: int = 42,
                              max_ues: int | None = None, n_steps: int | None = None,
                              step_s: float | None = None) -> pd.DataFrame:
    df = clean_df if max_ues is None else clean_df.iloc[:max_ues]
    frames = []
    for i, (_, row) in enumerate(df.iterrows()):
        tech_name = resolve_tech_name(file_name, row)
        cfg = get_tech_config(tech_name)
        if n_steps is not None or step_s is not None:
            cfg = replace(cfg, n_steps=n_steps or cfg.n_steps, step_s=step_s or cfg.step_s)
        scenario = build_scenario(row, tech_name)
        rng = np.random.default_rng(seed * 1_000_003 + i)
        trace = simulate_trace(scenario, cfg, rng)

        n = cfg.n_steps
        ue_id = row["UE_ID"]
        out = {
            "row_id": np.full(n, row.name),   # unique per source row; UE_ID repeats across scenario draws in most files
            "UE_ID": np.full(n, ue_id),
            "time_s": np.arange(n) * cfg.step_s,
            "network_type": np.full(n, tech_name),
        }
        out.update({col: trace[col] for col in TRACE_COLUMNS})
        frames.append(pd.DataFrame(out))
    return pd.concat(frames, ignore_index=True)


def write_outputs(noisy_df: pd.DataFrame, out_stem: str, sample_rows: int = 20):
    noisy_df.to_parquet(f"{out_stem}.parquet", index=False)
    sample_ids = noisy_df["row_id"].unique()[:sample_rows]
    noisy_df[noisy_df["row_id"].isin(sample_ids)].to_csv(f"{out_stem}_sample.csv", index=False)
