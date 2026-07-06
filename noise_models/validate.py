"""Physical-sanity checks and validation plots comparing regenerated noisy traces
against the profiled clean-data ranges."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .channels.fading import fading_gain_db


def check_invariants(noisy_df: pd.DataFrame) -> dict:
    results = {}
    results["sinr_le_snr_violations"] = int((noisy_df["SINR_dB"] > noisy_df["SNR_dB"] + 1e-6).sum())
    results["nan_count"] = int(noisy_df.isna().sum().sum())
    results["throughput_negative"] = int((noisy_df["Throughput_Mbps"] < 0).sum())
    results["packet_loss_out_of_range"] = int(
        ((noisy_df["Packet_Loss_pct"] < 0) | (noisy_df["Packet_Loss_pct"] > 100)).sum())
    results["ber_out_of_range"] = int(((noisy_df["BER"] < 0) | (noisy_df["BER"] > 1)).sum())
    # BER should (on average) decrease as SINR increases: bin and check monotonic trend
    bins = pd.qcut(noisy_df["SINR_dB"], 10, duplicates="drop")
    ber_by_bin = noisy_df.groupby(bins, observed=True)["BER"].mean()
    diffs = np.diff(ber_by_bin.values)
    results["ber_monotonic_violations"] = int((diffs > 1e-6).sum())
    return results


def summarize_vs_clean(noisy_df: pd.DataFrame, clean_df: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in clean_df.columns:
            continue
        rows.append({
            "column": col,
            "clean_mean": clean_df[col].mean(),
            "noisy_mean": noisy_df[col].mean(),
            "clean_min": clean_df[col].min(),
            "noisy_min": noisy_df[col].min(),
            "clean_max": clean_df[col].max(),
            "noisy_max": noisy_df[col].max(),
        })
    return pd.DataFrame(rows)


def plot_cdf_overlay(noisy_df, clean_df, column, tech_label, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for data, label, style in [(clean_df[column].dropna(), "clean (snapshot)", "--"),
                                (noisy_df[column].dropna(), "regenerated (time series)", "-")]:
        x = np.sort(data.values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, style, label=label)
    ax.set_xlabel(column)
    ax.set_ylabel("CDF")
    ax.set_title(f"{tech_label}: {column} CDF overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_time_trace(ue, row_id, tech_label, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(ue["time_s"], ue["SINR_dB"])
    axes[0].set_ylabel("SINR (dB)")
    axes[1].plot(ue["time_s"], ue["Doppler_Hz"], color="tab:orange")
    axes[1].set_ylabel("Doppler (Hz)")
    axes[2].plot(ue["time_s"], ue["Packet_Loss_pct"], color="tab:red")
    axes[2].set_ylabel("Packet loss (%)")
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{tech_label}: 6s trace for row_id={row_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_fading_autocorrelation(fd_hz, fading_type, k_db, step_s, n_steps, out_path, rng):
    from scipy.special import j0
    trace_db = fading_gain_db(n_steps, step_s, fd_hz, fading_type, k_db, rng)
    trace = trace_db - trace_db.mean()
    n = len(trace)
    acf = np.correlate(trace, trace, mode="full")[n - 1:] / (np.var(trace) * n)
    lags = np.arange(len(acf)) * step_s
    theoretical = j0(2 * np.pi * max(abs(fd_hz), 0.1) * lags)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lags[:100], acf[:100], label="empirical ACF")
    ax.plot(lags[:100], theoretical[:100], "--", label="J0 Clarke model")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"Fading ACF check (fd={fd_hz:.1f} Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_rain_curve(rain_rates, rain_fades, tech_label, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rain_rates, rain_fades, s=8, alpha=0.5)
    ax.set_xlabel("Rain rate (mm/hr)")
    ax.set_ylabel("Rain fade (dB)")
    ax.set_title(f"{tech_label}: rain rate vs. rain fade (ITU-R P.618-derived)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
