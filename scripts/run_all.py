"""Driver script: `python scripts/run_all.py sample` builds small per-tech samples,
runs invariant checks, and writes validation plots/report. `python scripts/run_all.py full`
runs the complete generation (all UEs, 600 steps x 10ms) to output/*.parquet."""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_models.generate import generate_noisy_dataframe, write_outputs
from noise_models.validate import (
    check_invariants, summarize_vs_clean, plot_cdf_overlay,
    plot_time_trace, plot_fading_autocorrelation, plot_rain_curve,
)
from noise_models.technologies import resolve_tech_name, get_tech_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["TN_Data_5G.csv", "TN_Data_WIFI_6.csv", "NTNData_HAPS.csv", "NTN_Data_LEO.csv", "Data_UAV.csv"]
METRIC_COLS = ["SNR_dB", "SINR_dB", "BER", "Throughput_Mbps", "Latency_ms",
               "Packet_Loss_pct", "Link_Quality_Index", "Spectral_Efficiency_bps_hz"]


def run_sample(sample_ues=40):
    val_dir = os.path.join(ROOT, "validation")
    os.makedirs(val_dir, exist_ok=True)
    report_lines = ["# Validation Report\n", f"Sample size: {sample_ues} UEs per technology, 600 steps @ 10ms.\n"]

    for f in FILES:
        clean_path = os.path.join(ROOT, f)
        clean_df = pd.read_csv(clean_path)
        noisy_df = generate_noisy_dataframe(clean_df, f, seed=7, max_ues=sample_ues, n_steps=600, step_s=0.01)
        tech_label = f.replace(".csv", "")

        inv = check_invariants(noisy_df)
        summary = summarize_vs_clean(noisy_df, clean_df.iloc[:sample_ues], METRIC_COLS)

        report_lines.append(f"\n## {tech_label}\n")
        report_lines.append("### Physical invariant checks\n")
        for k, v in inv.items():
            status = "PASS" if v == 0 else "CHECK"
            report_lines.append(f"- `{k}`: {v} ({status})\n")
        report_lines.append("\n### Clean vs. regenerated summary\n")
        report_lines.append(summary.round(4).to_markdown(index=False) + "\n")

        for col in ["SNR_dB", "Packet_Loss_pct"]:
            png = os.path.join(val_dir, f"{tech_label}_{col}_cdf.png")
            plot_cdf_overlay(noisy_df, clean_df.iloc[:sample_ues], col, tech_label, png)
            report_lines.append(f"![{col} CDF]({os.path.basename(png)})\n")

        first_row_id = noisy_df["row_id"].iloc[0]
        png_trace = os.path.join(val_dir, f"{tech_label}_trace.png")
        plot_time_trace(noisy_df[noisy_df["row_id"] == first_row_id], first_row_id, tech_label, png_trace)
        report_lines.append(f"![time trace]({os.path.basename(png_trace)})\n")

        row0 = clean_df.iloc[0]
        tech_name = resolve_tech_name(f, row0)
        cfg = get_tech_config(tech_name)
        fd0 = float(row0["Doppler_Hz"])
        rng = np.random.default_rng(99)
        png_acf = os.path.join(val_dir, f"{tech_label}_fading_acf.png")
        plot_fading_autocorrelation(fd0, cfg.fading_type, cfg.rician_k_db, cfg.step_s, cfg.n_steps, png_acf, rng)
        report_lines.append(f"![fading ACF]({os.path.basename(png_acf)})\n")

        if cfg.rain_capable and clean_df["Rain_Rate_mmhr"].max() > 0:
            png_rain = os.path.join(val_dir, f"{tech_label}_rain_curve.png")
            plot_rain_curve(noisy_df["Rain_Rate_mmhr"], noisy_df["Rain_Fade_dB"], tech_label, png_rain)
            report_lines.append(f"![rain curve]({os.path.basename(png_rain)})\n")

        print(f"[{tech_label}] invariants: {inv}")

    with open(os.path.join(val_dir, "report.md"), "w") as fh:
        fh.writelines(report_lines)
    print(f"\nValidation report written to {os.path.join(val_dir, 'report.md')}")


def run_full():
    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    for f in FILES:
        clean_path = os.path.join(ROOT, f)
        clean_df = pd.read_csv(clean_path)
        print(f"Generating full trace for {f} ({len(clean_df)} UEs x 600 steps)...")
        noisy_df = generate_noisy_dataframe(clean_df, f, seed=42)
        stem = os.path.join(out_dir, f.replace(".csv", "_noisy"))
        write_outputs(noisy_df, stem)
        print(f"  -> {stem}.parquet ({len(noisy_df)} rows), {stem}_sample.csv")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        run_sample()
    elif mode == "full":
        run_full()
    else:
        print("usage: python scripts/run_all.py [sample|full]")
