"""Builds a fully synthetic Ka-band (20 GHz) hybrid dataset from sim/python_sim_all.py.

Why this exists: the real data/raw/*.csv files that the original merged dataset
(data/Hybrid_Network_TN_NTN_Final.csv) was built from are NOT reproducible from this repo's
sim/python_sim_all.py -- same seed, wildly different values (confirmed: sim_leo(seed=42)'s
output matches data/regenerated/, not data/raw/). The real generator's source is lost. So a
Ka-band LEO link cannot be spliced into the real dataset without fabricating a relationship
between unrelated synthetic and real values.

Instead, this builds an entirely new, internally self-consistent synthetic dataset via
sim/python_sim_all.py -- the four non-LEO technologies at their existing (Ku-baseline-equivalent)
parameters, LEO regenerated at physically-consistent Ka-band (20 GHz, FSPL-adjusted SNR). This is
a deliberately different row count (55,600, not 55,503) from the real dataset -- it is a new
dataset, not a patch, and is documented as such.

The Area / Available_Networks / log-feature assignment logic below is ported from
notebooks/TN_NTN_Deep_RL_MODEL.ipynb cells 65-73 (probability tables copied verbatim), with one
change: Area assignment is explicitly seeded here (the original notebook process was unseeded and
its exact draw is unrecoverable), so this dataset is reproducible end-to-end.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from python_sim_all import sim_uav, sim_haps, sim_leo, sim_5g, sim_wifi6

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "Hybrid_Network_TN_NTN_Ka.csv"

AREA_ASSIGN_SEED = 42  # explicit, reproducible -- see module docstring

KA_FREQ_GHZ = 20.0
KA_SNR_OFFSET_DB = -20 * math.log10(KA_FREQ_GHZ / 12.5)  # FSPL delta vs the Ku baseline, ~-4.08 dB

NETWORK_AREA = {
    "NR_5G":      {"areas": ["Indoor", "Urban", "Rural", "Highway"],                          "p": [0.41, 0.30, 0.18, 0.11]},
    "WiFi":       {"areas": ["Indoor", "Urban", "Rural"],                                      "p": [0.47, 0.33, 0.20]},
    "SAT (LEO)":  {"areas": ["Indoor", "Urban", "Rural", "Highway", "Maritime", "Desert"],      "p": [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]},
    "HAPS":       {"areas": ["Urban", "Rural", "Highway", "Maritime", "Desert"],                "p": [0.39, 0.23, 0.15, 0.12, 0.11]},
    "UAV":        {"areas": ["Urban", "Rural", "Highway", "Maritime", "Desert"],                "p": [0.38, 0.22, 0.16, 0.13, 0.11]},
}

AREA_NETWORKS = {
    "Indoor":  "NR_5G,WiFi,SAT (LEO)",
    "Urban":   "NR_5G,WiFi,UAV,HAPS,SAT (LEO)",
    "Rural":   "NR_5G,WiFi,HAPS,SAT (LEO),UAV",
    "Highway": "NR_5G,UAV,HAPS,SAT (LEO)",
    "Maritime": "UAV,HAPS,SAT (LEO)",
    "Desert":  "UAV,HAPS,SAT (LEO)",
}


def build() -> pd.DataFrame:
    print(f"Ka-band LEO params: freq_ghz={KA_FREQ_GHZ}, snr_offset_db={KA_SNR_OFFSET_DB:.4f}")

    uav = sim_uav(); uav["network_type"] = "UAV"
    haps = sim_haps(); haps["network_type"] = "HAPS"
    leo = sim_leo(freq_ghz=KA_FREQ_GHZ, snr_offset_db=KA_SNR_OFFSET_DB); leo["network_type"] = "SAT (LEO)"
    fiveg = sim_5g(); fiveg["network_type"] = "NR_5G"
    wifi = sim_wifi6(); wifi["network_type"] = "WiFi"

    for name, df in [("UAV", uav), ("HAPS", haps), ("LEO(Ka)", leo), ("5G", fiveg), ("WiFi6", wifi)]:
        print(f"  {name:8s} rows={len(df)}")

    df = pd.concat([uav, haps, leo, fiveg, wifi], ignore_index=True, sort=False)
    df["altitude_m"] = df.get("altitude_m", 0.0)
    df["speed_ms"] = df.get("speed_ms", 0.0)
    df[["altitude_m", "speed_ms"]] = df[["altitude_m", "speed_ms"]].fillna(0.0)
    if "sat_type" in df.columns:
        df = df.drop(columns=["sat_type"])

    df["Log_Throughput_Mbps"] = np.log1p(df["Throughput_Mbps"])
    df["Log_BER"] = -np.log10(df["BER"] + 1e-12)
    df["Log_Packet_Loss_pct"] = np.log1p(df["Packet_Loss_pct"])

    rng = np.random.default_rng(AREA_ASSIGN_SEED)

    def assign_area(network: str) -> str:
        info = NETWORK_AREA[network]
        return rng.choice(info["areas"], p=info["p"])

    df["Area"] = df["network_type"].apply(assign_area)
    df["Available_Networks"] = df["Area"].map(AREA_NETWORKS)

    invalid = df[~df.apply(lambda row: row["network_type"] in row["Available_Networks"], axis=1)]
    assert len(invalid) == 0, f"{len(invalid)} rows have network_type not in their own Available_Networks"

    return df


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT_PATH, index=False)
    print(f"\nTotal rows: {len(df)}  (real dataset has 55,503 -- this is a new dataset, not a patch)")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nArea distribution:")
    print(df["Area"].value_counts().to_string())
    print(f"\nSaved: {OUT_PATH}")
