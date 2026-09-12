"""Maps a clean-data row (plus which source file it came from) to a technology config
and a scenario dict of geometry/environment parameters that drive the physics."""
from .config import get_config

PLATFORM_ALTITUDE_KM = {
    "5G_NR": 0.03,   # macro BS height
    "WiFi6_2.4GHz": 0.003,
    "WiFi6_5GHz": 0.003,
    "WiFi6E_6GHz": 0.003,
    "HAPS": 20.0,     # stratospheric platform
    "LEO": 550.0,     # representative LEO shell altitude
    "LEO_Ka": 550.0,  # same orbital shell -- altitude doesn't depend on carrier frequency
    "LEO_S": 550.0,   # same orbital shell as LEO/LEO_Ka
    "UAV": 0.0,       # handled via altitude_m directly
}

# Which CSV file maps to which tech-name resolution strategy
FILE_TECH_RESOLVERS = {
    "TN_Data_5G.csv": lambda row: "5G_NR",
    "TN_Data_WIFI_6.csv": lambda row: row["network_type"],
    "NTNData_HAPS.csv": lambda row: "HAPS",
    "NTN_Data_LEO.csv": lambda row: "LEO",
    "Data_UAV.csv": lambda row: "UAV",
}


def resolve_tech_name(file_name: str, row) -> str:
    return FILE_TECH_RESOLVERS[file_name](row)


def build_scenario(row, tech_name: str) -> dict:
    scenario = {
        "distance_km": float(row["distance_km"]),
        "doppler_char_hz": float(row["Doppler_Hz"]),
        "rain_rate_mmhr": float(row.get("Rain_Rate_mmhr", 0.0)),
        "platform_altitude_km": PLATFORM_ALTITUDE_KM.get(tech_name, 0.0),
    }
    if "altitude_m" in row.index:
        scenario["altitude_m"] = float(row["altitude_m"])
        scenario["platform_altitude_km"] = float(row["altitude_m"]) / 1000.0
    if "speed_ms" in row.index:
        scenario["speed_ms"] = float(row["speed_ms"])
    return scenario


def get_tech_config(tech_name: str):
    return get_config(tech_name)
