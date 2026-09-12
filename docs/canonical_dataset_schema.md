# Canonical revision-v2 dataset

Version 2.0.0 contains 1,200 matched trajectories: 800 training, 200 validation, and 200 final test. Each trajectory contains 60 decision epochs separated by 10 seconds. Each `(trajectory_id, step_index, band)` group contains exactly five rows in the stable order HAPS, NR_5G, SAT (LEO), UAV, and WiFi. Splitting is permitted only by `trajectory_id`.

The primary key is `(trajectory_id, step_index, band, network_type)`. Identity columns include dataset/scenario/trajectory/UE identifiers, split, step, physical time, interval, band, and geometry regime. Context includes area, network, bit-packed `available_mask`, `area_allowed`, geometric `visible`, and canonical `available`. Geometry includes slant or link distance in km, elevation in degrees, altitude in meters, speed in m/s, Doppler in Hz, and rain rate in mm/h.

Available rows contain RSSI (dBm), SNR/SINR (dB), BER, BLER, throughput (Mbit/s), propagation delay/latency (ms), packet loss (%), spectral efficiency (bit/s/Hz), LQI, selected MCS order and code rate, and a saturation flag. Outcomes on unavailable rows are null; downstream code must use the availability bit and may not impute those outcomes.

The CQL observation is 21 values: six area one-hot values followed by five `[available, RSSI_norm, SINR_norm]` triples. PPO and DQN append five previous-network indicators for 26 values. Normalization bounds are fitted once from available rows in the canonical training split across all three bands. Unavailable observations are `[0,0,0]` for their candidate slot.

Ku, Ka, and S differ only in the frozen LEO band configuration and frequency-dependent LEO random stream. Shared terrestrial/HAPS/UAV geometry and realized conditions are byte-identical across the three band views. Because fixed-aperture gain and carrier frequency vary, results are described as a **band-configuration study**, not a pure frequency experiment.
