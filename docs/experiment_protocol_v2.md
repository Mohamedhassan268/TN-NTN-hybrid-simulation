# Revision v2 experiment protocol

The revision-v2 campaign is intentionally separated from all legacy artifacts. Nothing under the legacy `drl/models*`, old CSVs, figures, or DOCX files is reused as a revision-v2 result.

## Immutable inputs

`data/canonical_v2/manifest.json` fixes 1,200 trajectory identifiers, their 800/200/200 split, the base seed, three bands, 60 steps per episode, and a 10-second decision interval. `reference_params.json` is fitted from available training rows across the three band configurations and is the only normalization/utility reference used for cross-policy comparison.

Every PPO, DQN, and CQL checkpoint carries the SHA-256 hash of `experiments/revision_v2.json`, package versions, seed, algorithm, band, and phase. A mismatch is a hard error, not a warm-start opportunity.

## Campaign order

1. Run the deterministic schema, environment, oracle, statistics, and internal conformance tests.
2. Produce and archive frozen outputs from OpenNTN and LLSim5G; run `scripts/run_physical_conformance.py --external ...`. Do not make an NTN or 3GPP realism claim without this gate.
3. Run `scripts/develop_ppo_critic.py`, review `selected_configuration.json`, and copy the selected capacity into the core manifest before core PPO training.
4. Run `scripts/train_revision_v2.py` for PPO and DQN, and `scripts/train_cql_revision_v2.py` separately for CQL. The three training branches never load each other's checkpoints.
5. Tune the 24 hysteresis configurations only on validation trajectories, then run `scripts/evaluate_revision_v2.py` once against the 200 final test trajectories.
6. Run `scripts/statistics_revision_v2.py`, `scripts/run_sensitivity_revision_v2.py`, `scripts/make_pareto_revision_v2.py`, and `scripts/evaluate_geometry_revision_v2.py`.
7. Rebuild tables/figures and create the Word manuscript only from the frozen results paths. Finally run the clean-clone/release gate.

## Evaluation definitions

The exact finite-horizon oracle receives the realized full-episode outcome matrix only. It charges the same 0.15 reference handover penalty used in reporting and must weakly dominate every policy on every shared trajectory. The oracle is not a deployable baseline.

Invalid actions receive -1.0, advance one decision epoch, and retain the previous valid attachment. The fixed comparison utility is balanced throughput, latency, and reliability QoE plus a 0.15 handover penalty. Sensitivity conditions are compared through raw KPIs and this fixed reference utility, never their different training rewards.

The outer unit for inference is the training seed. The bootstrap resamples ten training seeds and then shared scenarios within seed. Seed-wise paired differences, 95% intervals, and the exact ten-seed sign-flip p-value are reported if p-values are retained.

## Release blocking conditions

- An incomplete dataset manifest, split overlap, schema failure, or missing five-candidate group.
- A missing reference-vector result or failed physical conformance case.
- Missing external OpenNTN/LLSim5G frozen comparison output.
- Fewer than ten valid core seeds per algorithm/band or five valid seeds per sensitivity condition/band.
- An oracle value below an evaluated policy on any shared scenario.
- A final manuscript produced from unfrozen or incomplete campaign results.
