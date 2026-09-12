# Proposed pull request

## Title

Rebuild TN-NTN network selection around matched scenarios and seed-aware evaluation

## Summary

- Adds the versioned canonical v2 scenario bank with 1,200 trajectory-level train/validation/test splits, 10-second timing, five candidate rows per epoch, matched Ku/Ka/S conditions, and training-only common normalization.
- Replaces universal free-space use with selected UMa, UAV aerial-link, NTN basic, and explicitly separate Wi-Fi path-loss components; adds machine-readable conformance vectors and a release-blocking external-simulator harness.
- Adds independent offline 21-feature CQL and online 26-feature PPO/DQN campaign runners, validation-only hysteresis tuning, the exact paired finite-horizon oracle, hierarchical seed-first statistics, OOD geometry suites, and sensitivity/Pareto runners.
- Adds release gates, manuscript/figure builders that refuse legacy or unfrozen results, and a separate Package C classical/federated/hybrid-QNN scaffold.

## Verification completed

- `23 passed` from `python -m pytest tests -q`.
- Deterministic selected-model vectors pass at 0.1 dB or tighter.
- Canonical data audit passes: 1,080,000 rows, 1,200 trajectories, 800/200/200 trajectory splits.
- Three frozen OOD geometry banks pass validation: low shell, high shell, and low elevation, 200 trajectories each.
- `git diff --check` passes; only line-ending warnings appear for pre-existing modified tracked files.

## Release blockers deliberately retained

- Frozen OpenNTN and LLSim5G output must populate `standards/external_reference_outputs.csv` from the supplied template.
- PPO critic development must complete and freeze the validation-selected capacity before core PPO runs.
- The 10-seed core PPO/DQN/CQL campaign, 5-seed sensitivity campaign, final paired evaluation, and clean-clone verification must complete.
- The Word manuscript is intentionally result-gated; it is not generated until those frozen inputs exist.

## Scope note

This repository was already dirty before this revision and currently contains mixed legacy/user changes. Create the real pull request only from a clean branch containing the v2 paths listed above; do not include the old datasets, legacy checkpoints, existing figures, or unrelated manuscript files.
