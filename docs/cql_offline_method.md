# Revision v2 CQL method

CQL is an independent offline experiment. It neither initializes PPO/DQN nor consumes online checkpoints. For each CQL seed, only the canonical training trajectories are shuffled and partitioned by complete `trajectory_id` into a 90% offline-training group and a 10% internal-development group. The final test trajectories are never used to build the logged dataset.

The logged behavior policy selects uniformly from currently valid actions. CQL receives the documented 21-value current context, not the five previous-network bits used by online PPO/DQN. Its logged reward is current-step balanced QoE. Sequential handover accounting remains in the shared online evaluator, where CQL is evaluated as a policy without altering its offline training formulation.

For d3rlpy 2.8.1 `DiscreteCQL`, the implemented conservative term is the batch mean

`logsumexp_a Q(s,a) - Q(s,a_data)`.

The total critic loss is the Double-DQN temporal-difference loss plus `alpha` times that conservative term. It does not use the continuous-action importance-sampling term, a learnable temperature, or an undocumented reward normalization. The fixed configuration is in `experiments/revision_v2.json`; the per-epoch d3rlpy metrics, including conservative loss when emitted by the installed version, are retained with the checkpoint.
