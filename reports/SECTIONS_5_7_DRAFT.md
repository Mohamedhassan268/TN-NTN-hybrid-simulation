# Draft: Sections 5–7 (Results, Discussion, Conclusion)

**For:** `FRIEND_AI_Native_Framework.docx` (repo root; renamed from the original `AI_Native_Network_Selection_Framework_for_Hybrid_TN_NTN_Systems_using_Deep_Reinforcement_Learning.docx`)
**Drafted:** 2026-08-30

Every number below was recomputed from `drl/figures/phase5_summary.csv`,
`phase5_full_eval_results.csv` (2,000 evaluation episodes), `phase6_handover_sweep.csv`, and
`phase6_reward_profile_sweep.csv`. Nothing is estimated or carried over from the abstract.

> **Two corrections to fold into Sections 3–4 while you are editing:**
>
> 1. **`altitude_m` is a UAV-only column** (33–500 m, 3,883 distinct values), zero-filled for all
>    other technologies. It is *not* satellite altitude — LEO/HAPS geometry enters through
>    `distance_km` and elevation angle. Say so explicitly in the Dataset Overview; the column is
>    arithmetically consistent (UAV mean 266.22 × 21.58% UAV share = 57.46, the reported overall
>    mean), so there is no units bug to fix.
> 2. **Feature count:** 23 total columns = 19 numeric features + 4 non-numeric
>    (`UE_ID`, `network_type`, `Area`, `Available_Networks`). Both figures in the manuscript are
>    right; phrase it as "19 numeric features across 23 columns."
>
> **And one framing fix:** CQL and PPO are trained independently in this work — no checkpoint,
> policy, or Q-function is transferred between them. Present them as parallel offline and online
> approaches, not as a "two-phase pipeline."

---

# 5. Results

## 5.1 Experimental setup

All policies are evaluated in the `NetworkSelectionEnv` described in Section 4, over episodes of
exactly 60 decision steps. Reinforcement learning agents (PPO, DQN) are trained for 100,000
environment steps under each of 3 random seeds and evaluated over 200 episodes per seed (600
episodes per algorithm). Deterministic and rule-based policies (Random, Greedy, Hysteresis, Oracle)
are evaluated over 200 episodes. This yields 2,000 evaluation episodes in total. Confidence
intervals are 95% bootstrap intervals over 2,000 resamples of the episode-level reward.

Three metrics are reported. **Episode reward** is the cumulative composite QoE reward, net of
handover penalties. **Switch rate** is the fraction of steps at which the policy changes network.
**Outage rate** is the fraction of steps at which the policy selects an unavailable network.

## 5.2 Offline policy learning

The offline Conservative Q-Learning agent was trained on the full 55,503-sample dataset under the
contextual-bandit formulation of Section 4. With the library-default conservatism coefficient
(α = 1.0), the learned policy matched the empirically optimal network in only 3 of 6 deployment
areas. Reducing conservatism to α = 0.1 raised this to 5 of 6.

This behaviour is consistent with the structure of the offline problem: the state space comprises
only 6 area contexts, each densely covered by the dataset, so the pessimism that CQL applies to
guard against out-of-distribution actions is largely unnecessary here and instead suppresses
correct greedy selection. The composite reward over the dataset spans [0.0066, 0.9599] with a mean
of 0.6535.

The offline agent operates without any notion of handover cost, and is therefore not directly
comparable to the sequential policies of Section 5.3; it is reported as an indication of how much
of the selection problem is solvable from static context alone.

## 5.3 Sequential policy comparison

Table 5.1 reports all six sequential policies.

**Table 5.1** — Policy comparison. Mean episode reward with 95% bootstrap CI, switch rate, and
outage rate. *n* is the number of evaluation episodes.

| Policy | *n* | Reward | 95% CI | Switch rate | Outage rate |
|---|---|---|---|---|---|
| Random | 200 | 3.171 | [2.078, 4.223] | 0.4134 | 0.3317 |
| Greedy | 200 | 39.881 | [39.597, 40.171] | 0.1696 | 0.0000 |
| Hysteresis (A3 + TTT) | 200 | 40.881 | [40.587, 41.186] | 0.0623 | 0.0000 |
| DQN | 600 | 43.116 | [42.917, 43.330] | 0.1591 | 0.0000 |
| **PPO** | 600 | **44.193** | **[43.974, 44.417]** | **0.0434** | 0.0001 |
| Oracle (upper bound) | 200 | 45.200 | [44.830, 45.563] | 0.0513 | 0.0000 |

PPO attains a mean episode reward of 44.193 against an oracle upper bound of 45.200, closing
97.77% of the achievable return — a regret of 1.007 reward units, or 2.23%. Against the strongest
non-learned baseline, hysteresis, PPO improves mean reward by 3.312 units (+8.10%) while
simultaneously reducing the switch rate from 0.0623 to 0.0434, i.e. **30.3% fewer handovers**. The
95% confidence intervals for PPO and hysteresis are disjoint by a wide margin, as are those for PPO
and DQN.

The Random baseline confirms that the environment is not trivially solvable: selecting uniformly
among the five networks yields a mean reward of 3.171 and an outage rate of 0.3317, since roughly
a third of random selections target a network that is not currently visible.

## 5.4 Handover behaviour

The switch-rate column in Table 5.1 separates the policies into three regimes.

Greedy selection, which re-evaluates the instantaneous best network at every step with no
switching cost, incurs a switch rate of 0.1696 — the ping-pong behaviour that hysteresis margins
were introduced to suppress. Hysteresis with an A3 margin and time-to-trigger reduces this to
0.0623, a 63% reduction, at a modest reward gain.

PPO reaches a switch rate of 0.0434, below both hysteresis and the oracle's 0.0513, without any
explicit hysteresis mechanism, margin parameter, or timer. The only pressure toward switch
avoidance in its objective is the flat handover penalty subtracted from the reward. The agent
therefore recovers anti-ping-pong behaviour as a learned consequence of the cost structure rather
than as an engineered rule — and does so more effectively than the hand-tuned rule.

DQN does not share this property. Despite achieving a mean reward of 43.116, comfortably above both
classical baselines, its switch rate of 0.1591 is close to Greedy's 0.1696. The two RL algorithms
thus reach superior reward by different routes: PPO by learning a stable, switch-averse policy, and
DQN by accepting a high switching cost in exchange for instantaneous link quality. This asymmetry
is discussed in Section 6.2.

## 5.5 Sensitivity to handover penalty

To test whether the handover penalty is the operative mechanism behind PPO's switching behaviour, a
fresh PPO agent was trained at each of four penalty values, holding reward weights at the balanced
profile. Each point is an independent 50,000-step training run.

**Table 5.2** — Effect of handover penalty on learned behaviour.

| Handover penalty | Reward | Switch rate |
|---|---|---|
| 0.05 | 44.309 | 0.0820 |
| 0.15 (default) | 43.832 | 0.0565 |
| 0.30 | 43.382 | 0.0467 |
| 0.50 | 43.153 | 0.0298 |

Switch rate decreases monotonically as the penalty increases, falling by 64% from 0.0820 to 0.0298
across the tested range. Mean reward decreases modestly and monotonically in the opposite
direction, from 44.309 to 43.153. The penalty therefore behaves as a directly interpretable control
knob: an operator can trade approximately 1.16 reward units for a 64% reduction in signalling load,
and can position that trade-off anywhere along the curve by setting a single scalar.

We note that the sweep does not include a zero-penalty condition; the lowest value tested is 0.05.
The monotone trend is consistent with the penalty being the causal mechanism, but a true ablation at
penalty = 0 has not been run, and we do not claim one.

## 5.6 Sensitivity to reward weighting

The same protocol was applied to four reward-weight profiles at the default penalty of 0.15.

**Table 5.3** — Effect of reward-weight profile.

| Reward profile | Reward | Switch rate |
|---|---|---|
| Balanced | 43.832 | 0.0565 |
| Throughput-leaning | 43.476 | 0.0528 |
| **Latency-leaning** | **44.855** | 0.0507 |
| Reliability-leaning | 43.182 | 0.0465 |

The latency-leaning profile yields the highest reward, exceeding the balanced profile by 1.023
units. This is consistent with the structure of the hybrid scenario: propagation delay is the KPI
that most sharply separates the terrestrial from the non-terrestrial tiers, so a reward that
weights latency more heavily provides a correspondingly sharper learning signal for the network
selection decision the agent must actually make.

As an internal consistency check, the balanced configuration appears in both Table 5.2 and Table
5.3 — the same trained configuration evaluated independently — and yields identical values
(43.832, 0.0565) in each.

## 5.7 Training stability

Because RL results are sensitive to seed variation, per-seed means are reported in Table 5.4.

**Table 5.4** — Across-seed variability (3 training seeds, 200 episodes each).

| Algorithm | Seed 0 | Seed 1 | Seed 2 | Across-seed SD | Within-seed episode SD |
|---|---|---|---|---|---|
| PPO (reward) | 43.905 | 45.185 | 43.489 | 0.884 | 2.773 |
| PPO (switch rate) | 0.050 | 0.025 | 0.056 | 0.0166 | — |
| DQN (reward) | 43.662 | 42.694 | 42.991 | 0.496 | 2.576 |
| DQN (switch rate) | 0.147 | 0.157 | 0.174 | 0.0135 | — |

Across-seed reward variability is smaller than within-seed episode variability for both algorithms
(PPO: 0.884 vs 2.773; DQN: 0.496 vs 2.576), indicating that the reported ranking is driven by
policy quality rather than by training-run luck. However, the PPO-over-DQN reward ordering warrants
a caveat: the worst PPO seed (43.489) falls slightly below the best DQN seed (43.662), so with only
3 seeds that particular ordering is less firmly established than the pooled confidence intervals
alone would suggest. The switch-rate separation between the two algorithms is by contrast
unambiguous — every PPO seed lies below every DQN seed by a factor of roughly three.

---

# 6. Discussion

## 6.1 Learned versus engineered switching discipline

The central empirical finding is that a policy optimising a composite QoE reward with a flat
handover cost outperforms an explicitly engineered hysteresis rule on both axes that rule was
designed to balance. PPO achieves higher reward *and* fewer handovers than the A3-margin baseline
(Section 5.3), and approaches the oracle to within 2.23%.

This matters because hysteresis parameters — margin and time-to-trigger — must be tuned per
deployment, and their optimal values depend on cell geometry, mobility, and traffic. In a hybrid
TN–NTN setting those conditions vary not merely between deployments but *within a single episode*,
as a LEO platform rises and sets. A fixed margin cannot adapt to that; a learned policy conditioned
on live radio measurements can. The sensitivity analysis in Section 5.5 shows further that the
resulting behaviour remains controllable through one interpretable scalar, which is a practical
requirement for any mechanism an operator would consider deploying.

## 6.2 Why the two RL algorithms diverge

PPO and DQN achieve comparable reward through materially different policies (Section 5.4). A
plausible explanation lies in the algorithms' treatment of action stability. PPO's clipped
surrogate objective constrains each update to a trust region around the current policy, which
favours convergence to a temporally consistent action distribution. DQN selects greedily with
respect to a learned Q-function and carries no comparable smoothness constraint, so small
fluctuations in estimated action values translate directly into switching.

We offer this as an interpretation rather than a demonstrated mechanism; isolating it would require
an ablation over policy-update constraints that we have not conducted.

## 6.3 Threats to validity

**Fixed evaluation geometry.** The LEO and HAPS pass geometry is generated from a single analytic
elevation profile, and the same profile is used for training and evaluation. It is therefore
possible that part of PPO's near-oracle performance reflects adaptation to a repeatable geometry
pattern rather than a generally transferable selection policy. Distinguishing these requires
evaluation on held-out geometry — varied orbital parameters, pass durations, or elevation
profiles — which we have not performed. This is the most significant open question about the
result, and we do not claim generalization beyond the tested configuration.

**Single-source synthetic data.** All five technology datasets originate from one simulation
pipeline. While the channel model is grounded in ITU-R and 3GPP recommendations, it has not been
validated against measured field data, and systematic modelling error would be shared across all
five tiers rather than averaging out.

**Modelling simplifications.** Doppler shift is held at each technology's median value in the
mobility model rather than derived from instantaneous relative velocity. The area context evolves
according to a hand-specified Markov chain rather than a measured mobility trace.

**Value-function fit.** PPO's explained variance remained low throughout training (approximately
0.01–0.15), indicating that the learned value function accounts for little of the observed return
variance. The policy performs well regardless, but this suggests the advantage estimates are noisy
and that a better-fitted critic might yield further improvement.

**Scale.** Training runs of 100,000 steps across 3 seeds are appropriate for establishing the
comparison presented here, but are modest relative to the broader claim of a pathway toward
AI-native radio protocols. The results support the narrower conclusion that learned selection
outperforms classical handover heuristics in this environment.

## 6.4 Deployment considerations

An operator considering a learned selection controller for live infrastructure requires more than a
favourable simulation benchmark. Two requirements are worth noting. First, **explainability**: the
policy is a neural network whose decisions cannot be audited in the way an A3 margin can, and
regulatory or operational review may require a rule-based approximation or post-hoc justification
of its behaviour. Second, **fallback guarantees**: a deployed controller needs a well-defined
degraded mode when observations are missing or the policy produces an invalid selection. The
outage rate of 0.0001 observed for PPO, while small, is not zero, and a production system would
require a deterministic safety net rather than relying on the learned policy alone.

---

# 7. Conclusion

This paper presented a physically grounded dataset and reinforcement learning environment for
network selection across five heterogeneous access technologies — 5G NR, Wi-Fi 6, LEO satellite,
HAPS, and UAV relay — and evaluated learned selection policies against classical handover
algorithms.

The principal findings are as follows. A PPO agent conditioned on live per-candidate radio
measurements attains 97.77% of an oracle upper bound, improving mean episode reward by 8.10% over a
3GPP A3-style hysteresis baseline while simultaneously performing 30.3% fewer handovers. This
switch-avoiding behaviour is not engineered but emerges from a flat handover cost in the reward,
and is controllable through that single scalar: raising the penalty from 0.05 to 0.50 reduces the
switch rate by 64% at a cost of 1.16 reward units. A latency-weighted reward profile outperforms
balanced, throughput-leaning, and reliability-leaning alternatives, consistent with propagation
delay being the KPI that most sharply distinguishes the terrestrial from the non-terrestrial tiers.
Notably, the two RL algorithms tested do not behave alike: DQN matches PPO's reward advantage over
the classical baselines but not its switching discipline.

The most important limitation is that training and evaluation share a single analytic pass
geometry, so the extent to which these policies generalize across orbital and mobility conditions
remains untested. Establishing that, together with validation of the channel model against measured
data, is the natural next step. The dataset and environment are released to support such work.

---

## Still required before submission

- **Dataset availability statement** — Contribution #1 promises a public release; add a repository
  URL, DOI, and license.
- **Prior-work comparison table** — see marker #06 in `CITATIONS_NEEDED.md`.
- **Bibliography** — 24 citation sites, 29 references, none currently resolving.
- **Journal template** — the document is built on the Istanbul University *Physics and Astronomy
  Reports* class. Replace before submitting to a telecom venue.
- **Figures** — `drl/figures/phase5_reward_vs_switch_tradeoff.png` is the strongest single figure
  for Section 5.3; `phase6_handover_sweep.png` supports Section 5.5.
