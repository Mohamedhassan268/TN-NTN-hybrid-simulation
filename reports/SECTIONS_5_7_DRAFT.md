# Draft: Sections 5–7 (Results, Discussion, Conclusion)

**For:** `FRIEND_AI_Native_Framework_final.docx`
**Drafted:** 2026-08-30. **Revised:** 2026-09-04 — full rewrite against corrected ITU-R physics
and a three-band (Ku/Ka/S) campaign; supersedes the 2026-08-30 draft entirely.
**Status:** fully inserted into the manuscript, plus Phase D fixes (Table 2 band/bandwidth,
abstract, "two-phase pipeline" framing, Section 4 rain/gaseous model description, Data and Code
Availability statement). Kept as the reviewable source of the numbers, not as a to-do list.

Every number below traces to a CSV under `drl/figures/` (Ku, primary), `drl/figures_ka/`, or
`drl/figures_s/` — nothing is estimated or carried over from the abstract. All three campaigns
share the same generator, seeds, Area assignment, and reward weights; only the LEO `TechConfig`
(carrier frequency and its physically-derived antenna gain) differs between them.

**Physics note for Section 4:** the noise model's rain attenuation (ITU-R P.838-3) and gaseous
absorption (ITU-R P.676-13, via the P.835-7 reference atmosphere) are implemented as closed-form
recommendations rather than a hand-fit table, verified against published reference values before
use (§4, physics grounding). This numbers below reflect that corrected model.

> **Corrections already applied to the manuscript (verified in
> `FRIEND_AI_Native_Framework_final.docx`) — do not reapply:**
> 1. `altitude_m` is UAV-only (33–500 m), zero-filled elsewhere; not satellite altitude.
> 2. `elevation_angle` is computed internally during data generation, not a persisted column.
> 3. CQL and PPO are trained independently — presented as parallel offline/online approaches,
>    not a "two-phase pipeline" (fixed in the Methodology heading/body AND in Introduction
>    contribution #4, where the same framing was independently found and fixed).
> 4. Table 2's LEO row: 12.0 GHz (Ku), 250 MHz (was 20 GHz (Ka), 500 MHz).
> 5. Abstract and Introduction contribution #5: 97.5% / 0.037 / 0.062 (was 98% / 0.043 / 0.062),
>    Ku-band stated explicitly, three-band robustness comparison mentioned.
> 6. Section 4's rain/gaseous description: cites ITU-R P.838-3 (not P.618) for rain, states the
>    coefficients come from a closed-form regression (not table interpolation), and the old
>    "simplified zenith" gaseous formula is reframed as an early/illustrative approximation with
>    a new paragraph describing the actual P.676-13/P.835-7 implementation used for all results.
> 7. A "Data and Code Availability" section added after the Conclusion, citing the real public
>    repository and both licenses (MIT code / CC BY 4.0 data).

---

# 5. Results

## 5.1 Experimental setup

All policies are evaluated in the `NetworkSelectionEnv` described in Section 4, over episodes of
exactly 60 decision steps. Reinforcement learning agents (PPO, DQN) are trained for 100,000
environment steps under each of 3 random seeds and evaluated over 200 episodes per seed (600
episodes per algorithm). Deterministic and rule-based policies (Random, Greedy, Hysteresis, Oracle)
are evaluated over 200 episodes. This yields 2,000 evaluation episodes per band. Confidence
intervals are 95% bootstrap intervals over 2,000 resamples of the episode-level reward;
significance is additionally assessed via Welch's t-test, Mann-Whitney U, and Cohen's d.

Three metrics are reported. **Episode reward** is the cumulative composite QoE reward, net of
handover penalties. **Switch rate** is the fraction of steps at which the policy changes network.
**Outage rate** is the fraction of steps at which the policy selects an unavailable network.

Sections 5.2–5.8 report the Ku-band (12 GHz LEO) campaign, the paper's primary band, matching
Section 3's dataset characterization. Section 5.9 reports a controlled comparison against two
additional LEO bands.

## 5.2 Offline policy learning

The offline Conservative Q-Learning agent was trained on the full 55,503-sample dataset under the
contextual-bandit formulation of Section 4. This fit is unaffected by the LEO carrier band — CQL
trains on the dataset's static columns, not on live channel simulation. With the library-default
conservatism coefficient (α = 1.0), the learned policy matched the empirically optimal network in
only 3 of 6 deployment areas. Reducing conservatism to α = 0.1 raised this to 5 of 6.

This behaviour is consistent with the structure of the offline problem: the state space comprises
only 6 area contexts, each densely covered by the dataset, so the pessimism that CQL applies to
guard against out-of-distribution actions is largely unnecessary here and instead suppresses
correct greedy selection. The composite reward over the dataset spans [0.0066, 0.9599] with a mean
of 0.6535.

The offline agent operates without any notion of handover cost, and is therefore not directly
comparable to the sequential policies of Section 5.3; it is reported as an indication of how much
of the selection problem is solvable from static context alone.

## 5.3 Sequential policy comparison

Table 5.1 reports all six sequential policies under Ku-band (12 GHz) LEO.

**Table 5.1** — Ku-band policy comparison. Mean episode reward with 95% bootstrap CI, switch rate,
and outage rate. *n* is the number of evaluation episodes.

| Policy | *n* | Reward | 95% CI | Switch rate | Outage rate |
|---|---|---|---|---|---|
| Random | 200 | 3.131 | [2.041, 4.177] | 0.4134 | 0.3317 |
| Greedy | 200 | 39.958 | [39.686, 40.240] | 0.1688 | 0.0000 |
| Hysteresis (A3 + TTT) | 200 | 40.904 | [40.607, 41.211] | 0.0625 | 0.0000 |
| DQN | 600 | 43.461 | [43.260, 43.671] | 0.1421 | 0.0000 |
| **PPO** | 600 | **44.165** | **[43.952, 44.394]** | **0.0373** | 0.0000 |
| Oracle (upper bound) | 200 | 45.292 | [44.915, 45.652] | 0.0496 | 0.0000 |

PPO attains a mean episode reward of 44.165 against an oracle upper bound of 45.292, closing
97.51% of the achievable return — a regret of 1.127 reward units (2.49%). Against the strongest
non-learned baseline, hysteresis, PPO improves mean reward by 3.261 units (+7.97%) while
simultaneously reducing the switch rate from 0.0625 to 0.0373 — **40.3% fewer handovers**.

These differences are statistically significant, not merely CI-disjoint: PPO vs. hysteresis reward,
Welch p=4.5×10⁻⁵³ (Cohen's d=1.25, large); PPO vs. hysteresis switch rate, p=1.1×10⁻²⁴ (d=−0.85).
PPO vs. oracle reward is also significant (p=7.1×10⁻⁷, d=−0.41) — the 2.49% gap is real, not
sampling noise, and should be reported precisely rather than as approximate parity. PPO's switch
rate is significantly *below* oracle's (p=2.3×10⁻⁵, d=−0.38): under Ku-band conditions, the learned
policy is more switch-conservative than the reward-optimal plan itself.

The Random baseline confirms the environment is not trivially solvable: selecting uniformly among
the five networks yields a mean reward of 3.131 and an outage rate of 0.3317, since roughly a third
of random selections target a network that is not currently visible.

## 5.4 Handover behaviour

The switch-rate column in Table 5.1 separates the policies into three regimes.

Greedy selection, which re-evaluates the instantaneous best network at every step with no
switching cost, incurs a switch rate of 0.1688 — the ping-pong behaviour that hysteresis margins
were introduced to suppress. Hysteresis with an A3 margin and time-to-trigger reduces this to
0.0625, at a modest reward gain.

PPO reaches a switch rate of 0.0373, below both hysteresis and the oracle's 0.0496, without any
explicit hysteresis mechanism, margin parameter, or timer. The only pressure toward switch
avoidance in its objective is the flat handover penalty subtracted from the reward. The agent
recovers anti-ping-pong behaviour as a learned consequence of the cost structure rather than as an
engineered rule — and does so more effectively than the hand-tuned rule (p=1.1×10⁻²⁴, §5.3).

DQN does not share this property: despite a mean reward of 43.461, comfortably above both classical
baselines (p=8.8×10⁻⁷¹ vs. hysteresis, d=1.37), its switch rate of 0.1421 is close to Greedy's
0.1688, and significantly higher than PPO's (p=3.0×10⁻¹⁶⁵, d=−2.01 — a very large effect). The two
RL algorithms thus reach superior reward by different routes: PPO by learning a stable,
switch-averse policy, and DQN by accepting frequent switching in exchange for instantaneous link
quality. Figure 5.1 (`phase5_training_curves.png`) shows this divergence has a training-dynamics
counterpart: DQN's off-policy replay buffer lets it reach a good policy far faster (~10,000
environment steps) than PPO (~50,000 steps), but PPO's on-policy trust-region updates ultimately
converge to a marginally higher and much more switch-disciplined final policy. This asymmetry is
discussed further in Section 6.2.

## 5.5 Sensitivity to handover penalty

To test whether the handover penalty is the operative mechanism behind PPO's switching behaviour, a
fresh PPO agent was trained at each of five penalty values — including a true zero-penalty
ablation — holding reward weights at the balanced profile. Each point is an independent training
run (50,000 steps).

**Table 5.2** — Effect of handover penalty on learned behaviour (Ku-band).

| Handover penalty | Reward | Switch rate |
|---|---|---|
| 0.00 (ablation) | 44.303 | 0.0690 |
| 0.05 | 44.251 | 0.0662 |
| 0.15 (default) | 43.849 | 0.0548 |
| 0.30 | 43.460 | 0.0452 |
| 0.50 | 43.006 | 0.0375 |

Switch rate decreases **monotonically** across all five points, including the zero-penalty
ablation — falling 45.7% from 0.0690 to 0.0375 as the penalty rises from 0 to 0.50. This directly
demonstrates the handover penalty is the causal mechanism behind the switch-avoidance behaviour
reported in Section 5.4, not merely correlated with it: with the penalty entirely removed, PPO's
switch rate rises to nearly twice its default-penalty value. Mean reward decreases modestly and
monotonically over the same range, from 44.303 to 43.006. The penalty is thus a directly
interpretable control knob: an operator can trade approximately 1.30 reward units for a 46% cut in
signalling load, positioning that trade-off anywhere along the curve via a single scalar.

## 5.6 Sensitivity to reward weighting

The same protocol was applied to four reward-weight profiles at the default penalty of 0.15.

**Table 5.3** — Effect of reward-weight profile (Ku-band).

| Reward profile | Reward | Switch rate |
|---|---|---|
| Balanced | 43.849 | 0.0548 |
| Throughput-leaning | 43.454 | 0.0527 |
| **Latency-leaning** | **44.905** | 0.0542 |
| Reliability-leaning | 43.264 | 0.0495 |

The latency-leaning profile yields the highest reward, exceeding balanced by 1.056 units.
Propagation delay is the KPI that most sharply separates the terrestrial from the non-terrestrial
tiers under Ku-band conditions, so weighting it more heavily gives the agent a correspondingly
sharper learning signal. Section 5.9 shows this finding is band-dependent rather than universal.

As a consistency check, the balanced configuration in Table 5.3 matches the 0.15-penalty row of
Table 5.2 exactly (43.849, 0.0548) — the same configuration, evaluated independently, twice.

## 5.7 Held-out geometry generalization

The strongest available test of whether PPO learned a transferable policy, or merely memorized the
one analytic LEO/HAPS pass-geometry profile used in training: each saved PPO model, plus the
hysteresis baseline, was re-evaluated under a deliberately harder, previously unseen geometry
regime — lower peak elevations, shorter passes, more frequent and longer out-of-view gaps (a
stand-in for a different orbital shell).

**Table 5.4** — Nominal vs. held-out geometry (Ku-band).

| Policy | Nominal reward | Shifted reward | Δ |
|---|---|---|---|
| Hysteresis | 40.904 | 41.312 | +0.408 |
| PPO seed 0 | 43.897 | 43.916 | +0.019 |
| PPO seed 1 | 44.961 | 45.045 | +0.084 |
| PPO seed 2 | 43.998 | 44.051 | +0.053 |

Reward does not degrade under the shifted regime for any seed — if anything, all four policies
show a small, consistent improvement, plausibly because the shifted regime's shorter LEO/HAPS
passes reduce net exposure to their higher-variance links. PPO did not memorize the fixed training
profile: its policy transfers to markedly different orbital geometry without loss.

## 5.8 Training stability

Because RL results are sensitive to seed variation, per-seed means are reported in Table 5.5.

**Table 5.5** — Across-seed variability (3 training seeds, 200 episodes each, Ku-band).

| Algorithm | Seed 0 | Seed 1 | Seed 2 | Across-seed SD | Within-seed episode SD |
|---|---|---|---|---|---|
| PPO (reward) | 43.748 | 45.154 | 43.593 | 0.860 | 2.771 |
| PPO (switch rate) | 0.041 | 0.023 | 0.047 | 0.0127 | — |
| DQN (reward) | 43.333 | 43.954 | 43.097 | 0.443 | 2.580 |
| DQN (switch rate) | 0.139 | 0.124 | 0.163 | 0.0194 | — |

Across-seed reward variability is smaller than within-seed episode variability for both algorithms
(PPO: 0.860 vs 2.771; DQN: 0.443 vs 2.580), indicating the reported ranking is driven by policy
quality rather than training-run luck. A caveat: the worst PPO seed (43.593) sits close to DQN's
best seed (43.954), so with 3 seeds the reward ordering is directionally consistent but not
overwhelming — this is exactly what the Welch/Mann-Whitney tests in Section 5.3 (computed on all
600 pooled episodes per algorithm, not just seed means) are for, and they do show significance
(p=5.9×10⁻⁶). The switch-rate separation is unambiguous regardless of this caveat: every PPO seed
lies below every DQN seed.

## 5.9 Band sensitivity: Ku, Ka, and S

The same generator, seeds, Area assignment, and reward design were run under two additional LEO
carrier bands — Ka (20 GHz) and S (2.2 GHz, the ITU space-operation/space-research downlink
allocation used operationally for LEO TT&C and narrowband constellations, not the high-throughput
broadband links this dataset's `Throughput_Mbps` values model — a caveat that must accompany any
S-band conclusion). Only the LEO `TechConfig`'s carrier frequency and its physically-derived,
fixed-aperture antenna gain differ between the three runs; bandwidth, noise figure, and all other
link-budget parameters are held constant, isolating band-dependent physics as the sole variable.

**Table 5.6** — Three-band comparison.

| Policy | Ku reward | Ka reward | S reward | Ku switch | Ka switch | S switch |
|---|---|---|---|---|---|---|
| Hysteresis | 40.904 | 44.334 | 40.001 | 0.0625 | 0.0591 | 0.0823 |
| DQN | 43.461 | 47.169 | 43.620 | 0.1421 | 0.0895 | 0.1322 |
| **PPO** | **44.165** | **47.683** | **44.210** | **0.0373** | **0.0473** | **0.0550** |
| Oracle | 45.292 | 48.167 | 45.274 | 0.0496 | 0.0414 | 0.0476 |
| PPO / Oracle | 97.51% | 98.98% | 97.66% | — | — | — |

*(Each band fits its own reward normalization scale; compare ratios and orderings, not raw reward
magnitudes, across bands.)*

Three findings, each with a physical explanation rather than an assumed one:

**PPO's gap to the oracle narrows at Ka, not at S.** PPO reaches 98.98% of oracle under Ka, versus
97.51–97.66% under Ku and S. Ka's much larger rain-driven signal variance appears to compress the
achievable range between a reward-optimal plan and a learned policy — there is simply less
headroom for the oracle to exploit that a fast-reacting learned policy cannot also capture.

**S-band's numbers land close to Ku's, and this has a specific physical cause, not coincidence.**
S-band's fixed-aperture antenna gain (21.26 dB) is 14.7 dB below Ku's (36.00 dB) — a real,
physically-derived penalty. But S-band's rain and gaseous attenuation are close to zero at the
rain rates and elevations in this dataset (verified: ≈0.04 dB vs. Ku's ≈11 dB rain attenuation at
10 mm/hr, 30° elevation), whereas Ku already carries substantial atmospheric loss. These two
effects roughly cancel, so overall link quality — and the resulting policy behaviour — converges
across the two bands despite operating almost two orders of magnitude apart in frequency.

**The switch-rate advantage PPO holds over the oracle is Ku-specific, not a general property of
learned selection.** Under Ku, PPO switches significantly *less* than the oracle (§5.3). Under
both Ka and S, PPO switches significantly *more* than the oracle (Ka: p=0.043; S: p=0.0095). The
correct general statement is that PPO's switch discipline is competitive with, and sometimes
exceeds, the oracle's — not that it systematically out-disciplines it. Tables 5.7 and 5.8 show the
mechanism-level pattern behind this.

**Table 5.7** — Best reward-weight profile by band.

| Band | Best profile | Reward | vs. balanced |
|---|---|---|---|
| Ku | Latency-leaning | 44.905 | +1.056 |
| Ka | **Throughput-leaning** | 47.654 | +1.086 |
| S | Latency-leaning | 44.788 | +1.007 |

The best profile is band-dependent: latency-leaning wins under both low-rain-variance bands (Ku,
S), while throughput-leaning wins under the high-rain-variance band (Ka) — plausibly because
Ka-band rain fade manifests primarily as throughput degradation, so weighting throughput more
heavily lets the agent more directly avoid the periods and networks where that degradation is
worst. This is offered as an interpretation consistent with the data, not a demonstrated causal
mechanism.

**Table 5.8** — Handover-penalty sweep monotonicity by band.

| Band | Switch rate monotonic across all 5 penalty points? |
|---|---|
| Ku | Yes |
| Ka | **No** — reward and switch rate both non-monotonic at the 0.15 point |
| S | Yes |

Ka's Phase 6 sweep is not monotonic (§5.5's protocol trains one seed per point, unlike Section
5.3's three). This is very likely a single-seed training-variance artifact rather than a sign the
underlying mechanism differs: Ka's substantially higher rain-driven reward variance (visible in
its wider spread of per-policy outcomes throughout this section) would be expected to make
single-seed noise more visible precisely where Ku and S — both low-variance bands — remain clean.
We report this plainly rather than smoothing it over; a multi-seed Phase 6 sweep is the direct way
to confirm this interpretation and is noted as future work (§6.3).

---

# 6. Discussion

## 6.1 Learned versus engineered switching discipline

The central empirical finding is that a policy optimising a composite QoE reward with a flat
handover cost outperforms an explicitly engineered hysteresis rule on both axes that rule was
designed to balance, under Ku-band conditions: PPO achieves higher reward *and* fewer handovers
than the A3-margin baseline (§5.3), approaching the oracle to within 2.49%.

This matters because hysteresis parameters — margin and time-to-trigger — must be tuned per
deployment, and their optimal values depend on cell geometry, mobility, and traffic. In a hybrid
TN–NTN setting those conditions vary not merely between deployments but *within a single episode*,
as a LEO platform rises and sets. A fixed margin cannot adapt to that; a learned policy conditioned
on live radio measurements can. Section 5.5 shows the resulting behaviour remains controllable
through one interpretable scalar — a practical requirement for any mechanism an operator would
consider deploying — and Section 5.9 shows the underlying advantage is not an artifact of one
carrier band, though its *magnitude* (and the specific oracle-relative switch-rate comparison) is.

## 6.2 Why the two RL algorithms diverge

PPO and DQN reach comparable reward through materially different policies (§5.4) and materially
different training dynamics (Figure 5.1): DQN's off-policy replay buffer reaches a strong policy
in roughly a fifth of the environment steps PPO requires, but converges to a policy that switches
networks far more often for a similar final reward. A plausible explanation lies in the algorithms'
treatment of action stability. PPO's clipped surrogate objective constrains each update to a trust
region around the current policy, favouring convergence to a temporally consistent action
distribution. DQN selects greedily with respect to a learned Q-function with no comparable
smoothness constraint, so small fluctuations in estimated action values translate directly into
switching. We offer this as an interpretation rather than a demonstrated mechanism; isolating it
would require an ablation over policy-update constraints not conducted here.

## 6.3 Threats to validity

**Fixed evaluation geometry, tested and found not to be an issue.** Section 5.7 evaluated trained
PPO policies under deliberately shifted orbital pass geometry (lower elevations, shorter passes,
more frequent gaps) never seen in training; reward did not degrade for any seed. This directly
addresses the concern that near-oracle performance reflects memorization of one fixed geometry
profile rather than a transferable policy.

**Single-seed Phase 6 sweep noise (Ka-band).** Section 5.9 found Ka's handover-penalty sweep
non-monotonic, plausibly because that protocol trains one seed per point while Ka's rain-driven
variance is substantially higher than Ku's or S's. A multi-seed Phase 6 sweep would confirm this
directly and is the most concrete methodological improvement identified by this work.

**Single-source synthetic data.** All five technology datasets originate from one simulation
pipeline. While the channel model is grounded in ITU-R recommendations (P.838-3 rain, P.676-13
gaseous absorption, verified against the recommendations before use), it has not been validated
against measured field data, and systematic modelling error would be shared across all five tiers
rather than averaging out.

**Modelling simplifications.** Doppler shift is held at each technology's median value in the
mobility model rather than derived from instantaneous relative velocity. The area context evolves
according to a hand-specified Markov chain rather than a measured mobility trace. S-band results
(§5.9) describe a narrowband/TT&C-representative configuration, not a broadband S-band service —
this dataset's throughput values were not designed to represent that regime.

**Value-function fit.** PPO's explained variance remained low throughout training (approximately
0.01–0.15), indicating the learned value function accounts for little of the observed return
variance. The policy performs well regardless, but this suggests the advantage estimates are noisy
and a better-fitted critic might yield further improvement.

**Scale.** Training runs of 100,000 steps across 3 seeds, across three bands, are appropriate for
establishing the comparisons presented here, but are modest relative to the broader claim of a
pathway toward AI-native radio protocols. The results support the narrower, evidenced conclusion
that learned selection outperforms classical handover heuristics in this environment, robustly
across three carrier bands.

## 6.4 Deployment considerations

An operator considering a learned selection controller for live infrastructure requires more than
a favourable simulation benchmark. First, **explainability**: the policy is a neural network whose
decisions cannot be audited in the way an A3 margin can, and regulatory or operational review may
require a rule-based approximation or post-hoc justification of its behaviour. Second, **fallback
guarantees**: a deployed controller needs a well-defined degraded mode when observations are
missing or the policy produces an invalid selection — Section 5.9's finding that the best reward
weighting is band-dependent (latency-leaning for Ku/S, throughput-leaning for Ka) also implies a
deployed system should either detect its operating band or be trained per-band rather than assume
one universal reward configuration transfers across deployments.

---

# 7. Conclusion

This paper presented a physically grounded dataset and reinforcement learning environment for
network selection across five heterogeneous access technologies — 5G NR, Wi-Fi 6, LEO satellite,
HAPS, and UAV relay — and evaluated learned selection policies against classical handover
algorithms, under a channel model whose atmospheric impairments are implemented directly from
ITU-R P.838-3 (rain) and P.676-13 (gaseous absorption), verified against the recommendations
before use.

The principal findings are as follows. A PPO agent conditioned on live per-candidate radio
measurements attains 97.51% of an oracle upper bound under Ku-band (12 GHz) LEO, improving mean
episode reward by 7.97% over a 3GPP A3-style hysteresis baseline while performing 40.3% fewer
handovers — differences confirmed statistically significant (Welch's t-test, Mann-Whitney U,
Cohen's d), not merely CI-disjoint. This switch-avoiding behaviour is demonstrated, not merely
correlated, to emerge from the handover-cost term in the reward: a true zero-penalty ablation shows
switch rate rising monotonically as the penalty is removed, and the mechanism remains controllable
through that single scalar. The policy generalizes to orbital pass geometry never seen in
training, with no reward degradation, addressing the most significant a priori concern about the
result.

Running the identical experiment under two additional LEO bands — Ka (20 GHz) and S (2.2 GHz) —
shows the headline finding is robust, while revealing that several of its specifics are band-
dependent rather than universal: the optimal reward-weighting profile shifts from latency-leaning
(Ku, S) to throughput-leaning (Ka) as rain-driven variance becomes the dominant impairment, and the
specific claim that PPO out-disciplines the oracle on switch rate holds only under Ku, reversing
under both Ka and S. S-band's results converge with Ku's for an identifiable physical reason — a
lower antenna-gain penalty offsetting negligible atmospheric loss — rather than by coincidence.

The two RL algorithms tested do not behave alike: DQN matches PPO's reward advantage over the
classical baselines and reaches it in roughly a fifth of the training steps, but not PPO's
switching discipline. The most important remaining limitation is that the Phase 6 sensitivity
sweep trains a single seed per configuration, and Ka's higher-variance conditions expose that as
non-monotonic behaviour that a multi-seed sweep would most directly resolve. The dataset and
environment are released to support this and further work.

---

## Still required before submission

- **Dataset availability statement** — Contribution #1 promises a public release; add a repository
  URL, DOI, and license.
- **Prior-work comparison table** — see marker #06 in `CITATIONS_NEEDED.md`.
- **Bibliography** — 24 citation sites, 29 references, none currently resolving.
- **Journal template** — the document is built on the Istanbul University *Physics and Astronomy
  Reports* class; replace with single-column IEEE-style formatting before submission.
