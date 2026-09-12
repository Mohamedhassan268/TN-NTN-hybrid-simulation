# Peer Review: “AI-Native Network Selection for 6G Hybrid TN–NTN Systems Using Deep Reinforcement Learning”

## Manuscript information

- **Authors:** Mohamed Hussien Moharam, Mohamed Hassan, Omar Ahmed Shwqy, and Mohamed Esam
- **Manuscript type:** Simulation-based systems and machine-learning study
- **Review basis:** Submitted DOCX manuscript, including equations, figures, tables, references, and rendered page layout
- **Suggested decision:** **Major revision / weak reject in its current form**
- **Reviewer confidence:** High

## Executive summary

The manuscript studies network selection across 5G NR, Wi-Fi 6, LEO satellite, HAPS, and UAV access using conservative Q-learning (CQL), proximal policy optimization (PPO), deep Q-networks (DQN), heuristic baselines, and a dynamic-programming oracle. The topic is relevant, and the attempt to combine five access types, handover costs, frequency-band sensitivity, and an out-of-distribution geometry test is useful. The Ku-band PPO result—mean reward 44.165 versus 40.904 for the hysteresis baseline, with a lower switch rate—is potentially publishable.

The current evidence, however, does not yet support the strength of the conclusions. The largest problems are an unresolved contradiction about whether CQL initializes the online agents, pseudoreplication in the statistical tests, reward comparisons made under changing objective functions, under-specified state and simulator definitions, and several inconsistencies between the dataset description and the reported figures. The manuscript also overstates physical realism and deployment readiness. These are substantive issues, not merely editorial ones.

The prose needs a separate revision. The introduction, related work, and parts of the discussion use promotional abstractions, repeated three-part lists, section-preview language, and defensive commentary. Those patterns make the text sound generic and machine-produced. The appropriate remedy is not to optimize for an AI detector—such detectors are unreliable—but to make the paper more specific, restrained, and visibly grounded in the authors’ actual modeling decisions and evidence.

## Main contributions as presently understood

1. A synthetic, standards-informed simulation dataset spanning five terrestrial and non-terrestrial access types.
2. An offline contextual-bandit experiment using CQL and an online sequential network-selection environment using PPO and DQN.
3. Comparison with greedy, random, hysteresis, and clairvoyant dynamic-programming policies.
4. Ku-, Ka-, and S-band sensitivity studies, a handover-penalty sweep, and a held-out pass-geometry experiment.

These contributions should be stated this directly. At present, the paper alternates between describing one integrated offline-to-online framework and two independent experiments, which changes the nature of the claimed contribution.

## Strengths

- The paper addresses a practical control problem created by the different coverage, delay, bandwidth, and mobility properties of TN and NTN links.
- It evaluates learned policies against both classical baselines and an oracle-style upper bound.
- Handover frequency is treated as an operational outcome rather than reporting reward alone.
- The authors attempt frequency sensitivity and a geometry shift rather than limiting evaluation to one stationary scenario.
- The limitations section candidly acknowledges the synthetic data source, fixed median Doppler, simplified mobility, and low PPO explained variance.
- The intent to release data and code is valuable, provided that the referenced repository, environment, configurations, and seeds are complete and accessible.

## Assessment of the principal claims

| Claim | Current support | Main reason | Required revision |
|---|---:|---|---|
| PPO approaches the oracle and outperforms hysteresis in Ku band | Moderate | The observed effect is promising, but the baseline is not demonstrably tuned and the inferential statistics use non-independent episodes | Re-run a paired evaluation on common scenarios, tune baselines on validation scenarios, and report seed-level uncertainty |
| Handover penalty reduces ping-pong behavior | Weak to moderate | The intervention supports the direction of the claim, but each setting appears to use one training seed and total rewards use different objectives | Use multiple seeds and report raw QoE and switch rate under a fixed evaluation definition |
| The policy generalizes to held-out geometry | Weak | All policies improve under the new geometry, so the shift may simply be easier | Test several randomized held-out geometries and report oracle-normalized regret and raw link availability |
| Optimal reward profiles differ by frequency band | Weak | Each profile is assessed using its own weighted reward, and the sweep appears to use one seed per condition | Evaluate all policies under one common reference utility or present a raw-KPI Pareto analysis |
| The dataset is physically realistic and 3GPP-compliant | Weak | Several parameters are heuristic, some output extremes are implausible, and no external validation is presented | Narrow the terminology and validate link-level outputs against standards, published simulators, or measured ranges |
| The method is ready for real-time deployment | Unsupported | No inference-time, compute-budget, protocol, or field evaluation is reported | Remove the deployment claim or add latency, resource, and implementation measurements |

## Major concerns

### 1. The architecture is described in two incompatible ways

The abstract and Section 4.4 state that CQL and PPO/DQN are trained and evaluated independently and that CQL does not initialize PPO. In contrast, Figure 1, Section 2.5, and the accompanying framework description show CQL producing an initial policy that PPO or DQN subsequently refines. These are different methods.

The authors should decide which experiment was actually run and revise the title, abstract, diagram, contribution list, related work, and methods accordingly. If the experiments are independent, Figure 1 should show two separate branches. If CQL initializes online learning, the initialization mechanism, transferred parameters, training schedule, and ablation against training from scratch must be reported.

Figure 1 also names ns-3 and SNS3 as data-generation components, whereas the methods describe a single custom simulation pipeline. Retain those labels only if those tools were actually used and the integration is reproducible.

### 2. The statistical tests treat nested episodes as independent replicates

PPO and DQN are trained with three independent seeds, after which 200 evaluation episodes per seed appear to be pooled into a sample of 600. Welch and Mann–Whitney tests then operate on the pooled episodes. Episodes produced by one trained model share the same learned parameters and are not independent training replicates. Pooling them substantially inflates the apparent sample size and can yield extremely small p-values without demonstrating reliable performance across training runs. The deterministic baselines also use 200 episodes, creating an asymmetric comparison.

Use common paired evaluation scenarios for every policy. The independent unit for claims about training variability should be the training seed, not each episode. Increase the number of training seeds—ideally to at least 10 if computationally feasible—and show the seed-wise distribution. A hierarchical model or a bootstrap that resamples seeds first and episodes second would respect the design. With only three seeds, emphasize effect sizes, paired differences, and uncertainty rather than claims based on p-values such as \(10^{-53}\) or \(10^{-165}\).

### 3. Several sensitivity comparisons change the quantity being optimized

The reward-profile study compares policies using rewards whose weights differ by profile. A policy evaluated under its own objective can appear “best” because the measuring scale changes with the policy. The handover-penalty sweep has the same problem: total reward at \(\lambda=0\) is not directly comparable with total reward at \(\lambda>0\). Consequently, statements such as trading 1.30 reward units for a 46% reduction in switching do not describe an invariant operational tradeoff.

Evaluate every trained policy using one fixed reference utility, or avoid collapsing the outcomes into one reward. A stronger analysis would report throughput, latency, loss or reliability, invalid actions, and handovers separately and present the Pareto frontier. Use the same normalization bounds across bands if cross-band rankings or oracle ratios are compared.

### 4. The simulator and data description contain internal inconsistencies

The manuscript alternates among 23 features, 19 numeric features across 23 columns, and 20 persisted features plus three transformations. Provide one definitive schema stating each raw field, derived field, unit, data type, range, and whether it is visible to the agent.

Other inconsistencies require correction:

- Section 3.2 lists Urban, Suburban, Rural, and Indoor, while the reported areas are Urban, Indoor, Rural, Highway, Maritime, and Desert.
- The availability combinations `NR_5G, WiFi, UAV, HAPS, SAT` and `NR_5G, WiFi, HAPS, SAT, UAV` are the same set in a different order. They should not be treated as different categories.
- Elevation angle is said not to be persisted, yet a correlation of \(\rho=0.42\) involving elevation angle is reported.
- The text says Indoor is dominated by Wi-Fi and LEO, but the heat map shows more 5G NR and satellite samples than Wi-Fi samples. Similar descriptions of the Rural, Maritime, and Desert rows do not match the plotted counts.
- Equal numbers of observations for 20 UE identifiers do not demonstrate 20 distinct mobility trajectories. Trajectory generation and any group-wise split must be documented.

Because the dataset is synthetically balanced by construction, ordinary descriptive percentages and significance tests do not establish realism or external validity.

### 5. The physical model needs correction and validation

Several equations or descriptions are dimensionally or conceptually unclear:

- Equation 4 places `C_dB` inside a linear reciprocal sum. The term should be a linear power ratio, with a separate conversion from decibels.
- The final SINR expression takes a minimum of separately derived quantities. Hardware-ceiling and interference effects should be combined consistently in the linear domain, and the equation must match the code.
- Equation 7 calls a Gaussian gap non-negative, although an untruncated Gaussian can produce negative values. Define clipping, truncation, or an appropriate positive distribution.
- Equation 12 is described as obsolete but remains in the main method, followed by an orphan explanatory sentence. Remove it or move it to a clearly labeled historical note.
- The reward says BER is log-transformed and “higher is better.” An ordinary logarithm of BER has the opposite direction. State the exact transform, bounds, and normalization.
- The displayed CQL loss uses an expectation under \(\mu(a\mid s)\), whereas standard CQL implementations commonly use a conservative log-sum-exp term or a sampled approximation. Report the exact implemented objective and cite the implementation.

The output ranges also merit a plausibility audit: maximum SNR of 96.7 dB, SINR of 91.93 dB, RSSI of +1.70 dBm, spectral efficiency of 24.36 bit/s/Hz, and throughput of 1.827 Gbit/s are extreme for the described mobile links. Report per-technology distributions and explain distance floors, link budgets, interference, antenna gains, MCS or spectral-efficiency caps, and saturation behavior. Correlation among SNR, SINR, BER, throughput, and LQI is partly imposed by the generator and should not be presented as independent validation of realism.

The term “3GPP-compliant” is too broad for a simulator that also uses a hand-built area Markov chain, canonical analytic passes, heuristic rain-path assumptions, fixed median Doppler, Gaussian interference margins, and Shannon capacity. “Standards-informed link-level simulation” is more accurate unless every claimed standard procedure is demonstrated and validated.

### 6. The MDP and training procedure are not reproducible from the paper

The state is reported as 26-dimensional, but five RSSI values, five SINR values, six area indicators, and five previous-network indicators total 21. If the remaining five values are an availability mask, state this explicitly. Provide a table with the exact ordering, normalization, missing-value behavior, and observation available at decision time.

The duration of one time step is also absent. Without \(\Delta t\), a 60-step episode, a three-step time-to-trigger, switch rate, Doppler evolution, and a 10–15 minute satellite pass have no common physical interpretation.

At minimum, report:

- PPO and DQN network layers, widths, activations, optimizer, learning-rate schedule, discount, target-update and exploration settings, entropy/value coefficients, and stopping rule;
- software and library versions;
- all normalization constants and exact reward weights;
- environment and training seeds, data splits, and evaluation scenario seeds;
- invalid-action handling and availability-mask semantics;
- the area-transition matrix and mobility/pass generation;
- per-technology link-budget constants and units;
- whether the oracle is exact or approximate and what future information it receives.

The CQL result that selects five of six areas should not be called a generalization result if training and evaluation use the same 55,503 records. Use a held-out split, preferably by UE, trajectory, and scenario rather than a random row split.

### 7. Baseline comparisons are not yet sufficiently fair

The “3GPP-style” hysteresis policy uses a normalized SINR margin of 0.05 and a three-step time-to-trigger, but no tuning study is shown. PPO may be benefiting from comparison with an under-tuned rule. Tune hysteresis and time-to-trigger on validation scenarios and freeze them before testing.

The random policy apparently selects unavailable networks and is therefore only a sanity check. Add a random-valid policy. Since the greedy baseline uses only SINR while PPO optimizes a composite objective, add a myopic QoE-greedy or one-step look-ahead baseline using the same observable components. Describe the dynamic-programming policy as a clairvoyant upper bound and clarify whether it solves the finite-horizon problem exactly.

### 8. Robustness claims exceed the experiments

The handover-penalty and band/reward-profile sweeps appear to use one training seed per condition. The paper nevertheless uses causal and general language such as “monotonic,” “demonstrates the mechanism,” and “provides an operator control.” The non-monotonic Ka-band outcome already shows why training variability matters. Re-run the sweeps with multiple seeds and confidence intervals and soften the interpretation until then.

The held-out geometry is called harder, yet every policy improves. This result does not establish robust generalization. It may indicate that the new trajectory spends more time in favorable or less ambiguous coverage. Test several geometry distributions or randomized orbital/pass conditions, quantify their difficulty using link availability and oracle performance, and report regret relative to the oracle.

### 9. Deployment and protocol claims should be narrowed

The implementation is an abstract network-selection controller, not a MAC-layer vertical-handover protocol. No signaling procedure, protocol messages, execution timing, compute footprint, radio measurement delay, or field trial is evaluated. Replace “real-time deployment pathway” and related claims with a statement that the simulation is an initial algorithmic evaluation. If real-time suitability is important, report inference latency on specified hardware and discuss observation latency, signaling overhead, and safe fallback behavior.

The very low PPO explained variance (approximately 0.01–0.15) should be investigated. It can indicate a poorly fitted critic, high return noise, or unsuitable value-function settings. Show learning curves across seeds, value loss, entropy, KL divergence, episode-length behavior, and sensitivity to critic architecture or reward scaling.

## Methodology assessment

| Dimension | Rating | Assessment |
|---|---:|---|
| Technical soundness | 2/5 | Promising design, but the architecture, state, reward, and several equations are inconsistent or under-specified |
| Novelty | 3/5 | Five-tier selection with handover-aware DRL is a useful combination, though the conceptual methods are established |
| Reproducibility | 2/5 | The paper omits key simulator, model, seed, split, and timing details; repository accessibility should be verified |
| Experimental design | 2/5 | Multiple baselines and stress tests are included, but baseline tuning and common-objective evaluation are missing |
| Statistical rigor | 1/5 | The principal tests use pooled, nested episodes as independent observations, and most sweeps use a single seed |
| Scalability/deployment evidence | 2/5 | Five actions are manageable, but compute, latency, signaling, and larger-system behavior are not evaluated |

## Literature positioning

The related-work section should be updated and should distinguish among (i) link-level simulation, (ii) network-selection or mobility control, and (iii) resource allocation in integrated TN–NTN systems. Recent adjacent work includes a 2025 study of [hierarchical deep reinforcement learning for resource management in integrated TN–NTN networks](https://arxiv.org/abs/2501.09212) and a 2024 [reinforcement-learning approach to integrated terrestrial and non-terrestrial network optimization](https://arxiv.org/abs/2410.06700). The simulator discussion should also compare the proposed pipeline with [OpenNTN](https://github.com/ant-uni-bremen/OpenNTN), which implements 3GPP TR 38.811 channel components, and [LLSim5G](https://github.com/EFontesP90/LLSim5G), an open link-level 5G simulator.

Several existing citations appear mismatched to the claims they support. Reference [15] is a survey of 5G security but is cited for limitations of hybrid TN–NTN handover analysis. Reference [19] concerns age-of-information-aware vehicular resource management but is described as multi-agent 5G/Wi-Fi coexistence. Reference [21] is a broad prospective 6G article but is used for a specific federated-learning-in-NTN claim. Reference [23], IEEE 802.11ax, is not an indoor propagation-model reference, and DVB-S2X [25] does not directly establish orbital pass geometry. References [8]–[10] should also be checked against DOI or publisher records; I could not verify the exact bibliographic records from their current titles and metadata.

## Writing and “AI-generated” style audit

### Overall finding

The prose often sounds formulaic, especially in the introduction, related work, section openings, and conclusion. This does not prove AI authorship, and an AI-detector score should not be treated as a scientific quality measure. The concern is editorial: generic abstractions obscure what was actually built and tested.

The main patterns are:

- promotional phrases such as “foundational pillar,” “next transformative leap,” “seamless continuum,” “transformative paradigm,” and “critical research frontier”;
- repeated three-part or five-part lists with nearly identical sentence rhythms;
- frequent road-map language: “This section presents… We begin… followed by… and conclude…”;
- vague validating verbs: “confirms,” “underscores,” “highlights,” “demonstrates,” and “validates” where the evidence is limited;
- defensive meta-commentary such as “not merely,” “we report this plainly,” “not coincidence,” and “rather than an assumed one”;
- long sentences that combine motivation, novelty, evaluation, and implications before giving a verifiable technical fact.

Revise by making each paragraph perform one job: define the problem, state a method choice, report a result, or qualify its interpretation. Prefer concrete nouns, numbers, and conditions. Remove claims of novelty or importance that the reader can infer from the evidence.

### Example rewrites

**Opening motivation**

> 6G research increasingly considers integrated terrestrial and non-terrestrial access. ITU-R’s IMT-2030 framework includes ubiquitous connectivity and AI-assisted communication among its use scenarios [1]. Supporting these scenarios requires a controller that can select among access technologies with different coverage, latency, and mobility characteristics.

This is more credible than describing 6G as a “transformative leap” or a “seamless, self-aware fabric.” It states the external premise and immediately identifies the technical problem.

**System description**

> The system considered here includes 5G NR, Wi-Fi 6, UAV relays, HAPS, and LEO satellite links. These technologies differ in coverage, bandwidth, propagation delay, and channel variation. Network selection must therefore use more than a single received-power threshold.

**Contribution statement**

> This study makes four contributions. First, it constructs a standards-informed simulator and synthetic dataset covering five access technologies. Second, it defines a sequential network-selection environment with link-quality observations and an explicit handover cost. Third, it compares PPO and DQN with tuned heuristic policies and a clairvoyant dynamic-programming upper bound. Fourth, it tests sensitivity to frequency band, reward weights, handover penalty, and pass geometry.

Use “public” only after confirming that the repository and required artifacts are accessible.

**Results interpretation**

Instead of “The results unequivocally demonstrate the superior, robust, and intelligent behavior of the proposed AI-native framework,” write:

> In the Ku-band simulation, PPO achieved a mean episode reward of 44.165, compared with 40.904 for the hysteresis policy. Its switch rate was also lower (0.0373 versus 0.0625 per step). Because the comparison uses three training seeds and a synthetic environment, the result should be interpreted as evidence within the tested simulator rather than as a deployment-level conclusion.

**Conclusion**

> In the simulated Ku-band environment, PPO achieved mean reward 44.165 versus 40.904 for the hysteresis baseline and used fewer handovers (0.0373 versus 0.0625 per step). The result persisted in the Ka- and S-band frequency-sensitivity runs, although the preferred reward weights and oracle-relative switch rate changed by band. These findings apply to the present synthetic environment; field traces, realistic mobility, stronger baseline tuning, and more training seeds are needed before drawing deployment conclusions.

### Practical editing rules for the full manuscript

1. Delete most “This section…” road maps; keep only the paper-level organization paragraph if required by the venue.
2. Replace broad adjectives with a measurable fact or remove them.
3. Keep one main claim per sentence and one evidentiary purpose per paragraph.
4. Use “suggests” or “is consistent with” for single-seed and synthetic-data results.
5. Do not call a correlation “validation” when both variables are computed from the same underlying equation.
6. Report limitations beside the result they constrain rather than collecting all caveats at the end.
7. Preserve the authors’ natural technical vocabulary; avoid mechanically replacing words with synonyms, which usually makes detector-oriented editing worse.

## Document and layout audit

The manuscript’s 33 rendered pages contain several production issues that should be fixed before submission:

- Page 9 is almost blank and Table 1 appears to be missing or broken; only the following paragraph is placed near the bottom.
- Pages 7, 15, 22, 31, and 33 contain large unexplained blank areas, likely caused by section breaks, keep-with-next settings, or anchored tables.
- The format changes abruptly at page 23: headers and page numbers appear, fonts and spacing become denser, and numbered section headings are replaced by plain bold labels such as “RESULTS.” This looks like two templates were merged.
- Equations 2, 13, and 15 have visibly broken or overprinted notation. Rebuild them with native Word equations or clean LaTeX conversion.
- Captions, punctuation, underlining, and hyperlink color are inconsistent.
- The author and corresponding-author numbering is malformed. The standalone heading “Authors” is unnecessary.
- The sentence “This paper has been typeset from a TEX/LATEX file” is misleading in the submitted DOCX unless it is inserted automatically by the target publisher.
- American and British spelling alternate, including “behavior/behaviour” and “modeled/modelled.” Choose the target venue’s convention.

## Questions for the authors

1. Was PPO or DQN initialized from CQL in any reported experiment? If so, what was transferred and where is the ablation?
2. Were ns-3 or SNS3 used to generate the reported dataset, or is Figure 1 conceptual?
3. What are the exact 26 state variables and what is the duration of one time step?
4. Are the 200 evaluation episodes paired across all policies, and how many independently trained policies exist per algorithm?
5. Are policies in the reward-profile and handover-penalty sweeps evaluated with a common metric?
6. How are unavailable actions handled during both training and evaluation?
7. Which normalization bounds are learned per band, and are cross-band reward/oracle ratios comparable?
8. Why is the dynamic-programming oracle described as approximate? Does it know the full future trajectory?
9. What explains the low PPO explained variance, and was critic instability investigated?
10. Can the authors provide validation showing that each technology’s simulated distributions fall within defensible operating ranges?

## Minor and editorial issues

- Define TN, NTN, HAPS, UAV, CQL, PPO, DQN, TTT, and all link metrics at first use and use the same abbreviations thereafter.
- Distinguish propagation delay from end-to-end latency, particularly when comparing LEO with statements about satellite delays of hundreds of milliseconds.
- Frame the S-band result as a frequency-sensitivity experiment if the assumed 250 MHz bandwidth is not representative of the intended service.
- Report units in every table header rather than repeatedly in prose.
- Avoid referring to a reordered list as a distinct network-availability combination; canonicalize sets before counting.
- Replace “MAC-layer vertical handover protocol” with “network-selection policy” unless an actual protocol is specified.
- Number all results subsections consistently and ensure every table and figure is cited in numerical order.
- Verify every reference against a DOI, publisher page, or authoritative repository and ensure each citation supports the sentence in which it appears.

## Recommended revision sequence

1. Establish the true experimental architecture and reconcile CQL, PPO, DQN, Figure 1, and all contribution claims.
2. Correct the state, timing, reward, equation, feature-schema, and data-category inconsistencies.
3. Re-run experiments with paired scenarios, more seeds, tuned and fair baselines, and common-objective or Pareto evaluation.
4. Validate physical ranges and narrow “3GPP-compliant,” “real-time,” “protocol,” and deployment claims.
5. Update and audit the literature and references.
6. Rewrite the introduction, related work, discussion, and conclusion in restrained, evidence-led language.
7. Rebuild the document layout, missing table, equations, headings, captions, and references under one template.

## Final recommendation

**Major revision / weak reject in the current form.** The research question and experimental ambition are worthwhile, and the PPO-versus-hysteresis result may become a useful contribution. Publication should depend on resolving the architecture contradiction, repairing the statistical design, making comparisons invariant to reward definitions, validating the simulator, and aligning the claims with what was actually tested. A prose-only revision would not be sufficient.

## Review limitations

This review assessed the submitted manuscript and its rendered layout. I did not execute the simulator or reproduce the experiments, and I could not independently inspect the referenced repository. The target journal or conference was not specified, so the recommendation uses general standards for a systems/communications research paper rather than a venue-specific rubric.
