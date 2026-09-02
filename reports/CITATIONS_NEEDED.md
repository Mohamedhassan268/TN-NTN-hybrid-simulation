# Unresolved Citations — Extraction Map

**Source:** `FRIEND_AI_Native_Framework.docx` (repo root; renamed from the original `AI_Native_Network_Selection_Framework_for_Hybrid_TN_NTN_Systems_using_Deep_Reinforcement_Learning.docx`)
**Extracted:** 2026-08-30, directly from `word/document.xml`

**24 citation sites, 29 individual references.** (`?` = 1 reference, `??` = 2, `???` = 3.)

No `.bib`, `.tex`, or `.bbl` exists anywhere in this repository — the bibliography source lives
outside it. Nothing below is invented. Where the manuscript names an author in running text, the
canonical work is proposed and marked by confidence; where it does not, only the required reference
*type* is described.

**Confidence key:** **[HIGH]** = named author + described contribution match one universally
canonical paper. **[CONFIRM]** = author named but several plausible works; verify against your
intended source. **[SUPPLY]** = no author named; you must choose.

---

## Section 1 — Introduction

| # | Para | Claim being supported | Reference needed |
|---|---|---|---|
| 01 | 14 | IMT-2030 / 6G usage scenarios: AI-and-communication, ISAC, ubiquitous connectivity, massive twinning, enhanced HMI | **[SUPPLY]** ITU-R M.2160 (IMT-2030 Framework Recommendation) is the standard source for this exact five-scenario list |
| 02 | 18 | Multi-layer NTN architecture spanning LEO, GEO, HAPS, UAV | **[SUPPLY]** ×2 — an NTN architecture survey + a vHetNet/multi-layer overview |
| 03 | 19 | 3GPP A3-event framework: threshold rules, hysteresis margins, time-to-trigger | **[SUPPLY]** 3GPP TS 38.331 (RRC, A3 event definition), or TS 36.331 for the LTE lineage |
| 04 | 19 | LEO visibility windows of only a few minutes; Wi-Fi indoor micro-cells with stochastic wall penetration loss | **[SUPPLY]** Two distinct claims share one marker — a LEO pass-duration analysis, and an indoor propagation/penetration-loss model (ITU-R P.1238 or 3GPP TR 38.901) |
| 05 | 20 | DRL solves sequential decision problems with high-dimensional state, delayed reward, long-term dependencies | **[SUPPLY]** ×2 — Sutton & Barto (2018) textbook + a DRL survey |
| 06 | 20 | Existing body of DRL-based network selection work in terrestrial and satellite contexts | **[SUPPLY]** ×3 — these should be the closest prior works, and are the natural basis for the comparison table the panel requested |

## Section 2 — Related Work

### 2.1 TN–NTN integration and standardization

| # | Para | Claim | Reference needed |
|---|---|---|---|
| 07 | 34 | 3GPP NTN standardization from Rel-15, formalized in Rel-17 via 38.821 and 38.811 | **[HIGH]** ×2 — 3GPP TR 38.811 (NTN study) and TR 38.821 (NTN solutions). The text calls both "Technical Specifications"; both are **Technical Reports (TR)** — worth fixing |
| 08 | 35 | "Rinaldi et al." — comprehensive NTN integration survey across RAN, core, service layers | **[CONFIRM]** Rinaldi et al., "Non-Terrestrial Networks in 5G & Beyond: A Survey," IEEE Access, 2020 |
| 09 | 35 | "Vaezi et al." — evolution from 5G to 6G, role of NTN | **[CONFIRM]** Vaezi et al., "Cellular, Wide-Area, and Non-Terrestrial IoT: A Survey on 5G Advances and the Road Toward 6G," IEEE COMST, 2022 |
| 10 | 35 | "Giordani and Zorzi" — network selection / handover / RAT coordination as open challenges | **[CONFIRM]** Giordani & Zorzi, "Non-Terrestrial Networks in the 6G Era: Challenges and Opportunities," IEEE Network, 2021 |
| 11 | 35 | "Kodheli et al." — same challenge framing | **[CONFIRM]** Kodheli et al., "Satellite Communications in the New Space Era: A Survey and Future Challenges," IEEE COMST, 2021 |

### 2.2 Classical handover

| # | Para | Claim | Reference needed |
|---|---|---|---|
| 12 | 37 | 3GPP A3 event framework, hysteresis + TTT (second occurrence) | **[SUPPLY]** Same source as #03 — reuse one key |
| 13 | 40 | "Cao et al." — limits of classical handover under heterogeneous cell sizes | **[CONFIRM]** Author named, contribution described; several candidates |
| 14 | 40 | "Fu et al." — same | **[CONFIRM]** Author named; verify intended paper |

### 2.3 DRL foundations

| # | Para | Claim | Reference needed |
|---|---|---|---|
| 15 | 43 | "Mnih et al." — DQN, human-level control in high-dimensional tasks | **[HIGH]** Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518(7540):529–533, 2015 |
| 16 | 43 | "Van Hasselt et al." — Double DQN, overestimation bias | **[HIGH]** Van Hasselt, Guez & Silver, "Deep Reinforcement Learning with Double Q-Learning," AAAI 2016 |
| 17 | 43 | "Wang et al." — Dueling DQN, value/advantage decoupling | **[HIGH]** Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML 2016 |
| 18 | 44 | "Schulman et al." — PPO, clipped surrogate objective | **[HIGH]** Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017 |
| 19 | 44 | "Kumar et al." — CQL for offline RL | **[HIGH]** Kumar, Zhou, Tucker & Levine, "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS 2020 |

### 2.4 DRL for network selection

| # | Para | Claim | Reference needed |
|---|---|---|---|
| 20 | 45 | "Wang et al." — DQN for Wi-Fi/cellular vertical handover | **[CONFIRM]** Distinct from #17 despite the shared surname — **must not reuse that key** |
| 21 | 45 | "Xu et al." — actor-critic for satellite-terrestrial handover, LEO visibility windows | **[CONFIRM]** |
| 22 | 45 | "Chen et al." — multi-agent extension, distributed coordination | **[CONFIRM]** |

### 2.5 AI-native 6G

| # | Para | Claim | Reference needed |
|---|---|---|---|
| 23 | 50 | "Letaief et al." — architectural principles of AI-native 6G | **[CONFIRM]** Letaief et al., "The Roadmap to 6G: AI Empowered Wireless Networks," IEEE Comm. Mag., 2019 — *or* "Edge Artificial Intelligence for 6G," IEEE JSAC, 2022. "Architectural principles" fits the JSAC paper better |
| 24 | 51 | "Bariah et al." — Federated Learning in future NTN systems | **[CONFIRM]** Bariah et al., "A Prospective Look: Key Enabling Technologies, Applications and Open Research Topics in 6G Networks," IEEE Access, 2021 |

---

## Notes before you rebuild the bibliography

1. **#17 and #20 are different "Wang et al." papers.** Dueling DQN (ICML 2016) is not the
   Wi-Fi/cellular vertical-handover work. Assigning one key to both is an easy and damaging error.
2. **#03 and #12 are the same citation** used twice — one key, two uses.
3. **TR vs TS (#07).** TR 38.811 and TR 38.821 are Technical *Reports*; the manuscript calls them
   Technical Specifications.
4. **#06 is the highest-value site.** Those three references are the closest prior art, and the
   panel asked for a comparison table (domain / algorithm / dataset size / metrics) against exactly
   these. Choosing them well is what converts the "largely unexplored" gap claim from asserted to
   demonstrated.
5. **Markers #01–#06 are all [SUPPLY].** The entire Introduction cites nothing identifiable by name,
   so no amount of inference recovers it — those are yours to choose.
