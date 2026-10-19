# Simulated peer review: first draft of "Does Better Forecasting Buy Better Budget Pacing?"

Mode: `academic-paper` revision, self-review step (no external reviewers exist yet).
Draft reviewed: `paper/main.tex` at commit `700eea7`. Reviewer stance: skeptical, double-blind.
Target criteria: `criteria_binding_unavailable` (venue undecided), so this uses the field-general
five-dimension rubric and makes no venue-fit claim. `calibration_status: NOT_CALIBRATED`.
Numbers below were recomputed from files in `v2/results/` (cited per item); they are
review evidence, not paper content.

## 1. Initial impression

A clean empirical study with a real dataset, a protocol, tests and honest negative findings
(hedging hurts, forecaster choice barely matters). The main risk is not the experiments but the
framing: several headline sentences claim more than the evidence supports, and a few
uncertainty sources that the data could show are not shown.

## 2. Criterion-bound judgements

| Dimension | Judgement | Evidence | Decision-bearing? |
|---|---|---|---|
| Originality | MEETS | A fresh applied question (forecaster vs. hedging quantile inside a fixed throttle); related work does not claim it. Thin literature on forecast-driven pacing is acknowledged only implicitly. | No |
| Methodological rigor | PARTLY_MEETS | Protocol, common random numbers, cluster bootstrap, Holm. But tests ignore training-seed variance (M2), the logging-policy dependence is undisclosed (M4), and the quantile grid stops at the median (M5). | Yes |
| Evidence sufficiency | PARTLY_MEETS | Strong for "forecast throttle > even/PID" and "q=0.9 is costly". Weak for "TFT/NHITS clearly beat LightGBM", for "median is best", and for "forecasting itself is what helps" (M1, M3, M5). | Yes |
| Argument coherence | PARTLY_MEETS | Title asks whether *better* forecasting helps; the abstract's headline answers "forecasting helps" using even pacing as the comparison, which conflates using any demand profile with using a better forecaster (M1). | Yes |
| Writing quality | MEETS | Clear and short. Minor: PID and CI unexpanded, sPL scaling and MASE$_h$ undefined, config numbers typed in prose, "released" claims. | No |

## 3. Issues by severity

### Major

| # | Section | Issue | Evidence | Suggested fix | Needs new experiment? |
|---|---|---|---|---|---|
| M1 | Abstract, 6.2, 7, 8 | Headline gain over even pacing (3.4–5.4 pts) mixes two effects. A seasonal-naive demand profile alone already gains 1.4–4.1 pts over even pacing; the neural forecasters add only 1.3–2.1 pts on top. | `pacing_summary.csv`: naive q0.5 minus Even = 1.36 / 2.90 / 4.11 pts; best NN q0.5 minus naive q0.5 = 2.14 / 1.77 / 1.32 pts at rho = 0.3 / 0.5 / 0.7 | Say explicitly that most of the gain at tight and medium budgets comes from using a demand profile at all; report the added value of the learned forecaster separately. | No |
| M2 | 6.2, abstract | Significance of forecaster differences ignores training-seed variance. Seeds are pooled inside each per-campaign ratio, so Wilcoxon p-values condition on the three trained models. NHITS seed SD of the click ratio is 0.40–0.45 pts (q=0.5) and about 0.9 pts (q=0.9); the TFT–NHITS gaps are 0.05–0.16 pts, and the one "significant" gap (rho=0.7, 0.12 pts) is smaller than the seed SD. LightGBM and naive have a single run, so their seed variability is unknown. | `pacing_summary.csv`, column `click_ratio_seed_sd`; `pacing_comparisons.csv` rows TFT vs NHITS q=0.5 | Report seed SD next to the claims; state that tests condition on trained models; do not present any TFT–NHITS or NN–LightGBM gap below the seed SD as a difference. | Full fix (seed-aware inference or more seeds): yes, deferred to the author |
| M3 | 6.1, 8 | "Both clearly ahead of LightGBM" in accuracy is unsupported. Mean MASE favors the neural models, but the median campaign does not: median MASE is 0.722 (LightGBM) vs 0.723 (NHITS) vs 0.708 (TFT). Per-campaign wins vs LightGBM: TFT 70.5%, NHITS 56.0%. No interval or test is given for the MASE gaps. | `forecast_metrics.json` (`median_mase_over_campaigns`), `forecast_metrics_per_campaign.parquet` | Report median MASE and win rates; replace "clearly" with a description that matches them; hedge the conclusion that neural models are "the most accurate". | A CI/test on MASE is a new analysis: deferred to the author |
| M4 | 3, 7 | Undisclosed validity threat: the Criteo log contains only impressions *displayed* (won) under Criteo's own bidding and pacing. "Eligible demand" in the replay is therefore the stream the logging bidder won, already shaped by its policy; throttling it further is a counterfactual that the log cannot verify. Not in Limitations. | Dataset card: "each line corresponds to one impression that was displayed to a user" | State this in the setup and in the limitations. | No |
| M5 | 6.2, 7, 8 | "Median works best" and "plan against the median" overreach. Only q in {0.5, 0.7, 0.9} were tested and clicks fall monotonically with q in every forecaster and budget level, which implies the optimum may lie below 0.5. The explanation "upper quantiles overstate demand" is offered as fact; a plausible cause (summing hourly 0.9 quantiles bounds the daily total far above its 0.9 quantile) is untested. | `pacing_summary.csv` | Say "best of the three tested"; flag lower quantiles and a daily-total quantile as untested; present the mechanism as a hypothesis. | Full fix: yes (lower quantiles), deferred to the author |
| M6 | 7, 8 | Practical recommendations ("worth deploying") exceed the evidence: five test days, one dataset, a throttling-only simulator that ignores price and competition, and a gain over tuned PID of only 0.7–2.0 pts (0.8 pts at rho = 0.7, where the TFT CI lower bound 0.9936 barely clears the PID CI upper bound 0.9933). | `pacing_summary.csv` | Rephrase as conditional findings "in this setting"; keep the limitations visible next to the guidance. | No |

### Minor

| # | Section | Issue | Suggested fix |
|---|---|---|---|
| m1 | Abstract, 1 | "Code, protocol and all results files are released" is not yet true; the two large files (forecasts, per-episode outcomes) are not in the repository. | Use future/conditional wording; describe the large files honestly. |
| m2 | 5 | Constants typed by hand in prose (21/4/5 days, 168 h, 90%, hour 20, 31 days) break the paper's own rule that every number comes from a results file. "31 days" also contradicts the dataset card (30 days; the data span 742 hours, 22 of which are discarded). | Generate these from `config.py` as macros; state the 742-hour span and the 22 discarded hours. |
| m3 | Abstract, 3, tables | PID and CI never expanded; sPL scaling undefined; MASE$_h$ undefined in the Table 2 caption. | Define at first use. |
| m4 | 2 | Related work omits throttling at scale (Dai et al., 2023) and feedback-control bidding with PID (Zhang et al., 2016); the PID baseline is uncited. The description of Lee et al. (2013) goes beyond what was verified. | Add the two verified references; narrow the Lee et al. sentence. |
| m5 | 1 | "A widely deployed mechanism" is not directly supported by the cited sources. | "A mechanism used in production systems [LinkedIn, Yahoo]". |
| m6 | 5 | The protocol paragraph reads as if design was untouched by data. Choices such as the 90% activity threshold were informed by a descriptive profile of the full dataset (including test hours), and naive/LightGBM/100-step TFT accuracy on test origins was inspected in smoke checks before the final runs. Neither is in the deviation log. | Add both to the paper's protocol paragraph and as a late-logged protocol entry. |
| m7 | 7 | The PID baseline's PVI of 0.43–0.60 suggests an under-damped controller. A better-designed feedback controller might narrow the gap. | Say so in the limitations. |
| m8 | 8 | Conclusion: "neural forecasters are the most accurate and give the best pacing" overstates M2/M3. | Align with the revised results. |
| m9 | 5 | Data licence (CC BY-NC-SA 4.0) not mentioned. | Add a clause. |

## 4. Cross-section checks

| Check | Status | Notes |
|---|---|---|
| Title matches content | Partly | Title asks about *better* forecasting; the abstract's lead claim answers a different question (M1). |
| Abstract reflects findings | Partly | Omits that the neural advantage over seasonal naive is much smaller than over even pacing. |
| Intro contributions vs conclusion | Aligned after M1/M5/M6 fixes | |
| Every table/figure referenced | Yes | |
| Numbers vs results files | Yes (all `\res{}` keys resolve); typed config constants violate the stated rule (m2) | |
| Citations | Verified for the main pacing and NHITS/neuralforecast entries; standard references still to spot-check | |

## 5. Verdict

**Major Revision.** The experiments are sound and the negative findings are valuable. The problems are
claim scope (M1, M3, M5, M6), missing uncertainty (M2), and one undisclosed validity threat (M4). All of
them can be repaired by re-scoping the text and surfacing numbers the results files already contain;
two (seed-aware inference, lower quantiles) can only be fully closed with new experiments, which are
outside this revision's authorization.

## 6. Revision instructions (this round)

1. Fix everything that needs no new experiment: M1, M2 (reporting and hedging), M3 (reporting and hedging),
   M4, M5 (hedging), M6, m1–m9.
2. Add only macros that re-express existing results or config constants (listed in the response log).
3. Do not change experiments, forecasts, simulations or any existing macro value.
4. Items that need new analysis go to the response log as "acknowledged limitation / proposed next experiment".
