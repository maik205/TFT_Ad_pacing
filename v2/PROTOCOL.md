# Study Protocol v2 — Forecast-Driven Daily Budget Pacing

Status: FROZEN 2026-10-16 (later changes only via the Section 10 deviation log)
Date: 2026-10-16

All results produced before this protocol (branch `master`, `data/`, `paper/`) are
considered invalid and are not reused. See "Why v2" at the end.

## 1. Research question

Does probabilistic multi-horizon spend forecasting improve daily budget pacing over
forecast-free controllers on real campaign traffic, and does the forecast model
(seasonal naive vs. gradient boosting vs. TFT) or the quantile used for hedging
matter for the outcome?

No hypothesis is assumed to hold. A negative or mixed result is reported as such.

### Paper type

This is an applied-science paper. Forecasting models (TFT, LightGBM) and the throttling-based
pacing model are established methods. They are used as cited tools, not re-derived or
re-proven. There are no theoretical guarantees and no architecture-novelty claims.

Contributions are empirical and practical:
1. A reproducible pipeline that couples hourly probabilistic spend forecasts with
   throttling-based daily pacing on real campaign logs.
2. A controlled comparison of forecast-free controllers and forecast-driven policies under
   several budget tightness levels.
3. Practical guidance on when forecasting pays off, which forecaster to use, and how much
   quantile hedging to apply. This includes compute cost and failure cases.

## 2. Data

- Source: Criteo Attribution Modeling for Bidding dataset (Diemert et al., AdKDD 2017),
  Hugging Face `criteo/criteo-attribution-dataset`, CC BY-NC-SA 4.0.
- 16,468,027 impressions, 675 campaigns, 742 hours (timestamps relative to first impression).
- Per impression: timestamp, campaign, cost (paid price, transformed by Criteo), click,
  conversion, attribution.
- Known limitations (to be stated in the paper):
  - `cost` is a disclosed-as-transformed price, not raw currency.
  - Timestamp 0 is not local midnight; "hour of day" and "day" are phases relative to
    dataset start.
  - The data is a sub-sample of live traffic.
  - Logs were produced by Criteo's own bidder; counterfactual spend under a different
    policy is approximated by throttling (Section 4).

## 3. Splits and campaign selection

| Split | Hours | Days |
|---|---|---|
| Train | 0–503 | 0–20 |
| Validation | 504–599 | 21–24 |
| Test | 600–719 | 25–29 |
| Discarded | 720–741 | partial day 30 |

- Campaign inclusion uses **training hours only**: a campaign has non-zero spend in at least 90%
  of training hours. No test information is used for selection.
- All scalers and static features are computed on training hours only.
- The test split is evaluated **once**, after all model and controller choices are frozen
  on validation.

## 4. Pacing environment (impression replay with probabilistic throttling)

- Episode = one campaign-day (24 h). Daily budget B = ρ × (mean daily training spend of
  that campaign), with ρ ∈ {0.3, 0.5, 0.7}.
- At the start of each hour the policy sets a participation rate α_t ∈ [0, 1]. Each logged
  impression in that hour is kept independently with probability α_t; it contributes its
  logged cost, click and conversion. Serving stops when the budget is exhausted.
- 10 random replay seeds per episode.
- Assumption (stated as a limitation): throttling does not change the price or value of
  the impressions that are kept. This is the standard throttling-based pacing model
  (cf. Agarwal et al. 2014; Xu et al. 2015). No first-price vs. second-price claims are
  made.

## 5. Policies (all share the same environment)

1. **No pacing**: α = 1 until the budget is exhausted.
2. **Even pacing**: target cumulative spend B·h/24, proportional correction.
3. **PID**: gains tuned by grid search on validation episodes.
4. **Forecast-based throttling**: α_t = min(1, R_t / Σ_{remaining hours of day} ŷ_q),
   re-planned every hour from a fresh forecast, with forecasters:
   - a. seasonal naive (mean of same hour on previous 7 days)
   - b. LightGBM, direct multi-horizon (horizon as feature), quantile objective
   - c. TFT (Lim et al. 2021), via Nixtla `neuralforecast`
   - d. NHITS via `neuralforecast` (a second neural reference model, so TFT-specific
     conclusions are not drawn from a single neural model)
   - Each with q ∈ {0.5, 0.7, 0.9} (hedging ablation).
5. **Oracle (upper bound)**: knows the day's true demand D; constant α* = min(1, B/D).

## 6. Forecasting setup

- Target: hourly campaign spend (cost sum). Lookback 168 h, horizon 24 h, rolling origin
  every hour.
- Features: past spend, impressions, clicks; known future hour-of-day and day-of-week
  phase; static campaign statistics from training hours.
- Model selection and early stopping on validation only. Neural models: 3 seeds.

## 7. Metrics

Forecasting (validation and test, per campaign, then aggregated):
- MASE scaled by the **in-sample** (training) seasonal-naive MAE of the same campaign, computed per series.
- Scaled pinball loss for q ∈ {0.1, 0.5, 0.9}.
- 80% interval coverage and quantile-crossing rate.

Pacing (per episode, then aggregated):
- **Primary**: clicks captured / oracle clicks (also conversions / oracle conversions).
- Budget utilization rate (spend / B).
- Early depletion rate (budget exhausted before hour 20).
- Pacing volatility: RMS of Δα_t.
- Hourly spend deviation from the oracle profile.

Statistics: mean with 95% paired bootstrap CI over campaign-days, Wilcoxon signed-rank for
the main comparisons, and Holm correction for multiple comparisons.

## 8. Integrity rules

- Every number in the paper comes from a results file written by code in `v2/`.
  Tables are generated as LaTeX from these files; no hand-typed numbers.
- Every run writes its config, seed, git commit hash and timestamp next to its results.
- Metrics have unit tests against hand-computed values before use.
- Figures are rendered only from saved model outputs. There are no fallback or
  illustrative data paths.
- Any deviation from this protocol is logged in Section 10 with its reason, before the
  affected results are produced.

## 9. Decisions

- 2026-10-16: The TFT implementation is `neuralforecast`. There are no architecture-component
  ablations; the ablations are policy-level (forecaster × hedging quantile × budget tightness).
- 2026-10-16: This is an applied-science paper, so v1 theory (the solvency lemma, the spike
  detector, and FPA/SPA unification) is dropped rather than re-proven.
- 2026-10-16: The v1 material moves to `legacy/` with a README stating that it is invalid.
- Also report wall-clock training and inference cost per forecaster, as practical guidance.

## 10. Deviation log

All entries below were logged on 2026-10-16, before any test-split pacing result was produced.

1. **Reference policy wording.** The "Oracle (upper bound)" in Section 5.5 is a *hindsight
   constant-rate reference*. It is not an upper bound: a time-varying policy can capture
   more clicks than a constant rate. It is the denominator of the primary metric.
2. **Bootstrap unit.** Confidence intervals resample **campaigns** (cluster bootstrap), not
   campaign-days, because days of the same campaign are correlated. The paired Wilcoxon
   tests use per-campaign click ratios.
3. **Last test origin for neural models.** `neuralforecast` cross-validation needs 24 future
   hours per origin, so it stops at origin 718. For hour 719 (the last hour of the last test
   day), the origin-718 forecast shifted by one step is used.
4. **Quantile 0.7.** All forecasters output q ∈ {0.1, 0.5, 0.7, 0.9}, so that the 0.7 hedging
   level uses a fitted quantile rather than an interpolated one.
5. **Hyperparameters fixed a priori (no tuning on test).**
   - LightGBM: lr 0.05, 63 leaves, early stopping on validation targets; one run.
   - TFT: hidden size 64, dropout 0.1.
   - NHITS: library defaults.
   - Both neural models: input 168, max 3000 steps, early stopping on validation hours
     504–599 (patience 10 checks × 50 steps), robust scaler, 3 seeds.
   - Neural models are trained on CPU.
6. **Forecast-free controllers' initial rate.** Even pacing and PID start each day at
   α₀ = min(1, B / mean daily training spend), which equals min(1, ρ). This is information they
   legitimately have.
7. **Smoke tests.** Pipeline smoke tests (for example TFT with 100 steps) are deleted and
   never reported.
8. **Test-day exposure in the smoke run.** On 2026-10-16, an end-to-end smoke test simulated
   10 of the 414 campaigns on test days to check that the scripts run. Its outputs were
   deleted. No design choice was changed because of them.
9. **PID grid widened.** The same smoke run's *validation* tuning picked kp = 8, the upper
   edge of the grid, so the kp grid was widened to {0.5, 1, 2, 4, 8, 16, 32} before the full
   run. This decision used validation data only.
10. **PID grid widened again.** In the full validation tuning, the optimum was still on edges:
    kp = 32 at ρ = 0.3, and kp = 0.5 with kd = 2 at ρ = 0.5 and 0.7. The validation surface is
    flat (top-4 within 0.5%). The grid was widened to kp ∈ {1/32, 1/8, 0.5, 1, …, 128},
    ki ∈ {0, 0.05, 0.2, 0.5, 1} and kd ∈ {0, 0.5, 2, 8, 32}, and tuning was re-run before the
    test simulation. This decision used validation data only. If an optimum is still on an
    edge, it is reported as such and not widened further. (Result: the optimum stayed on an edge,
    kd = 32 and ki = 1, and is reported as such.)
11. **Neural training hardware.** Neural models are trained on one NVIDIA RTX 3070 Ti (CUDA),
    not on CPU as stated in entry 5. All six neural runs (TFT and NHITS × 3 seeds) use this
    GPU. The CPU run of TFT seed 0, started on the Mac, was aborted before it finished and its
    output is discarded. All other steps (data, naive, LightGBM, PID, simulation, summary) run
    on the Mac. Forecast files are copied to the Mac unchanged.
12. **GPU faults and retries.** Between 22:51 and 23:00 on 2026-10-16, four asynchronous TFT
    attempts on the GPU host ended in CUDA "unspecified launch failure" errors with driver
    events (nvlddmkm Event 153; the Windows GPU timeout was disabled, so these were not
    timeout resets). The first attempt ran while ComfyUI was using the same GPU.
    Diagnostics, all with deleted outputs:
    - Matmul and 45 s memory-bandwidth stress tests passed, with zero integrity errors.
    - NHITS passed.
    - TFT passed with synchronous launches, with per-module sync points, and with the core
      clock capped at 1700 MHz.
    - After 23:00, eight consecutive stock-clock asynchronous TFT runs passed.

    The cause is undetermined. It may be an intermittent hardware fault or residual driver state
    after the ComfyUI shutdown. Production runs use default asynchronous execution with
    ComfyUI stopped. A crashed run (no output written) is re-run, up to 3 attempts, and every
    attempt is logged in `logs/progress.log`. A crash produces no result, so retries do not
    select between outcomes. Logs of the failed attempts are in `logs/failed_gpu_attempt*`.
    Outcome: all six production runs (23:21–23:35) succeeded on their first attempt.
13. **Neural training step counts not recorded.** Because of an instrumentation bug, every
    `train_steps` field is 0. No run log contains Lightning's "max_steps reached" message,
    which it prints when the cap is hit (as seen in the smoke test), so all runs ended by early
    stopping before 3000 steps. Wall times (fit and predict) are recorded: TFT 192–230 s,
    NHITS 23–60 s. This was decided before any test metric was computed: the runs are not
    repeated just to record step counts, because a re-run would create a second set of
    forecasts to choose from.
14. **Presentation assets written after the results.** `scripts/07_paper_assets.py` (figures,
    compact tables and `numbers.tex` macros) was written after the test results existed. It only
    re-expresses the saved summaries. The one exception is per-step MASE (Figure 3), a
    breakdown of the pre-specified MASE by forecast step, computed from the saved forecasts.
    It adds no new comparison, metric or test. Figure layout was revised for legibility only
    (trade-off plot: log y-axis, label placement).
15. **Disclosure added after the results (revision round 1, logged 2026-10-17).** The log above was
    incomplete about what was looked at before the final runs. No design decision or hyperparameter
    was changed because of any of it.
    - *Descriptive profile of the full dataset.* Before the protocol was frozen, `profile.py`
      summarized the whole 742-hour dataset, including the test hours: campaign counts by
      activity share, concentration of spend, and lag-24 and lag-168 autocorrelations. The 90%
      activity threshold was chosen after seeing it. The inclusion rule itself uses training hours only.
    - *Test-origin accuracy before the final runs.* While smoke-testing, MASE on the test origins was
      computed for seasonal naive (0.917), LightGBM seed 0 (0.853) and a 100-step TFT smoke run
      (0.902, outputs deleted), plus an alignment check that shifted forecasts by one hour. The first
      two are the final values, since those runs are deterministic or were rerun unchanged.
    - *Pacing smoke test* on ten campaigns' test days: already logged in entry 8.
16. **Revision round 1 (logged 2026-10-17).** A self-review of the draft led to text and presentation
    changes only. No experiment, forecast, simulation or existing macro value changed. Details are
    in `paper/revision/`.
    - New macros in `scripts/07_paper_assets.py`: design constants read from `config.py` (so the
      prose no longer types them), training-seed SDs of the click ratio (already in
      `pacing_summary.csv`), the share of the 12 forecaster-by-budget combinations with monotone
      decline in the hedging quantile, the signed-comparison count, and per-campaign MASE win
      rates computed from `forecast_metrics_per_campaign.parquet`. The win rates are a descriptive
      re-aggregation of saved per-campaign metrics, not a new test.
    - Not done, because each needs new experiments or analysis: confidence intervals or tests on
      the MASE differences, seed-aware inference for the pacing comparisons, quantiles below 0.5,
      a forecast of the daily total, and a second dataset. The paper states these as limitations.
17. **Exploratory group analysis and independent rerun (logged 2026-10-19).** Two additions after the main
    results were final. No existing number changed.
    - *Group analysis (exploratory).* `scripts/08_groups.py` splits the evaluated campaigns into terciles of
      size (mean daily training spend) and volatility (coefficient of variation of daily training spend) and
      reports the click ratio by group and budget level, with paired differences between forecasters and
      campaign-bootstrap intervals. It was not pre-specified, was added after the main results were seen, and
      its intervals are not corrected for the 18 group-and-budget cells (plus the three all-campaign rows).
      Output: `results/pacing_groups.json`; new macros `grp-*` (29) and the table `tables/groups.tex` come from
      `07_paper_assets.py`.
    - *Independent rerun.* The pipeline was rerun on a second machine (Windows PC): data rebuilt from the raw
      file, PID gains retuned, seasonal-naive and LightGBM forecasts regenerated, simulation and summary rerun.
      The saved TFT and NHITS forecasts were reused (SHA-256 match `LARGE_FILES_SHA256.txt`); neural training
      was not repeated. The only code change was a serial episode loop in `simulate.py` in place of the
      Mac-only `fork` pool. Result: all 48 rows of `pacing_summary.csv` and the PID gains are identical, the
      forecast metrics are identical, and the paired p-values agree to within 1e-15. Of the 465 existing macros
      one differs, the LightGBM wall time (62 s against 76 s), which depends on the machine; the original
      value is kept. The group results were computed from the rerun's episode file.
18. **Post-hoc analyses: spend versus efficiency, and scaled seasonal naive (logged 2026-10-19, rule written
    before the second run).** Both were added after the main results were seen and after an external review
    raised the question. Neither changes an existing number.
    - *Spend versus efficiency (descriptive).* For each pair of methods, the click gain is split into the
      change in budget spent and the change in clicks per unit of spend, from `pacing_test_episodes.parquet`
      with `scripts/09_spend_efficiency.py` (ratio of totals, forecast seeds averaged, campaign bootstrap). No new simulation.
    - *Scaled seasonal naive (one new simulation).* Question: does a single scale on the seasonal-naive
      forecast total close the gap to the learned forecasters? Policies: the throttle of Section 5 with the
      seasonal-naive median forecast multiplied by a scale s in {0.80, 0.825, ..., 1.10} (13 values); and the
      throttle with the mean of the previous 7 days in place of the median (the Section 5 wording; the
      main study used the median, which was not logged). The scale is chosen per budget level on the
      validation days 21 to 24 (3 replay seeds, highest click ratio), as the PID gains were; the test days are
      not used for the choice. The test days are then simulated once for all scales (10 replay seeds, the
      same uniform draws as the main run), which also gives the curve of click ratio against s. TFT and NHITS
      forecasts exist only for the test origins, so they cannot be tuned the same way and enter unchanged.
      Reading rule, fixed before the run, per budget level, on the click ratio against the hindsight
      reference with campaign-bootstrap intervals: let g be the gain of TFT over the unscaled naive and c the
      gain of the tuned-scale naive over the unscaled naive. If c is at least g/2 and the interval of
      (TFT minus tuned naive) contains zero at two of three budget levels, the finding is "the total
      explains most of the gain". If the interval of c contains zero, or c is below g/4 at two of three
      levels, the finding is "scaling does not help; the gain needs more than the total". Anything else is
      reported as partial. A scale at the edge of the grid is reported as such. Output:
      `scripts/10_scaled_naive.py`, `results/scaled_naive.json`.
      Result of the run (logged 2026-10-19): chosen scales 0.85, 0.82 and 0.80 for rho 0.3, 0.5 and 0.7 (the
      last at the grid edge); gains over the unscaled naive 0.73, 1.03 and 1.22 points against 2.03, 1.73 and
      1.32 for TFT; the interval of TFT minus tuned naive excludes zero at rho 0.3 and 0.5 and includes it at
      0.7. Neither the "explains most" nor the "does not help" condition holds: reported as partial.
    - *Spend-matched comparison (added after that result, exploratory).* Because scaling mainly changes the
      amount spent, `scripts/11_frontier.py` reads the curve that the 13 scales trace in (spend, clicks) and
      reports how far each method lies above that curve at its own spend (linear interpolation; points of the
      hindsight reference; campaign bootstrap, resamples whose spend falls outside the curve are dropped and
      counted). It is not pre-specified and uses the test days, so it is descriptive. Output:
      `results/frontier.json`.
    - *Forecast plus feedback (one new simulation; rule fixed before the run).* Question: does adding a
      proportional-integral-derivative (PID) correction to the forecast throttle help, and does the neural
      edge survive it? Policy: the Section 5 throttle (median forecast, optionally times a scale), plus
      `kp*e + ki*sum(e) + kd*(e - e_prev)` added to the rate and clipped to [0, 1], where `e` is the share of
      the daily budget by which spend so far trails the day's plan (`min(B, D)` spread in proportion to the
      forecast demand profile issued at hour 0; `D` the forecast daily demand). With all gains zero it is the
      existing throttle (checked on one episode in the script). The gains come from the grid kp in {0, 2, 8, 32},
      ki in {0, 0.5}, kd in {0, 4}, chosen per budget level on the validation days (3 replay seeds, highest
      click ratio), separately for two seasonal-naive bases: the plain median, and the median times the
      scale chosen in the previous bullet's run (0.85, 0.825, 0.80). TFT and NHITS forecasts exist only for the
      test origins, so they use the gains tuned on the plain naive base; this transfer is a limitation. The test
      days are simulated once (10 replay seeds, same draws as before). Reading rule, per budget level, on the click
      ratio with campaign-bootstrap intervals: (1) feedback helps a forecast if its interval for
      (with feedback minus without) is above zero at two of three budget levels, for TFT; it does not help if the
      interval includes zero or is negative at two of three. (2) The neural edge survives if
      (TFT with feedback minus scaled naive with feedback) is above zero at two of three levels. A gain at the
      grid edge is reported as such. Output: `scripts/12_feedback.py`, `results/feedback.json`.
      Result of the run (logged 2026-10-19): chosen proportional gain 8 (tight), 32 (medium, grid edge) and 8
      (loose) for the plain base, and 8, 8 and 0 for the scaled base; integral and derivative gains were zero in
      every case. Feedback minus no feedback, in points of the hindsight reference: TFT +0.50 [0.23, 0.79],
      -0.97 [-1.68, -0.41] and -2.01 [-2.79, -1.43] at the tight, medium and loose budgets. By rule (1) feedback
      does not help TFT (negative at two of three levels). By rule (2) the neural edge does not survive
      (TFT with feedback minus scaled naive with feedback: +0.26 [-0.02, 0.59], +0.38 [-0.31, 1.42], -1.92
      [-2.75, -1.26]); at the loose budget the scaled base chose zero gains, so the comparison there is
      between unequal designs. The validation gain of feedback for plain naive at the loose budget (+1.0
      points) did not carry to the test days (-0.8, interval [-2.50, 0.17]), which shows that gains chosen on four
      validation days are unstable.
    - *Coverage of the summed hedge plans (descriptive, no simulation).* The throttle divides the remaining
      budget by the sum of the hourly q-quantile forecasts over the rest of the day. That sum is not the
      q-quantile of the day's remaining demand. `scripts/13_hedge_coverage.py` measures, on the test days, how
      often the realised rest-of-day spend falls at or below each plan's total (the effective quantile), by
      forecaster and hour of the day, and the factor on the median plan that would give the nominal coverage.
      It reads the saved forecasts and the hourly spend only; it changes no policy and no existing number.
      Output: `results/hedge_coverage.json`.
      Result (logged 2026-10-19): the summed 90% plan covered 94.0% (seasonal naive) to 97.4% (NHITS) of
      rest-of-day totals and was 1.80 to 2.06 times the median plan, where 1.62 to 1.70 would give 90%; at 70%
      the coverage was 76% to 80%. The hourly quantiles themselves were calibrated (TFT 90.8% at q=0.9).
    - *Calibrated-total hedge (one new simulation; rule fixed before the run).* Question: how much of the
      hedging cost is the summing of hourly quantiles? Policy: the throttle with the median plan times a factor
      f(q, hour of day) in place of the sum of q-quantile forecasts, q in {0.7, 0.9}, for every saved forecast
      (seasonal naive, LightGBM, TFT, NHITS; each seed separately). f is the q-quantile, over campaigns and
      the calibration days, of realised rest-of-day spend divided by the median plan, per origin hour.
      Two variants. *Causal (primary):* f for test day d uses all campaigns on the earlier test days only, so it
      is evaluated on days 26 to 29 (day 25 has no earlier day); comparison policies are restricted to the
      same days. *Cross-fitted (secondary, an upper bound, because it sees same-day shocks of the other campaigns):*
      campaigns are split into two random halves (seed 0) and f for one half comes from the other half over all
      five test days. Metrics as in the main study (click ratio against the hindsight reference, spend, clicks per
      spend; campaign bootstrap). Reading rule on the primary variant, TFT at q=0.9: the hedging cost is mostly
      the summing if (calibrated minus summed) has an interval above zero at two of three budget levels and
      recovers at least a third of the summed plan's loss against the median plan; it is inherent if that
      interval includes zero at two of three levels. Anything else is reported as partial. The secondary
      variant and the other forecasters are reported, not used for the verdict. Output:
      `scripts/14_calibrated_hedge.py`, `results/calibrated_hedge.json`.
      Result (logged 2026-10-19; the first attempt crashed in the analysis step on the hindsight-reference
      rows after the simulation had finished, the script was fixed to drop them and to reuse the saved episodes;
      no policy or number changed). Primary variant, days 26 to 29, TFT at q=0.9: calibrated minus summed plan
      +1.54 [0.81, 2.46], +1.21 [0.42, 2.15] and +0.95 [0.10, 2.00] points at the tight, medium and loose
      budgets, all above zero; but they recover only about 20%, 14% and 11% of the summed plan's loss against the
      median plan (-7.71, -8.61, -9.02), below the third the rule required. By the rule: partial. At q=0.7 the
      calibrated plan recovers 85%, 55% and 37% of the loss (+1.27, +1.07, +0.80), and at the tight budget costs
      only -0.23 [-0.41, -0.05] against the median plan. For seasonal naive the calibrated 90% plan is worse than
      the summed one (-1.9, -2.9, -3.6 points); the 70% plan is better at two of three budgets. The cross-fitted
      variant (upper bound) is no better for TFT at q=0.9 (+0.04, -0.16, -0.28 on days 26 to 29, intervals
      include zero).
    - *Wording corrections found during the revision (no analysis).* (i) The paper called the forecast quantiles
      "calibrated"; the nominal 80% interval covers 61% to 77% of outcomes, so the word was removed. (ii) The
      protocol of Section 5 defines seasonal naive as the mean of the same hour on the previous 7 days, but the
      study ran empirical quantiles of those 7 days, the median at q=0.5; the mean version, run in the
      scaled-naive analysis, gives 0.7 to 0.95 points fewer clicks, so the median is the stronger baseline.
      (iii) The paired tests use 409 campaigns because they need positive hindsight clicks; the other
      analyses use all 414.
    - *Paper revision (logged 2026-10-19).* `paper/main.tex` was rewritten around the question of what a forecast
      adds: new title, abstract, introduction, a setup paragraph for the scaled, calibrated and feedback variants,
      and Sections 6.3 to 6.5 (where the gain comes from, hedging, feedback). The group table, the horizon figure
      and the standalone conversions paragraph were removed from the paper (the group result is kept as one
      sentence; the figure and table files are still generated by `07_paper_assets.py`). Every new number is a
      macro from a new block of `07_paper_assets.py` that reads the six new results files
      (`spend_efficiency`, `scaled_naive`, `frontier`, `feedback`, `hedge_coverage`, `calibrated_hedge`);
      the table `tables/decomp.tex` comes from the same block. The new macros were merged into
      `tables/numbers.tex` without changing any existing line. The related-work section cites one preprint
      (arXiv:2606.10187) from its abstract only; it was not read in full.
19. **Second batch of post-hoc experiments, after an external review (logged 2026-10-19, rules written before
    the run).** An independent review of the revised paper (a model-written review, not a human one) found that the
    scale grid stops at its best value, that the "edge at matched spend" may come from per-campaign daily totals
    and not from hourly shape, and that no headroom was measured. `scripts/15_extended.py` runs three
    things on the test days with 10 replay seeds and the same draws as before. It adds no tuning on test data.
    - *Extended scale grid.* The scaled seasonal-naive throttle of entry 18 with 25 scales from 0.50 to 1.10 in
      steps of 0.025, scale chosen per budget level on the validation days (3 replay seeds) as before; the
      first run's grid (0.80 to 1.10) is superseded. Reading rule, per budget level, on the click ratio against the
      hindsight reference with campaign-bootstrap intervals: let g be the gain of TFT over the unscaled naive
      and c that of the tuned-scale naive. If c is at least 0.8 g at all three levels, the finding is "the total
      explains almost all of the gain". If the interval of (TFT minus tuned naive) is above zero at two or three
      levels, the finding is "an edge beyond one scale remains". If neither, partial. A chosen scale at the
      edge of the grid is reported as such.
    - *Total and shape swap.* Because the throttle uses only the remaining forecast sum, shape is defined from the
      forecast issued at hour 0 of the day: the planned remaining demand at hour h is the day total times the share
      of the day-start forecast that lies in hours h to 23, with no re-planning. Four static plans: naive total with
      naive shape, TFT total with TFT shape, TFT total with naive shape, naive total with TFT shape (TFT: the three
      seeds, averaged). With T the gain of (TFT total, naive shape) and S that of (naive total, TFT shape), both over
      (naive total, naive shape): the total is the source of the TFT gain if T is at least twice S and the interval
      of S includes zero; shape matters if the interval of S is above zero and S is at least half of T. Otherwise
      partial. The static plans are compared with the re-planned throttle of the main study to show what
      re-planning adds.
    - *Oracle remaining demand.* The throttle with the true remaining demand of the day in place of the forecast
      (an oracle forecast), and the throttle with the true day total times the naive or TFT day-start shape.
      Reading: headroom is the click ratio of the oracle minus that of TFT; under 1 point means that forecast
      accuracy has little left to buy in this simulator, 2 points or more that it matters. The oracle total with a
      forecast shape separates the value of the total from the value of the shape.
      Output: `scripts/15_extended.py`, `results/extended.json`.
    - *Results (logged 2026-10-19).* Extended grid: chosen scales 0.85, 0.825 and 0.775 at the tight, medium and
      loose budgets, none at the grid edge; on the test days the curve peaks at 0.75 to 0.775 at all three. The
      tuned scale gains 0.73 [0.55, 0.93], 1.03 [0.82, 1.25] and 1.24 [0.96, 1.53] points over the plain naive plan,
      that is 36%, 60% and 94% of the TFT gain (2.03, 1.73 and 1.32). TFT minus tuned naive is 1.30 [0.93, 1.77],
      0.69 [0.39, 1.03] and 0.08 [-0.19, 0.37]. By the rule: 0.8 g is reached only at the loose budget, so not
      "the total explains almost all"; the interval is above zero at two of three levels, so the finding is "an
      edge beyond one scale remains".
      Swap, with G the gain of (TFT total, TFT shape) over (naive total, naive shape): G is 1.52, 1.35 and 1.07;
      T, the gain of (TFT total, naive shape), is 1.25 [0.82, 1.83], 1.02 [0.71, 1.38] and 0.72 [0.50, 0.95]; S, the
      gain of (naive total, TFT shape), is 0.16 [-0.02, 0.31], 0.17 [0.05, 0.29] and 0.25 [0.14, 0.37]. By the rule:
      at the tight budget T is above twice S and the interval of S includes zero, so the total is the source; at the
      medium and loose budgets the interval of S is above zero but S is below half of T, so partial (the total
      supplies 82%, 76% and 68% of G, the shape a small positive amount). Re-planning adds 0.51, 0.37 and 0.25 points
      to TFT (static minus re-planned: -0.51 [-0.66, -0.37], -0.37 [-0.52, -0.24], -0.25 [-0.36, -0.15]) and nothing
      to seasonal naive, as expected since its forecast does not depend on the origin hour.
      Oracle: true remaining demand minus TFT is 1.03 [0.73, 1.35], 1.50 [1.23, 1.78] and 1.54 [1.27, 1.81] points,
      between the two thresholds of the rule (under 1 point, 2 points or more), so reported as moderate headroom.
      True remaining demand minus seasonal naive is 3.06, 3.22 and 2.86. The true day total with the naive shape
      gains 1.81, 1.84 and 1.44 over the static naive plan; the true total with the TFT shape gains 0.63, 0.86 and
      0.75 over the static TFT plan; true remaining demand beats the true total with the TFT shape by 0.91, 1.00 and
      1.04.
    - *Frontier recomputed (not pre-registered).* `scripts/11_frontier.py` now reads the 25-scale episodes, whose curve
      covers TFT's spend at the tight budget (the 13-scale curve did not). TFT lies above the curve by 1.20 [0.91,
      1.56], 0.74 [0.51, 1.01] and 0.47 [0.30, 0.66] points at the tight, medium and loose budgets; NHITS by 1.31,
      0.83 and 0.44; LightGBM by 0.45 [0.13, 0.77], 0.01 and 0.05; PID lies below. This supersedes the 13-scale
      output of entry 18.
    - *Corrections to entry 18.* (a) The cross-fitted calibrated hedge is not an upper bound on clicks. Its factors
      are larger than the causal ones and it spends less, and in this simulator any factor above 1 lowers clicks; the
      label "upper bound" is withdrawn. (b) Two metrics listed in Section 7 are not reported in the paper: the hourly
      spend deviation from the oracle profile (computed, in `pacing_summary.csv`) and the forecast metrics on the
      validation split.

## Why v2

An audit on 2026-10-16 found that the v1 results were not valid:
- The v1 data was a synthetic weekly template built from a label-encoded CTR dataset (iPinYou_x1) with no
  price or timestamp fields.
- Ablation results were constant multiples of the full model's numbers.
- Figures 1 and 5 were generated from formulas.
- The MASE computation compared different campaigns.
- The test budget leaked into the features.
