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

## Why v2

An audit on 2026-10-16 found that the v1 results were not valid:
- The v1 data was a synthetic weekly template built from a label-encoded CTR dataset (iPinYou_x1) with no
  price or timestamp fields.
- Ablation results were constant multiples of the full model's numbers.
- Figures 1 and 5 were generated from formulas.
- The MASE computation compared different campaigns.
- The test budget leaked into the features.
