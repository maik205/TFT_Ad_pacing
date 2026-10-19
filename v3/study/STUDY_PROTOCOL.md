# Study Protocol: Audience-Aware Budget Pacing

Status: **FROZEN 2026-10-18** (version 1.0; author approved the Section 13 decisions). Any later change goes
to the deviation log (Section 12) with date and reason; nothing is changed silently. Section 9 (iPinYou arm)
is completed in version 1.1 before any iPinYou run.

Companion documents: `v3/synthetic/GENERATOR_SPEC.md` (frozen 2026-10-17), the literature-review
protocol in `v3/research/phase1_protocol/protocol.md`.

## 1. Question

When the ad platform knows **who is likely to arrive and when**, and how well each user matches each
campaign at that hour, can it spend each campaign's budget better than a controller that only watches
spend?

- **RQ1 (pacing).** Does setting the pacing multipliers from a forecast of the remaining day's audience
  beat reactive multipliers, holding the click predictions fixed?
- **RQ2 (value).** Do learned user vectors, and hour-modulated affinity `e_u · W(h) · e_c` on top, improve
  allocation over campaign- and context-level predictions?
- **RQ3 (attention).** Does an estimate of the user's attention state reduce wasted spend without losing
  value?
- **RQ4 (recovery).** Do the learned vectors recover the planted structure (archetypes, time zones,
  hour-dependent interest)?
- **RQ5 (evaluation).** How far do off-policy estimates from the logs deviate from the exact value?
- **RQ6 (conditions).** Under which world conditions (generator knobs) do the gains of RQ1–RQ3 appear or
  vanish?

RQ1–RQ6 are answered on synthetic worlds. The iPinYou arm (Section 9) asks RQ1–RQ2 at segment level on
real logs.

## 2. Problem

- An **opportunity** is an ad slot `s = (u, t, context, format, duration)`. The platform shows one campaign
  or none.
- Campaign c has value per click `v_c` and a daily budget `B_c` per advertiser-local day (budget column
  `budget_C{k}_rho{rho}` of `synth_campaigns`, for cost model k and tightness rho).
- **Objective:** maximise delivered value `sum v_c · pCTR_true(u, c, t)` over the test days, subject to
  realised spend `<= B_c` per campaign-day. Value is counted in expectation (oracle pCTR); spend is
  realised (Section 6), so budgets bind as they would in production.
- A hard stop applies to every method: a campaign that has spent its daily budget is ineligible for the
  rest of that local day.

## 3. Decision rule shared by all methods

For each opportunity, among eligible campaigns:

```
score_c = v_c · p̂(u, c, t) · (1 - â_s) - lambda_c · ĉ(u, c, t)
show argmax_c score_c if it is > 0, else show nothing
```

- `p̂` is the method's click prediction; `â_s` its estimate that the user is absent (0 for methods
  without an attention model); `ĉ` the expected cost under the run's cost model (known exactly for C1,
  C4; `v_c · p̂` for C2; predicted for C5, C6).
- `lambda_c >= 0` is the per-campaign pacing multiplier, updated every pacing step (6 minutes,
  `pacing_steps_per_hour` = 10).
- **C3 (second price)** uses the auction form instead: bid `b_c = v_c · p̂ · (1 - â_s) / (1 + lambda_c)`;
  the top bid wins if it is at least the external bid `E_s`; cost = `max(E_s, second-highest bid)`. C3 is
  reported as a secondary result (generator spec Section 17, item 4).

Methods differ only in how `p̂`, `â_s` and `lambda_c` are produced. This makes every difference
attributable.

## 4. Method ladder

| Id | `p̂` (value) | `lambda_c` (pacing) | `â_s` | Answers |
|---|---|---|---|---|
| P0 | none: even throttle, as the logging policy | — | 0 | floor |
| P1 | campaign × context × local hour (aggregate) | reactive | 0 | standard baseline |
| P2 | + static user vectors `e_u · e_c` | reactive | 0 | RQ2 |
| P3 | + hour-modulated `e_u · W(h) · e_c` | reactive | 0 | RQ2 |
| P4a | as P3 | forecast, aggregate pool weights | 0 | RQ1 |
| P4 | as P3 | forecast, user-level pool weights `a_u(h)` | 0 | RQ1 |
| P5 | as P4 | as P4 | HMM filter | RQ3 |
| O-P1, O-P4 | **true** pCTR | as P1 / P4 | 0 | pacing without prediction error |
| UB | true pCTR | hindsight-optimal (LP per campaign-day with truth) | true | upper bound |

True pCTR already contains the attention term `s(state)`, so the O rows and UB know attention
implicitly; their cost side still pays for absent impressions they choose to show.

### 4.1 Reactive multiplier (P1–P3)

Dual descent on the budget constraint, per campaign, every pacing step k:
`lambda_c <- max(0, lambda_c + eta · (spend_c,k - target_c,k) / target_c,k)`, where `target_c,k` is the
budget share planned for step k by the aggregate traffic curve of the campaign's local day (from training
days). `eta` is tuned (Section 7).

### 4.2 Forecast-driven multiplier (P4a, P4)

At each pacing step, `lambda_c` is set so that **projected spend for the rest of the local day equals
the remaining budget**:

1. **Scenario pool.** Opportunities seen in the same hours-of-week over the previous 7 days (data the
   policy has observed; no future information).
2. **Weights.** P4a: every pool opportunity has weight 1 (aggregate forecast). P4: each opportunity is
   reweighted by the user's predicted presence, `a_u(h) / a_u_hist(h)`, so that users who churned or
   changed rhythm count less and the recent active user base counts more.
   The weights are then rescaled for each hour of the pool so that the weighted pool carries the expected
   traffic of the projected hour, arrivals included (deviation 4); without this step the weights remove
   departing users but add no new ones, and the projection falls short.
3. **Projection.** Every pool opportunity is scored with the current `p̂` and the decision rule; the
   projected spend of c is the weighted sum of its costs where it wins.
4. **Solve.** Find the joint `lambda` by a few rounds of per-campaign bisection (others fixed) until
   projected spend of each campaign is within 2% of its remaining budget, or `lambda_c` = 0 if the budget
   cannot be spent.

This is where personalization reaches pacing: a games campaign keeps `lambda` high in the afternoon
because the model projects that night-time gamers, its best matches, are still to come.

### 4.3 Click and audience model (P2–P5)

One multi-task model, trained on training days 0–20 (Section 7), embedding dimension d = 16:

- **User vector `e_u`**, campaign vector `e_c`, and `W(h) = W_0 + sum_j (cos(2πjh/24) W_j + sin(2πjh/24)
  W'_j)`, j = 1..2, with `h` the user's **local** hour. P2 uses `W(h) = W_0`.
- **Local time.** `h` uses the time zone of the observed region (public `region_tz`); for users with a
  missing or inconsistent region, the zone is inferred as the offset that best aligns the user's visit
  times with the population rhythm, re-estimated daily.
- **Heads and losses** (summed, weights tuned):
  1. **Context:** predict the slot's context category from `(e_u, W(h))` — softmax over 12 categories.
     This is the dense signal that teaches user vectors (every slot, not only clicks).
  2. **Activity `a_u(h, daytype)`:** Poisson loss on visit counts per user-hour.
  3. **Engagement:** interaction (logistic), watch fraction (in-feed, Beta mixture likelihood),
     skip (pre-roll), dwell (log-normal).
  4. **Click:** logistic, `logit = b + e_u · W(h) · e_c + q_c + delta · match + fatigue term`.
     Clicks are sparse per user (about 0.2 per user in two weeks at 1% CTR); they mainly place the 40
     campaign vectors. Unweighted loss; an IPS-weighted variant is a sensitivity run.
- **Cold users** (joined after training) start from the population mean vector and are updated online
  from their context stream; no clicks needed.
- **P1 aggregate model:** logistic on campaign × context category × local-hour block, same click data.

### 4.4 Attention filter (P5)

A 3-state hidden Markov model per visit (engaged / casual / absent, absent absorbing), with emission
families per format for the observed signals (Section 7a.2 of the generator spec gives the families, not
the parameters). Parameters are fitted by EM on training days. At decision time the **forward filter over
the visit's earlier slots only** gives `â_s`. No ground-truth state is used in fitting or tuning.

## 5. Worlds and runs

- **Main grid:** generator spec Section 13, one-at-a-time around the default, 3 world seeds each.
- **Headline worlds:** default (gamma = 1.0) and **gamma_real** (1.011). The calibration put gamma_real
  almost exactly at the default (MI excess 0.0249 vs 0.0255 bits at seed 0), so the default world is
  realistic in hour-interest strength. Gains are quoted at gamma_real; gamma = 2.0 is labelled
  optimistic, 0.5 pessimistic.
- **Negative controls:** gamma = 0; gamma = 0 with a single zone.
- **Model seeds:** 3 per world seed. A cell is one (world config, world seed, model seed, cost model, rho).
- Cost models: C1, C4, C5, C6 in the main tables (C2 in an appendix, deviation 2); C3 secondary.

## 6. Evaluation harness (exact, synthetic)

- Replays the **test-day opportunity stream** (days 24–27, from `synth_opportunities`) in time order
  through a policy, online. Opportunities do not depend on the ads shown, so the stream is the same for
  every policy.
- The policy sees only observed data: its own model trained on days 0–20 (and, for online updates, the
  test-day signals of impressions it has already served, drawn as below).
- **Outcomes** of each shown ad (click, engagement signals, realised cost) are drawn with **common random
  numbers** keyed by (opportunity, campaign), so two policies that make the same choice get the same
  outcome. Fatigue `n_uc` is tracked under the policy.
- **Scoring** uses the oracle (`synth_truth.oracle`): expected value, expected clicks, true state of every
  impression. The harness is evaluation code and may import the oracle; policies may not (spec rule R2,
  enforced by the isolation test).
- **Harness validation (must pass first):** replaying the logging policy on training days reproduces the
  logged total spend within 0.5% and per-campaign-day spend no further from the logs than two runs of the
  rule with different seeds are from each other (median and p90 at most 1.25x that noise); the logged click
  count within sampling error (deviation 1).

## 7. Tuning and splits

- Train days 0–20; **validation days 21–23** for all tuning; **test days 24–27** touched once per frozen
  configuration.
- Equal tuning budget: every method gets **20 random-search configurations** on the validation days of
  the default world, seed 0, scored on the primary metric. The chosen configuration is then used on every
  world without re-tuning (the knob grid tests robustness, not re-tuned performance).
- Campaign-days straddling a split boundary are dropped and counted (spec Section 12).

## 8. Metrics and statistics

**Primary:** delivered value relative to UB, as a ratio of totals over complete campaign-days: `sum value(method) / sum value(UB)` (deviation 3; the per-campaign-day average of the ratio is reported as secondary).
Gains are reported as the paired difference to P1.

**Secondary:** expected clicks; budget utilisation; share of campaign-days that hit the hard stop before
the last local hour (early depletion); smoothness (mean absolute deviation of cumulative spend from the
campaign's own planned curve); **wasted spend** (share of spend on absent impressions, from truth);
cold-user value share.

**Recovery (RQ4):** linear probe from `e_u` to true archetype weights (R² on held-out users); time-zone
accuracy for users with missing region; for office-heavy users, the rank of enterprise-software
campaigns at 10:00 vs 22:00; HMM state accuracy (AUROC of `â_s` for absent).

**Off-policy (RQ5):** IPS, SNIPS and DR estimates of each method's test-day value from the logs, with
the method's decisions computed under its exact-run `lambda` trajectory; report bias and 95% interval
coverage against the exact value.

**Statistics:** hierarchical bootstrap (world seed, then campaign-day; 2,000 resamples) for 95% CIs on
paired differences. No significance stars; effect sizes with CIs.

### 8.1 Decision gates (fixed before any test run)

- **G1, negative control.** At gamma = 0, the P3 − P2 value difference must have a CI that includes 0
  and |difference| < 0.5% of UB. Otherwise nothing is reported until the leak is found.
- **G2, go/no-go on pacing (run first, oracle predictions).** O-P4 − O-P1 at gamma_real, default other
  knobs. If the gain is below **1% of UB** with a CI including 0, the paper is reframed around *when*
  audience-aware pacing helps (RQ6), and the learned-model work is scoped down. The author decides.
- **G3, harness validation** (Section 6) passes before G2.

## 9. iPinYou arm (segment level; to be completed before any iPinYou run)

- The "user" side is a vector built from the 43 profile tags plus region and hour (users do not return,
  generator spec Section 17).
- Replay of logged won impressions with their paying price as cost; budgets at rho of logged spend.
- Counterfactual limit stated up front: only won impressions are logged, so allocations that would have
  shown a different campaign cannot be scored exactly. Candidate design: per-advertiser replay with
  bid-landscape assumptions, reported as indicative.
- Details (split by day, budget rule, cost model) go into version 0.2 of this protocol.

## 10. Integrity

- Policy and model code lives in `v3/study/`; it may read `observed/` only (generator spec R2; the
  isolation test is extended to this folder).
- All runs write config hash, world hash and code hash; results tables are built only from result files.
- Every reported number on synthetic data is labelled synthetic (spec R1).

## 11. Order of work

1. Generator finished (deviation 3, gamma_real and control worlds, grid).
2. This protocol reviewed and frozen.
3. Harness + logging-policy replay (G3).
4. Oracle runs O-P1, O-P4, UB at gamma_real and default (G2).
5. Model P1–P5, tuning on validation, recovery checks.
6. Test runs over the grid; negative controls (G1).
7. OPE comparison.
8. iPinYou arm (after protocol 0.2).
9. In parallel: literature review Phase 2 (novelty of forecast-driven multipliers from user-level audience
   forecasts).

## 12. Deviation log

| # | Date | Section | Change | Reason |
|---|---|---|---|---|
| 1 | 2026-10-18 | 6, 8.1 | G3 part B: "logged spend per campaign-day within 0.5%" → total spend within 0.5% and per-campaign-day differences at most 1.25x the seed-to-seed noise of the logging rule (run without hard stop, training days) | Measured after the freeze: two runs of the logging rule with different seeds differ per campaign-day by a median 8.2% and p90 26.7% (who shows up in a 6-minute step is random; small budgets overshoot within a step), so a 0.5% per-campaign-day bound cannot be met by any harness. Harness vs logs: total +0.06% (second seed −0.09%), median 8.2%, p90 28.8%, i.e. median ratio 1.00, p90 ratio 1.08. Approved by the author 2026-10-18 |
| 2 | 2026-10-18 | 8.1, 9 | C2 (CPC) is left out of gate G2 and of the main tables; its results go in an appendix, marked degenerate. Main tables use C1, C4, C5, C6 | Under C2 the cost of an impression is `v_c · p` and its value is also `v_c · p`, so value equals spend. Every impression has the same value per unit of cost, so the hindsight optimum is "spend the whole budget" and the dual bound does not converge (dual gap 28–83%, against about 0.1% for C1, C4, C5, C6; measured on 6 gamma_real worlds). Ratios to UB exceed 1 (O-P1 0.94–1.04, O-P4 1.02–1.09), and O-P4 spends 103–106% of budget (hard stop lets a step overshoot), which raises its value one for one. The C2 gain cannot measure pacing quality. Approved by the author 2026-10-18 |
| 3 | 2026-10-18 | 8 | Primary metric: mean over campaign-days of `value / value(UB)` → ratio of totals, `sum value(method) / sum value(UB)` over complete campaign-days, per world; gains are paired differences pooled over worlds, with a hierarchical bootstrap (world, then campaign-day, 2,000 resamples) | Measured after the freeze: UB gives some campaign-days almost no value, so single-day ratios reach 76 and the mean of ratios is dominated by those days and exceeds 1. The ratio of totals weights each campaign-day by its UB value and stays below 1 for C1, C4, C5, C6. Approved by the author 2026-10-18 |
| 4 | 2026-10-19 | 4.2, 8.1 | The user-level pool weights of P4 and O-P4 (presence ratio `a_u(h) / a_u_hist(h)`) are rescaled per pool hour and lag so that the weighted pool carries the expected traffic of the projected hour. O-P4 uses the true expected traffic; P4 uses its own forecast of it. The earlier form (O-P4 without rescaling) is kept as a labelled variant | Measured after the freeze: presence weights remove users who leave but add none who join, so the projected traffic of the day was 0.52–0.54 of the actual (mean lifetime 5 days) and 0.84–0.85 (14 days); after rescaling 0.996–1.010. With a 5-day lifetime the old weights scored 0.6–0.9 points of UB below the unweighted pool in all four cost models; rescaled they match it (−0.05 to +0.06 points, intervals about ±0.25). At gamma_real the rescaled O-P4 − O-P1 is +2.2, +3.0, +2.5, +1.6 points (C1, C4, C5, C6; 6 worlds) against +2.1, +3.0, +2.4, +1.4 before, so the G2 verdict does not change. At gamma_real, the rescaled pool (aggregate or user-level) adds +0.1 to +0.3 points over the unweighted pool, with intervals that include 0 in all four models; user-level composition on top of the rescaled total adds at most +0.03 points. Approved by the author 2026-10-19 |

## 13. Decisions (made before the freeze; kept as the record)

1. **Objective:** ~~value (`v_c` × clicks) as primary, clicks secondary — or the reverse?~~
   **Decided 2026-10-18 (author): value primary, clicks secondary.** Value decides tuning and gate G2;
   both are reported.
2. **G2 threshold:** 1% of UB at gamma_real. **Decided 2026-10-18 (author): yes.**
3. **Tuning budget:** 20 configurations per method on validation, no re-tuning per world.
   **Decided 2026-10-18 (author): yes.**
4. **P4 scenario pool:** previous 7 days at the same hours-of-week, reweighted by `a_u(h)`.
   **Decided 2026-10-18 (author): 7-day pool** (not a learned arrival model alone).
