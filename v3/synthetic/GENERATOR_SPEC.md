# Synthetic Ad-Platform Generator — Specification

Status: **FROZEN 2026-10-17** (author approved the pre-freeze review). Any later change goes to the
deviation log in Section 16 with date and reason; nothing is changed silently.
Date: 2026-10-17

## 0. Purpose and integrity rules

The generator produces a **synthetic** ad-platform log with planted, known structure: users whose
interests over named categories change with their *local* time of day, many campaigns competing for
the same users, and budgets. Its only job is to answer questions that real logs cannot:

1. Does a model with learned user and campaign vectors and time-modulated affinity **recover the
   planted structure** (archetypes, time zones, user x campaign x hour affinity)?
2. Under which world conditions (Section 13 knobs) does audience-aware pacing beat aggregate pacing,
   and where does the advantage vanish?
3. How far do off-policy estimates computed from the logged data deviate from the exact value?

Rules:

- **R1. Labelled as synthetic everywhere.** File names start with `synth_`; every table and figure that
  uses it says "synthetic"; the paper draws no conclusion about real traffic from it.
- **R2. Ground truth is physically separate.** Observable data goes to `out/observed/`; true parameters
  go to `out/truth/`. Model and policy code may not read `out/truth/` or import the ground-truth oracle
  module. A test enforces this (Section 15).
- **R3. Written before generation.** All parameters, knob grids and calibration targets below are fixed
  before the first world is generated. Calibration targets come from real *training* data only.
- **R4. Seeded and reproducible.** A world is fully determined by (config hash, world seed). The config
  and the code version are written into every output's metadata.
- **R5. Negative controls are mandatory.** Results are reported together with the gamma = 0 world
  (Section 13), where personalization has no signal to find.

## 1. Time

- Horizon: **28 days** = 672 hours. All logged timestamps are **UTC** (seconds since world start; world
  start is UTC midnight, a Monday).
- Each user lives on a **local clock**: `local_hour = (utc_hour + tz_offset_u) mod 24`, with the local
  weekday derived the same way. Behaviour is defined only in local time.
- No daylight-saving shifts; offsets are fixed for the whole horizon.
- Local **time blocks** used for interests: night 00–06, morning 06–09, work 09–17, evening 17–20,
  late 20–24. Day types: weekday (Mon–Fri local), weekend (Sat–Sun local).

## 2. Categories (named)

| id | Category |
|---|---|
| 0 | enterprise_software |
| 1 | office_equipment |
| 2 | video_games |
| 3 | gaming_hardware |
| 4 | streaming_media |
| 5 | fashion |
| 6 | beauty |
| 7 | groceries |
| 8 | home_family |
| 9 | travel |
| 10 | personal_finance |
| 11 | education |

## 3. Time zones

- User home zone drawn from a categorical distribution over UTC offsets. Default mix:
  UTC+7 40%, UTC+8 20%, UTC+1 20%, UTC-5 20%.
- **Observed region feature.** The platform sees `region_obs`, a region code that implies a zone:
  - correct for a share `1 - p_missing - p_wrong` of users,
  - missing (null) for `p_missing` = **0.20**,
  - wrong (a random other region) for `p_wrong` = **0.05** (travel, VPN, bad geo-IP).
  Several region codes map to the same zone, so region is a realistic categorical, not the offset itself.
- Campaign day boundaries follow the **advertiser's zone** (Section 6), drawn from the same mix.

## 4. Archetypes (planted semantics)

An archetype `a` defines two things, both in local time.

### 4.1 Activity rhythm `r_a(h, daytype)`

A sum of von Mises bumps over the 24-hour circle,
`r(h) = b + sum_j w_j * exp(k_j * (cos(2*pi*(h - mu_j)/24) - 1))`, normalised to mean 1 over the day.
The `- 1` makes each bump peak at height `w_j` whatever its concentration `k_j`, so `w` sets the height
and `k` only the width. Default bumps (peak hour mu, concentration k, weight w):

| Archetype | Weekday bumps (mu, k, w) | Weekend bumps (mu, k, w) |
|---|---|---|
| office_worker | (12.5, 3, 1.0), (21, 4, 0.8) | (11, 2, 0.6), (21, 3, 1.0) |
| night_gamer | (23, 4, 1.2), (14, 2, 0.3) | (22, 3, 1.2), (15, 2, 0.6) |
| student | (14, 2, 0.6), (22, 3, 1.0) | (16, 1.5, 1.0), (23, 3, 0.8) |
| parent | (7, 5, 0.7), (20.5, 4, 1.0) | (10, 2, 0.9), (20, 3, 0.7) |
| shift_worker | (4, 3, 0.8), (16, 3, 0.8) | (13, 1.5, 0.8), (21, 2, 0.6) |

Base level `b` = 0.05 for all.

### 4.2 Interest profile `I_a(k, block, daytype)`

Log-affinity per category per local time block. Default: 0 everywhere except the entries below
(weekday / weekend share the table unless marked):

| Archetype | night 00–06 | morning 06–09 | work 09–17 | evening 17–20 | late 20–24 |
|---|---|---|---|---|---|
| office_worker | — | personal_finance 1.0 | enterprise_software 2.0, office_equipment 1.5 (weekday); travel 1.0 (weekend) | groceries 0.8 | video_games 1.5, streaming_media 1.2 |
| night_gamer | video_games 2.0, gaming_hardware 1.5 | — | streaming_media 0.8 | video_games 1.0 | video_games 2.0, gaming_hardware 1.8 |
| student | streaming_media 1.0 | education 1.2 | education 1.5 (weekday); fashion 1.2 (weekend) | fashion 1.0 | video_games 1.5, beauty 0.8 |
| parent | — | groceries 1.5 | home_family 1.0 | groceries 1.2, home_family 1.2 | travel 1.2, personal_finance 0.8 |
| shift_worker | groceries 1.0, streaming_media 1.0 | streaming_media 0.8 | — | video_games 0.8 | home_family 0.8 |

Block boundaries are smoothed: the value at hour h is a 1-hour-wide linear blend between adjacent
blocks, so interests do not jump at block edges.

## 5. Users

N_users default **50,000**. For each user u:

- **Archetype mixture** `w_u ~ Dirichlet(alpha * pi)`, with archetype prevalence
  pi = (office 0.30, gamer 0.15, student 0.20, parent 0.20, shift 0.15) and alpha = **2.0**
  (users are mixtures; most have one dominant type).
- **Personal interest noise** `eps_uk ~ N(0, 0.3^2)` per category, constant over time.
- **Interest** `interest_u(k, t) = sum_a w_ua * I_a(k, block(local_hour), daytype) + eps_uk + drift_uk(t)`.
- **Drift** `drift_uk` is a Gaussian random walk per day with step sd **0.05**, started at 0.
- **Rhythm** `r_u(h, daytype) = sum_a w_ua * r_a(h, daytype)`.
- **Activity level** `lambda_u ~ LogNormal(log 3, 1.0)` visits per active day (heavy tail).
- **Lifetime**: join day uniform on [0, 28) for a share **0.3** of users (the rest are present from
  day 0); leave after `Exp(mean = L)` days, L default **14**. Joining and leaving shift the audience mix.
- **Time zone** and `region_obs` as in Section 3.

## 6. Campaigns

N_campaigns default **40**.

- **Category**: 3–4 campaigns per category (round-robin, then random fill); 20% of campaigns get a
  second category with half weight.
- **Advertiser zone** from the Section 3 mix; the campaign's budget day is 00:00–24:00 in that zone.
- **Value per click** `v_c ~ LogNormal(mu_cat, 0.4)` with mu_cat higher for enterprise_software,
  personal_finance and travel (+0.7) and lower for groceries and beauty (-0.3). Units are abstract
  "value units"; no currency is implied.
- **Creative quality** `q_c ~ N(0, 0.3^2)` added to the CTR logit.
- **Daily budget** `B_c = rho * S_c`, where `S_c` is campaign c's mean daily spend under the logging
  policy run **without budgets** on the training days. rho in {0.3, 0.5, 0.7}.
- All campaigns run all 28 days (start/end dates are a later knob, off by default).

## 7. Opportunities

- **Visits.** In each UTC hour, user u (if alive) visits `Poisson(lambda_u * r_u(local_hour, daytype) / 24)` times.
- **Slots.** Each visit produces `1 + Poisson(0.5)` ad slots while the user is attentive; an absent
  visit keeps producing autoplay slots (Section 7a).
- **Format.** Each slot is a **video** slot with probability `p_video` = **0.4**, otherwise a **display**
  slot. A video slot is **in-feed** with probability `p_feed` = **0.6**, otherwise **pre-roll**.
  - *Pre-roll:* duration D is 6, 15 or 30 s with probabilities 0.3 / 0.4 / 0.3; 15 s and 30 s videos are
    skippable after 5 s, 6 s videos are not.
  - *In-feed* (short-video feed; the user can swipe away at any moment): D ~ LogNormal fitted to KuaiRand
    AD durations (median about 131 s on KuaiRand-Pure). An absent user does not swipe, so the video loops.
- **Context.** Each slot has a page-context category drawn from
  `0.7 * softmax(gamma * interest_u(., t)) + 0.3 * uniform` — what the user is browsing reflects current
  interest. The same gamma as in Section 8 scales it, so with gamma = 0 the context is uniform and carries
  no hour signal (negative control, Section 15 test 4).
- Every alive campaign with remaining budget is **eligible** for every slot (no targeting rules by default).
- The platform's decision per slot: choose one campaign or show none.

## 7a. Attention state and engagement signals

Clicks are rare, so every impression also yields **engagement signals**. They are generated from a
hidden **attention state** per visit, which models are never shown.

### 7a.1 Hidden state

States: **engaged**, **casual**, **absent**.

- **Start of a visit.** `P(engaged) : P(casual) = sigmoid(eta0 + eta1 * max_k interest_u(k, t))` against
  its complement; visits never start absent. Defaults eta0 = **-0.5**, eta1 = **0.8**.
- **Drift into absence.** After each slot an attentive visit becomes absent with probability
  `h_abs(local_hour) = h0 * m(local_hour)`, h0 = **0.01** (deviation 1; was 0.03), m = **3** for local
  hours 00–05 and 1 otherwise.
  Absent is absorbing for the rest of the visit.
- **Absent visits burn impressions.** Once absent, the visit continues as autoplay for
  `Geometric(mean = 8)` further slots, then ends. These slots cost money and deliver nothing.
- **Engaged to casual.** After each slot an engaged visit becomes casual with probability **0.1**.

### 7a.2 Observed signals per impression

Let `x = sigmoid(gamma * sum_k m_ck * interest_u(k, t))` be the planted match between user and campaign.

| Signal | Engaged | Casual | Absent |
|---|---|---|---|
| Skip at 5 s (skippable video) | prob `sigmoid(0.5 - 3x)` | prob `sigmoid(1.5 - 2x)` | 0 (nobody presses) |
| If not skipped: watched to the end | 0.85, else stops at Uniform(5 s, D) | 0.5, else Uniform(5 s, D) | 1.0 (autoplay) |
| Post-ad dwell, seconds | LogNormal(log 20, 1.0) | LogNormal(log 8, 1.0) | LogNormal(log 300, 0.8) |
| Interaction (tap, scroll, volume) | 0.6 | 0.2 | **0.03** |
| Muted or app backgrounded during ad | 0.05 | 0.25 | 0.5 |

The skip and watch-to-end rows above apply to **pre-roll**. **In-feed** video replaces them with a
watch fraction `f = play_time / D`:

| State | Watch fraction f |
|---|---|
| Engaged | with prob `sigmoid(e0 + 3x)` a long watch `Beta(a_L, b_L)`, else a quick swipe `Beta(a_S, b_S)` |
| Casual | with prob `sigmoid(c0 + 2x)` a long watch `Beta(a_L, b_L)`, else a quick swipe `Beta(a_S, b_S)` |
| Absent | the video loops: f = 1 plus the loop time until the visit ends |

`(e0, c0, a_L, b_L, a_S, b_S)` are fitted once, before generation, so that the in-feed watch fraction
under the logging policy matches KuaiRand AD views: quantiles p10/p25/p50/p75/p90 =
0.00 / 0.01 / 0.02 / 0.13 / 0.70 and long-view rate 24% (KuaiRand-Pure). A **completed view** for
in-feed uses KuaiRand's `long_view` rule: play >= D if D <= 18 s, else play >= 18 s.

Display slots record post-ad dwell, interaction and backgrounded only.

**Planted ambiguity** (so no single threshold recovers the state): with probability
`a_amb` = **0.10**, an engaged impression draws its dwell from the absent distribution (a long stay,
e.g. reading the landing page) and keeps its own interaction rate; and absent impressions keep a 3%
interaction rate. Long dwell therefore means "absent" only together with no interaction.

### 7a.3 Effect on clicks and costs

- The click model (Section 8) gains a state term.
- Under C6 (Section 9), absent users "complete" views, so wasted spend is directly visible.
- Evaluation metric **wasted spend**: the share of each campaign's spend on impressions whose true state
  is absent (computed from ground truth only).

## 8. Ground-truth click model

```
logit pCTR(u, c, t) = beta0
                      + gamma * sum_k m_ck * interest_u(k, t)      # time-modulated affinity
                      + q_c                                          # creative quality
                      + delta * 1[context(slot) in categories(c)]    # page-context match
                      - phi * log(1 + n_uc(t))                       # fatigue
                      + s(state)                                     # attention (Section 7a)
```

- `s(engaged)` = 0, `s(casual)` = **-1.0**, `s(absent)` = **-5.0** (an absent user almost never clicks).

- `m_ck` = campaign category weights (1, or 1 and 0.5 for a second category).
- `n_uc(t)` = impressions of c shown to u in the preceding 24 hours.
- Defaults: gamma = **1.0**, delta = **0.5**, phi = **0.4**.
- `beta0` is solved numerically per world so that the **mean CTR under the logging policy** equals the
  target CTR (Section 14; default **1%**).
- A click is drawn as Bernoulli(pCTR). Conversions are not modelled (later extension; the code keeps a
  hook for a second stage).

## 9. Cost models

Every impression gets a cost under **each** of the six models; experiments pick one per run and the
paper reports all six.

| id | Name | Cost of showing campaign c in slot s |
|---|---|---|
| C1 | Flat CPM | `m_c` per impression, `m_c = 0.01 * v_c * mean CTR` (campaign-level constant) |
| C2 | CPC | `v_c` if clicked, else 0 (cost known only after the click) |
| C3 | Second price | `max(E_s, second-highest internal score)`, where `E_s` is an external competitor bid and internal scores are the policy's own bids; the policy-dependence is recorded |
| C4 | Time-of-day market | `m_c * g(local_hour_u)`, `g` = the synthetic population's activity curve over local hour (mean of `r_u` over users), normalised to mean 1 (prime time costs more) |
| C5 | Logged-like | `LogNormal` per impression, parameters fitted to the real paying-price distribution (Section 14), scaled to mean `m_c` |
| C6 | Cost per completed view | Video: `m_c / r_bar` if watched to the end, else 0, with `r_bar` the population completion rate under the logging policy (so C6 and C1 have the same mean). Display: C1 |

`E_s ~ LogNormal(log(m_bar * g(local_hour_u)), 0.5)`, with `m_bar` the mean of `m_c`.

## 10. Logging policy (produces the training data)

An epsilon-mixture with recorded propensities:

- With probability 1 - epsilon: a heuristic score `v_c * (1 + delta_ctx * 1[context match])`, softmax
  with temperature tau over eligible campaigns, plus a "no ad" option with fixed score; budgets paced
  evenly by a per-campaign throttle.
- With probability epsilon: uniform over eligible campaigns plus "no ad".
- Defaults: epsilon = **0.10**, tau = **1.0**, delta_ctx = **1.0**.
- Recorded per impression: chosen action and its **propensity** (exact probability under the mixture).

The logging policy uses **no** user-level knowledge, so the logs do not already encode the answer.

## 11. Outputs

### `out/observed/` (what models may use)

| File | Columns |
|---|---|
| `synth_opportunities.parquet` | opp_id, ts_utc, user_id, visit_id, region_obs, context_category, slot_index, format (display / preroll / infeed), video_duration_s |
| `synth_impressions.parquet` | opp_id, campaign_id (or null = no ad), propensity, cost_C1..cost_C6 |
| `synth_engagement.parquet` | opp_id, skipped, skip_time_s, watch_time_s, completed, post_dwell_s, interacted, muted_or_backgrounded |
| `synth_clicks.parquet` | opp_id, clicked |
| `synth_campaigns.parquet` | campaign_id, category names, advertiser_tz, value_per_click, budget per rho |
| `synth_meta.json` | config, config hash, world seed, code version, generation time |

### `out/truth/` (evaluation only)

| File | Content |
|---|---|
| `truth_users.parquet` | user_id, archetype weights, true tz offset, lambda_u, join/leave day, eps_uk |
| `truth_drift.npy` | drift_uk per day |
| `truth_states.parquet` | opp_id, attention state (engaged / casual / absent), ambiguity flag |
| `truth_world.json` | archetype tables, beta0, gamma, delta, phi, cost parameters |

The **oracle** module (`synth_truth.oracle`) computes the exact pCTR(u, c, t) and expected outcomes
of any policy from these files. Only evaluation code may import it.

## 12. Splits

- Train: UTC days 0–20. Validation: days 21–23. Test: days 24–27.
- A campaign-day belongs to the split that contains its whole advertiser-local day; campaign-days that
  straddle a split boundary are dropped from evaluation (the count is reported).
- Users joining after day 20 are cold-start users in validation/test; their count is reported.

## 13. Knobs and the experiment grid

Defaults in **bold**. Grid is one-at-a-time around the default, plus the negative control combos.

| Knob | Values | Purpose |
|---|---|---|
| gamma (time-modulated affinity strength) | 0, gamma_real, 0.5, **1.0**, 2.0 | gamma = 0 is the **negative control**: hour does not affect interest. gamma_real is solved (bisection on gamma, default world otherwise) so that the synthetic within-user MI excess equals the KuaiRand value 0.0255 bits, using the estimator defined in Section 14; values above it are optimistic and must be labelled so |
| tz mix | single zone (UTC+7), **default 4-zone**, uniform over 24 zones | tz diversity drives the gain from personalization |
| p_missing (region unseen) | 0, **0.2**, 0.6 | time-zone inference difficulty |
| mean lifetime L (churn) | infinite, **14 d**, 5 d | audience-mix shift |
| lambda_u median (sparsity) | 1, **3**, 10 visits/day | strength of per-user signal |
| target CTR | 0.1%, **1%**, 5% | click sparsity (0.1% approx. real display CTR) |
| rho (budget tightness) | 0.3, **0.5**, 0.7 | as in v2 |
| p_video (video share) | 0, **0.4**, 0.8 | 0 = display only: engagement signals reduced to dwell and interaction |
| p_feed (in-feed share of video) | 0, **0.6**, 1.0 | swipe-any-time feed vs. skippable pre-roll |
| h0 (drift into absence) | 0.003, **0.01**, 0.03 | how much budget idle sessions can burn; 0.03 is above what KuaiRand supports and is labelled a stress level (deviation 1) |
| a_amb (planted ambiguity) | 0, **0.10**, 0.25 | whether engagement heuristics still work when signals overlap |
| N_users, N_campaigns | **50k / 40**; 200k / 100 | scale |

- Negative controls: (gamma = 0) and (gamma = 0, single zone).
- **3 world seeds** per grid point; models trained with their own seeds on top.

## 14. Calibration targets

Measured **before generation** from real training data, recorded here with their source. Rows are of
three kinds: **fit** (the generator is tuned to match), **check** (computed on generated worlds and reported
next to the real value, never tuned) and **not a target** (differs by design). Fits: target CTR scale, C5
spread, in-feed watch fraction, gamma_real. The archetype rhythms stay design choices and are not fitted to
real traffic.

**MI estimator** (used for KuaiRand and for gamma_real): for each user with at least 50 events, the mutual
information in bits between local time block (the five blocks of Section 1) and category, from the
block × category count table; the view-weighted average over users; minus the same quantity after shuffling
blocks within each user (mean of 5 shuffles). On KuaiRand the events are views and the category is the
first-level video category; on synthetic worlds the events are slots and the category is the page-context
category.

| Target | Source | Value |
|---|---|---|
| **Check.** Population hourly traffic curve (shape) | iPinYou season-2 training impressions (12,237,087; 06–12 June 2013), local China time, share % by hour 0..23 | 8.33, 2.47, 1.33, 1.25, 0.65, 0.71, 0.90, 2.04, 2.91, 3.23, 5.26, 5.40, 5.43, 5.94, 5.36, 5.88, 5.87, 5.60, 6.08, 4.48, 4.92, 4.08, 6.94, 4.94. Won impressions, so shaped by the advertisers' bidding and budgets as well as by users; the hour-0 spike is probably daily budget reset (not verified). Use hours 1–23 for the shape; hour 0 interpolated. Reported: distance to the synthetic UTC+8-only population curve |
| **Not a target.** Impressions per user distribution | same | p50 1, p75 1, p90 2, p99 4 (10,146,411 distinct users; 0% missing IDs). Cookie churn; synthetic users persist by design |
| **Fit.** Paying-price distribution (C5) | same, paying price column (CNY fen per CPM) | mean 78.3; p10/25/50/75/90 = 19/38/70/92/166; LogNormal fit mu 4.052, sigma 0.865; no zero prices. C5 uses sigma 0.865, rescaled to mean `m_c` |
| **Fit (stress knob).** Real display CTR (for the 0.1% knob) | Zhang et al. 2014, Table 3: 0.075% over all training data; measured on season-2 training: 9,978 clicks / 12.24M impressions | 0.082% (per advertiser 0.055%–0.096%) |
| **Fit (gamma_real).** Within-user hour-category dependence | KuaiRand-1K, standard log 04-08..04-21, first-level categories, 5 local time blocks: MI excess over within-user shuffled-hour baseline | 0.0255 bits (all views), 0.0347 bits (long views) |
| **Check.** Rhythm stability | same, week 1 vs week 2 hourly profile correlation (users with >= 200 events in both weeks; synthetic: same rule on visits) | same user median 0.48, random user 0.15. Synthetic rhythms are fixed per user, so the synthetic value is driven by visit counts; reported, not tuned |
| **Check.** Ad views per user | same, AD views | median 12, p90 48 (2 weeks) |
| **Fit.** In-feed watch fraction and long-view rate (Section 7a.2) | KuaiRand-Pure AD views (fields verified 2026-10-17) | p10/25/50/75/90 = 0.00 / 0.01 / 0.02 / 0.13 / 0.70; long_view 24%; median duration 131 s. KuaiRand-1K for comparison: 0.00 / 0.007 / 0.031 / 0.278 / 0.833; long_view 31% |

The **planted effect sizes** (gamma, interest tables, archetype mix) are not calibrated to reality,
because the real values are unknown. That is why they are knobs, and why conclusions are stated
conditionally on them.

## 15. Integrity tests (must pass before any experiment)

1. **Isolation:** no module outside `synth_truth` and `evaluation` imports `synth_truth` or opens
   `out/truth/` (static import scan + path check).
2. **Determinism:** same config + seed gives byte-identical observed files.
3. **Planted-structure sanity:** in the generated log, with ground truth, the per-archetype local-hour
   activity and category click rates reproduce the Section 4 tables (within sampling error).
4. **Negative control sanity:** with gamma = 0, the true pCTR does not depend on local hour except via
   fatigue and the attention state (absence drift is higher at night), the context category distribution
   is the same at every local hour, and the MI excess of Section 14 is 0 within sampling error.
5. **Propensities:** logged propensities sum to 1 per slot over all actions and match the policy.
6. **Calibration:** mean logging-policy CTR equals the target within 2% relative.
7. **Engagement sanity:** with ground truth, the generated signal rates per state match the Section 7a.2
   table within sampling error, and absent impressions have a click rate below 1% of engaged ones
   (deviation 2; s(absent) = -5 implies about exp(-5) = 0.7%).
8. **Ambiguity check:** a fixed-threshold rule on dwell alone (no model) misclassifies at least `a_amb`
   of engaged long-dwell impressions, so the heuristic cannot be read off one signal.
9. **Fits hit their targets:** gamma_real world MI excess within 10% of 0.0255 bits; each in-feed
   watch-fraction target point (q_t, p_t) is met in either direction, i.e. the quantile at p_t is within
   0.02 of q_t **or** the share at or below q_t is within 2 points of p_t (Lévy distance <= 0.02;
   deviation 3), and long-view rate within 2 points of KuaiRand-Pure; C5 log-cost sd within
   2% of 0.865. The Section 14 checks are computed and written to `synth_meta.json`.

## 16. Deviation log

| # | Date | Section | Change | Reason |
|---|---|---|---|---|
| 0a | 2026-10-17 | 7 | Context category drawn from `softmax(gamma * interest)` (was `softmax(interest)`) | Pre-freeze review: the hour signal leaked into the gamma = 0 negative control through context × delta, and gamma_real could not be solved because context did not depend on gamma |
| 0b | 2026-10-17 | 13, 14 | MI estimator defined once in Section 14; gamma_real solved by bisection against it | Pre-freeze review: estimator was referenced but not defined |
| 0c | 2026-10-17 | 9, 14 | iPinYou traffic curve is a check, not a fit; C4 uses the synthetic population curve | Pre-freeze review: fitting would silently change the planted archetype rhythms |
| 0d | 2026-10-17 | 14 | Rows labelled fit / check / not a target; impressions per user marked not a target; rhythm stability reported as a check | Pre-freeze review |
| 0e | 2026-10-17 | 15 | Test 4 tightened (no context exception); test 9 added (fits hit targets) | Follows from 0a–0d |

Entries 0a–0e were agreed by the author before the freeze. Entries from 1 onward are changes after the
freeze.

| # | Date | Section | Change | Reason |
|---|---|---|---|---|
| 1 | 2026-10-17 | 7a.1, 13 | h0 default 0.03 → 0.01; grid {0.01, 0.03, 0.09} → {0.003, 0.01, 0.03} | Calibration smoke test (3,000 users, seed 0): at h0 = 0.03, 23% of logged in-feed impressions are absent (looping, watch fraction >= 1), so the KuaiRand p90 of 0.70 cannot be reached by any watch-shape parameters (best p90 1.86). At h0 = 0.01 (absent share 9.7%) the fit is near exact: p10..p90 0 / 0 / 0.016 / 0.125 / 0.700, long view 24.1% vs target 0.00 / 0.01 / 0.02 / 0.13 / 0.70, 24%. A structural alternative (absent in-feed ends the visit) fitted worse and was not adopted. Approved by the author 2026-10-17 |
| 2 | 2026-10-17 | 15 | Test 7 threshold "below 0.1% of engaged" → "below 1%" | The test text contradicted the frozen click model: s(absent) = -5 gives a click-rate ratio of about exp(-5) = 0.7% at low CTR, so the 0.1% test could never pass. The model is unchanged; only the test threshold is corrected |
| 3 | 2026-10-18 | 15 | Test 9 in-feed criterion: "quantiles within 0.02" → "each target point met by quantile (within 0.02) or by cumulative share (within 2 points)" | Default world (50k users, seed 0, 632,786 in-feed impressions): quantiles 0 / 0 / 0.017 / 0.126 / 0.653 vs target 0 / 0.01 / 0.02 / 0.13 / 0.70, so p90 failed by 0.027; but the share at or below 0.70 is 90.05% vs 90%. The p90 sits in a near-empty gap between quick swipes and full/looping views (quantile 0.46 at 89.5%, 1.01 at 90.5%), so quantile distance is unstable there; at the low end the curve is steep (47% of views at or below 0.01, and the targets are rounded to 2 decimals), so share distance is unstable there. The either-direction rule (Lévy distance) is stable at both ends. Changed after seeing the result; the model and fitted values are unchanged. Approved by the author 2026-10-18 |

## 17. Open items (all decided before the freeze; kept as the record)

1. **CTR default.** Real display CTR (~0.075%) yields few clicks per user, so per-user learning may be
   nearly impossible; the default is set to 1% with 0.1% as a stress knob. Confirm this choice.
2. **C3 detail.** Second-price cost depends on the policy's own bids; the exact bid definition per
   policy family must be fixed in the study protocol.
3. **Targeting.** Default is no targeting rules (all campaigns eligible). Region or category targeting
   could be added as a knob.
4. **Archetype tables.** The values in Section 4 are design choices meant to express the intended
   semantics (office worker vs. night gamer). Review before freezing.
5. **Calibration source.** Requires the raw iPinYou season-2 logs; if those are unavailable, the
   calibration rows are filled from the published statistics only and the gap is logged.
6. **Engagement parameters.** The Section 7a.2 table is a design choice. KuaiRand should calibrate the
   watch-time and completion shapes; it is recommendation data, not ads, so it calibrates how people
   watch, not how they respond to ads. Its fields must be verified first.

Decided 2026-10-17 (author agreed to the suggested resolutions):

- Item 1: default target CTR 1%, with 0.1% as a reported stress knob.
- Item 2: archetype tables accepted as drafted (Section 4), subject to the freeze review.
- Item 3: no targeting by default; region targeting is an optional knob.
- Item 4: C3 bid definitions go in the study protocol; C3 stays out of headline results until then.
- Item 5: obtain raw iPinYou season-2 logs next; fall back to published statistics and log the gap.
- Ad slots are a mix of video and display; C6 (cost per completed view) is added (Section 7a).
- Item 6: KuaiRand fields verified from the official page (2026-10-17). It has per-view watch time and
  video duration, an AD/NORMAL video type, local date and hour-minute, a 0.37% randomly exposed subset
  (`is_rand`), CC BY-SA 4.0. It has no post-video dwell field, no ad cost and no budgets; post-ad dwell
  can only be approximated from the gap to the user's next interaction. Proposed (awaiting author): calibrate
  the Section 7a watch-time and completion shapes from it; dwell parameters stay design choices.
  Checked on KuaiRand-Pure (2026-10-17, `~/datasets/kuairand` on the Mac, read-only):
  - AD views are about 1% of views (27,639 of 2.62M; 76 AD videos; 13,971 users saw at least one;
    median 1 AD view per user). Pure is too thin for per-user ad modelling; the 1K or 27K versions are needed.
  - `hourmin` is the hour bucket (HH00) in UTC+8 (matches `time_ms` + 8 h for 82.7% of rows; the rest
    differ by about an hour). Single time zone, so KuaiRand cannot test the multi-zone mechanism.
  - Category file: 32,038,725 rows, covers 100% of Pure videos, names in Chinese, 38 first-level
    categories (second level known for 86%, third 38%, fourth 7%). The labels come with a probability,
    so they are classifier outputs, not hand labels. AD videos: games 18, comedy 9, tech/digital 9,
    short drama 7, food 7, anime 6, beauty 4, other 16.
  - Watch behaviour is in-feed (swipe any time), not pre-roll: AD watch fraction quantiles
    p10/p25/p50/p75/p90 = 0.00/0.01/0.02/0.13/0.70, `long_view` 24%, median AD duration 131 s.
    `is_click` here means a valid play, not an ad click.
  - Consequence for Section 7a: the "engaged watches to the end 85%" assumption does not match in-feed
    ads. Decided 2026-10-17: an in-feed format was added (Sections 7, 7a.2), calibrated to these
    quantiles; pre-roll keeps its table.
  Checked on KuaiRand-1K (2026-10-17; standard log 04-08..04-21; 983 users, 5.06M views, median 3,489
  views and 14 active days per user):
  - Personal rhythms are stable: week-1 vs week-2 hourly profile correlation median 0.48 for the same
    user vs 0.15 for a random pair.
  - Rhythms do not fall into a few clean types: 5 clusters leave 74% of rhythm variance within clusters.
    Cluster shapes: midday peak 39%, early evening 21%, late evening 17%, night owls (23–01 h, 32% of
    views at 00–05) 15%, 20–22 h 8%. This supports continuous user vectors (the Dirichlet mixture of
    Section 5) over hard archetypes.
  - The hour predicts what a user watches, weakly but clearly: within-user mutual information between
    local time block and first-level category is 0.0443 bits vs 0.0188 for a shuffled-hour baseline
    (excess 0.0255); on long views 0.0989 vs 0.0642 (excess 0.0347). Caveats: views are chosen by
    Kuaishou's recommender, and platform-wide hour effects in content supply also count as within-user
    dependence here, so this is an upper bound on the user-specific effect.
  - Ads: 18,208 AD views (0.36%), 5,247 AD videos, median 12 AD views per user (p90 48). AD watch
    fraction p10..p90 = 0.00 / 0.007 / 0.031 / 0.278 / 0.833, long_view 31%, median duration 134 s.
  - Consequence: the default gamma = 1.0 may be much stronger than reality (not yet measured on a
    generated world). A gamma_real level matched to the KuaiRand excess was added to the Section 13 grid.
- Item 5: raw iPinYou season 2 obtained (Figshare, CC0; `~/datasets/ipinyou/season2.zip` on the Mac) and
  checked 2026-10-17 on the 7 training days (values in Section 14):
  - 12.24M impressions, 5 advertisers (one campaign each), all users have an ID.
  - Users almost never return: 10.1M distinct users, median 1 impression, p90 2. Only 5.65% of users are
    seen on 2 or more days (13% of impressions); day-to-next-day overlap is 3–4%. 8.2% of users are
    reached by 2 or more advertisers, 1.1% by 3 or more. The cookie ID churns, so iPinYou **cannot**
    support learned per-user vectors.
  - It does carry 43 named user-profile tags (`user.profile.tags.en.txt`, e.g. long-term interest
    news / education / automobile / IT / electronic game; in-market segments) on 86% of impressions, plus
    region and city codes. Decided 2026-10-17: on iPinYou, the "user" side is a vector built from the tag
    set (plus region, hour), i.e. segment-level personalization; per-user identity is tested only on the
    synthetic world (and rhythm stability on KuaiRand).
  - Copies of all datasets are on the PC (`C:\Users\maik\datasets`, sizes verified 2026-10-17); v3
    generation and experiments run on the PC.
  - Measured CTR 0.082%, consistent with the paper's 0.075%; this supports the 1% default plus 0.1%
    stress knob (Item 1), since per-user click learning at real CTR is not feasible.
  - Single time zone (China), so the multi-zone mechanism is tested on synthetic data only.

Known property of the logs (added 2026-10-18, found by the study harness check G3; no model change):

- The logging policy's throttle checks a campaign's spend once per pacing step (6 minutes), so it can
  overshoot the daily budget inside a step. Under C1 the logged mean budget utilisation is about 1.17.
  This affects the training logs only; the study harness enforces budgets exactly for every method (max
  measured overspend 0.16%, one impression). Worlds generated before this note carry the earlier spec hash.
