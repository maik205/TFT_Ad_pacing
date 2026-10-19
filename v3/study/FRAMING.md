# Framing note: deep forecasting for budget pacing

Date: 2026-10-19. Status: working note, not part of the frozen protocol. It changes no protocol rule; the protocol
(`STUDY_PROTOCOL.md`) and its deviation log stay the record. If any research question below is adopted as a
change to the protocol's Section 1, it goes into the deviation log first.

## Aim

The work started as "apply a Temporal Fusion Transformer (TFT) to ad pacing". After the first results it is
framed more broadly and more honestly:

> Does a learned forecast of the remaining day improve budget pacing over a controller that only watches
> spend, and over a plain replay of last week? Under which conditions? Is any particular architecture needed?

TFT, N-HiTS and LightGBM are candidate forecasters. No claim depends on TFT winning.

## What the evidence says so far

| Question | Answer | Where it comes from |
|---|---|---|
| Does looking ahead beat reacting to spend? | Yes, by about 1.2 to 3.0 points of the best possible result (flat, time-of-day, noisy and per-view prices; only time-of-day is clear on its own) | v3 synthetic worlds with a perfect forecaster (Protocol Section 8, G2); 6 worlds at the realistic setting, 3 at the default |
| Is the plain 7-day replay enough there? | Nearly. Scaling the replay to the true total traffic adds 0.1 to 0.3 points at the realistic setting (0.0 to 0.1 with a 5-day user lifetime), intervals include zero; knowing exactly who is present, on top of that, adds at most 0.03 points | v3 ablations, including 3 worlds with a 5-day user lifetime |
| Is real traffic as regular as the synthetic worlds? | No. Bid requests swing up to 2.5x day to day; single advertisers jump 3x or fall 70% in a day | iPinYou season 2 (7 days, 5 advertisers) |
| Do learned forecasters beat the replay on real data? | Yes, by 1.3 to 2.0 points of the hindsight benchmark's clicks. Hedging low costs 8 to 9 points | v2 Criteo study (414 campaigns, about 30 days hourly); results files read, not rerun |
| Does the Criteo gain depend on campaign size, volatility or budget tightness? | The gain of TFT over the replay is positive in every group: +1.0 to +3.1 points, every interval above zero. It shrinks as the budget loosens (+2.0, +1.7, +1.3 points). It does not depend on size. It is larger in the most volatile third at tight and medium budgets (+3.1 and +2.2 against +1.7 and +1.6 in the least volatile third), though the intervals overlap | `criteo_groups.py` on the reproduced v2 episodes; groups from training hours only, intervals resample campaigns, not corrected for the 18 group-and-budget cells; now also in the v2 paper as an exploratory subsection |
| Is TFT better than other deep forecasters? | No clear evidence. TFT ties N-HiTS overall (error 0.787 vs 0.794; seed spread 0.004 to 0.014; pacing within 0.3 points in every group). The one hint is the most volatile third of campaigns, where TFT is 0.2 to 0.4 points ahead at the medium and loose budgets; it is small and uncorrected for multiple comparisons | v2 Criteo study; `criteo_groups.py` |

## What this does not show

- The personalization questions are untouched. RQ2 (learned user vectors for click prediction), RQ3 (attention
  filter), RQ4 (recovering planted structure) and RQ5 (off-policy error) have not been run; the findings above
  concern only the forecasting part (RQ1).
- The synthetic result uses true click probabilities, so it bounds what forecasting can add to pacing, not what
  a learned click model can add to allocation.
- The Criteo pacing benchmark is a same-day hindsight constant rate, not a true ceiling, and the replay is
  throttling on logged impressions, not a live auction. One dataset, relative timestamps.
- iPinYou is too short (7 days) to train or fairly test a deep forecaster.
- Synthetic worlds plant clean daily rhythms; if a forecaster has an advantage there, it would be small. That is
  a property of the generator, not a general fact.

## Revised research questions (proposal)

1. **Value of looking ahead.** How much does a forecast-driven pacer gain over a reactive one, and how much of
   that does a plain replay capture? (Synthetic: mostly answered. Real: Criteo.)
2. **Value of a learned forecaster.** When does a learned forecaster beat the replay, and by how much?
   (Criteo: 1.3 to 2.0 points. Open: how it depends on budget tightness, campaign size and traffic volatility.)
3. **Architecture.** Do TFT, N-HiTS and LightGBM differ in pacing outcome once forecast error is matched?
   (Criteo so far: no.)
4. **Conditions.** Which world or data conditions make the forecast matter: churn, volatility, budget
   tightness? (Synthetic knob grid; Criteo by campaign group.)
5. Personalization questions RQ2 to RQ5 stay as written and run separately.

## Deviations that shaped this note (see Protocol Section 12)

1. G3 part B uses seed-to-seed noise instead of a fixed 0.5% bound per campaign-day.
2. CPC pricing is dropped from the main tables (value equals spend; the bound does not converge).
3. The primary metric is a ratio of totals, not a mean of per-day ratios.
4. Presence weights in the forecast pool are rescaled so the projected traffic is right (the old weights removed
   leavers but added no arrivals and projected 52% to 85% of actual traffic).

## Next steps

1. ~~Check that the v2 Criteo result reproduces.~~ Done 2026-10-19. An independent rerun on the Windows PC rebuilt
   the data from the raw file, retuned the PID controller, regenerated the seasonal-naive and LightGBM forecasts,
   and reran the simulation and summary. The saved TFT and N-HiTS forecasts were reused (file hashes match the
   recorded ones; neural training needs a GPU and was not repeated). Every number in the pacing summary (48 rows) and
   the PID gains match the original exactly; the paired-test p-values differ by under 1e-15. The only code change
   was a serial episode loop in place of the Mac-only `fork` pool. Still to say: how much headroom the benchmark
   leaves.
2. Report the Criteo result by campaign group (size, volatility, budget tightness) to answer question 4 on real
   data.
3. Update the Synthetic Audience World page with the final synthetic results.
4. Decide whether the personalization arm (RQ2 to RQ5) is run in this study or split into a second one.
