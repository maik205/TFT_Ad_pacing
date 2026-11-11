# Ad-format fork of the synthetic generator

Status: fork of the frozen v3 specification (`BASE_SPEC_v3_frozen.md`, frozen 2026-10-17, sha256
`4445b00d14be1dea7c0197ada0434b508e84009037b0d871b16076692239da99`). Date: 2026-11-11.

The frozen specification stays as it was. This file lists every difference between this fork and that
specification; anything not listed here is unchanged, including the integrity rules R1 to R5, the click
model terms of Section 8, the logging policy, the cost models and the splits.

## Why the fork exists

The CTR model for the mgads ad server (admob_platform ADR 0004, issue #194) has one head per ad format
the server can render: banner, interstitial (served as VIDEO) and rewarded. The frozen generator simulates
display, pre-roll and in-feed slots. Display corresponds to banner and pre-roll is the nearest to an
interstitial video, but nothing corresponds to rewarded, and in-feed has no counterpart among the served
formats. ADR 0004 decision 13 forks the generator, not edits it, to cover the served formats.

## Deviation log

| # | Date | Section | Change | Reason |
|---|---|---|---|---|
| F1 | 2026-11-11 | 7 | Slot formats are banner, interstitial and rewarded, drawn with shares 0.5 / 0.3 / 0.2 (`format_mix`) for slots of an attentive visit. An absent visit never opens a rewarded slot; its autoplay slots split banner : interstitial in the same 5 : 3 ratio. Replaces `p_video` and `p_feed` | Rewarded ads are opt-in: a user who has stopped paying attention does not ask for one. The shares are design choices; no data source for them is used |
| F2 | 2026-11-11 | 7, 7a.2 | Interstitial slots use the v3 pre-roll mechanics unchanged (durations 6 / 15 / 30 s with probabilities 0.3 / 0.4 / 0.3, skippable after 5 s from 15 s). Rewarded slots last 15 or 30 s (0.5 / 0.5), are never skippable, and are completed with probability 0.97 when engaged and 0.90 when casual (`p_complete_rewarded`); a rewarded view that is not completed stops at Uniform(5 s, D). Banner slots record dwell, interaction and backgrounded only, as display did | The reward is paid only for a completed view, so completion is high and a skip control does not exist. The completion rates are design choices |
| F3 | 2026-11-11 | 8 | The click logit gains a format term: `+ s_format[f]` with s_format = (banner 0.0, interstitial +0.4, rewarded -0.4), and the affinity term is scaled per format: `gamma * gamma_format[f] * sum_k m_ck * interest_u(k, t)` with gamma_format = (1.0, 1.0, 0.5). `beta0` is still solved so the mean logging-policy CTR equals the target | A full-screen interstitial draws more clicks than a banner. A rewarded viewer watches for the reward rather than the product, so clicks are fewer and depend less on how well the ad matches the user's interest. This is the stated incentive effect ADR 0004 asks the fork to record. Both vectors are design choices, not calibrated; they give the per-format heads a real difference to learn |
| F4 | 2026-11-11 | 7, 7a.2, 14, 15 | The in-feed slot is removed, with its watch-fraction model, its KuaiRand fit (Section 14) and test 9's in-feed criterion. `calibrate.py` is not part of the fork; `calibration.json` keeps only `gamma_real` | No served format corresponds to in-feed. `gamma_real` does not change: page context does not depend on the slot format, so the MI excess it is solved against is the same |
| F5 | 2026-11-11 | 11 | `synth_meta.json` reports per-format slot and impression shares, CTR and completion (`checks.formats`) in place of the in-feed check; `truth_world.json` records `s_format`, `gamma_format` and `p_complete_rewarded` and no longer `infeed` | Follows from F1 to F4 |
| F6 | 2026-11-11 | 15 | New integrity test `test_formats`: no rewarded slot in an absent visit, attentive format shares within 1 point of `format_mix`, rewarded views never skipped, rewarded completion per state at the planted rates, no completion recorded for banners | Checks F1 and F2 |

## Defaults that matter for the CTR work

The CTR work uses the default world (gamma = 1.0) and the negative control (gamma = 0), three world seeds
each, with every other knob at its v3 default. Under gamma = 0, F3's affinity factor has nothing to scale
and only the format offsets remain, so the negative control still has no personal signal to find.
