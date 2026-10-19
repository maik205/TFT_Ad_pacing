# Revision round 1: point-by-point response and tracking

Source of comments: the self-review in `01_self_review.md` (no external reviewers exist yet).
This is an advisory author-side log. It is **not** a pipeline artifact: no Schema 11 commitment
ledger, no Material Passport entry, no "ready to submit" status. Do not send it to a venue as a
response letter.

## Paper information

| Field | Value |
|---|---|
| Title | Does Better Forecasting Buy Better Budget Pacing? |
| Revision round | 1 (self-review) |
| Date | 2026-10-17 |
| Venue | Undecided (no template-specific limits applied) |
| Length | 4 pages -> 5 pages in `acmart` sigconf |
| Prose words (approx., excluding tables and references) | about 2,110 -> about 3,000 |
| References | 14 -> 16 (added `dai2023percentile`, `zhang2016feedback`, both checked against arXiv listings) |
| Tables / figures added | 0 / 0 (alt text added to the three existing figures) |
| Macros | 438 -> 464 (26 added; 0 existing values changed; both generated tables byte-identical) |

## Revision method and its limits

- **Fidelity: high.** The experiments, forecasts, simulations, tables and figures were not touched.
  Only the paper text, the bibliography, the macro generator (`07_paper_assets.py`, new macros only)
  and the protocol's deviation log (entries 15–16) changed.
- **Patch toolchain not used.** The skill's mechanical patch/apply scripts (`ars_anchorize_draft.py`
  and friends) assume a Markdown draft with block anchors and a roadmap / claim-surface manifest set.
  This is a LaTeX source with neither, so edits were made as targeted replacements instead. The usual
  guarantee ("a block no operation names cannot be distorted") is therefore **not machine-enforced
  here**. The before/after copies are in git (`700eea7` vs. the working tree) for a manual diff.
- **Checks performed:** all 80 `\res{}` keys resolve; the macro file diff shows zero changed values;
  typed numerals in prose were listed (see rule below); the PDF was recompiled with no `??`.

### Rule on numbers in the paper

No measured, derived or configuration-sourced number is typed in the prose; they all come from
`\res{key}` macros. The numerals that remain typed by hand are definitional, not results: the 24-hour
day and lag, the quantile and budget level labels (0.5/0.7/0.9, 0.3/0.5/0.7), the 80% interval and
95% bootstrap levels, hardware names, the CC BY-NC-SA 4.0 licence, and the dataset documentation's
"30 days".

## Tracking table

Status values follow the template: RESOLVED, DELIBERATE_LIMITATION, UNRESOLVABLE, REVIEWER_DISAGREE.
Where a fix is complete only for the text and the full fix needs a new experiment, the row is RESOLVED
for this round and the open follow-up is listed in the last column.

| # | Issue | Type | Section | Resolution summary | Location of change | Status | Open follow-up (needs author approval) |
|---|---|---|---|---|---|---|---|
| M1 | Headline gain over even pacing conflates "using a demand profile" with "a learned forecaster" | Major | Abstract, 1, 6.2, 7, 8 | Added the seasonal-naive share of the gain and the neural increment on top of it; reworded contribution 3, the guidance and the conclusion | Abstract; Intro contribution 3; 6.2 para 1; 7 (i); 8 | RESOLVED | None |
| M2 | Tests ignore training-seed variance; TFT–NHITS gaps are smaller than NHITS's seed SD | Major | Abstract, 5, 6.2, 7, 8 | Reported seed SDs; stated that tests treat trained models as fixed; flagged the NN–LightGBM lower bound as inside seed noise; LightGBM and naive single-run caveat; "matters little" now says differences are within seed variation | Abstract; 5 Statistics; 6.2 para 2; 7; 8 | RESOLVED (reporting and hedging) | Seed-aware inference or more seeds (new experiment) |
| M3 | "Clearly ahead of LightGBM" is unsupported; median-campaign MASE does not separate NHITS and LightGBM | Major | 6.1, 8 | Reported median MASE and per-campaign win rates; stated that no CIs are given for MASE differences; replaced "clearly ahead" | 6.1 | RESOLVED (reporting and hedging) | Paired CI or test on MASE differences (new analysis) |
| M4 | Undisclosed validity threat: the log holds only impressions won under the logging bidder's own policy | Major | 3, 7 | Stated in the setup and in the limitations that replayed demand is conditional on the logging policy and the counterfactual is unverifiable | 3 Throttling environment; 7 Limitations | RESOLVED | A logging-policy-free dataset or off-policy evaluation would be needed to remove it |
| M5 | "Median works best" overreaches (quantile grid starts at 0.5, clicks fall monotonically); mechanism stated as fact | Major | Abstract, 6.2, 7, 8 | Now "best of the three levels tested"; the monotone decline is quantified; the sum-of-hourly-quantiles mechanism is a labelled hypothesis; quantiles below 0.5 and daily-total forecasts listed as untested | Abstract; 6.2 para 3; 6.3; 7 (ii); 8 | RESOLVED (hedging) | Quantiles 0.1–0.4 and a daily-total quantile forecast (new experiment) |
| M6 | Practical recommendations exceed the evidence | Major | 7, 8 | Guidance retitled "in this setting", conditions stated, small gain over PID at loose budget quantified | 7; 8 | RESOLVED | None |
| m1 | "Code, protocol and all results files are released" is not yet true | Minor | Abstract, 1 | Changed to "will release"; the large forecast/episode files remain on the compute host with checksums in the repository | Abstract; Intro | RESOLVED | Decide the release venue before submission |
| m2 | Config numbers typed in prose; "31 days" conflicts with the dataset documentation | Minor | 4, 5 | Generated from `config.py` as macros; stated the 742-hour span, 30 days used and 22 hours discarded | 4; 5 Data, Metrics | RESOLVED | None |
| m3 | PID and CI unexpanded; sPL scale and MASE$_h$ undefined | Minor | Abstract, 1, 5, Table 2 | Defined at first use and in the Table 2 caption | Abstract; Intro; 5 Metrics, Statistics; Table 2 caption | RESOLVED | None |
| m4 | Related work missing throttling at scale and PID feedback control; PID baseline uncited; Lee et al. description beyond what was verified | Minor | 2 | Added Dai et al. (2023) and Zhang et al. (2016); narrowed the Lee et al. sentence to what the abstract supports | 2 Budget pacing | RESOLVED | A wider literature search for forecast-driven pacing is still advisable before submission |
| m5 | "Widely deployed" not supported by sources | Minor | 1 | "A mechanism used in production systems" | 1 para 1 | RESOLVED | None |
| m6 | Protocol paragraph implied untouched design; profile of the full dataset and smoke checks on test data were not in the log | Minor | 5 Protocol, PROTOCOL.md | Disclosed both in the paper and as protocol entry 15 | 5 Protocol; PROTOCOL.md entry 15 | RESOLVED | None |
| m7 | PID baseline looks under-damped (PVI 0.43–0.60) | Minor | 7 | Added to limitations: a better-designed controller may narrow the gap | 7 Limitations | DELIBERATE_LIMITATION | Tune a stronger feedback baseline (new experiment) |
| m8 | Conclusion overstated the neural advantage | Minor | 8 | Rewritten to match the revised results | 8 | RESOLVED | None |
| m9 | Data licence not stated | Minor | 5 Data | Added CC BY-NC-SA 4.0 | 5 Data | RESOLVED | Check licence fit with the final venue's policy |
| m10 | acmart warning: figures lack descriptions | Editorial | Figures 1–3 | Added number-free `\Description` alt text | Figure environments | RESOLVED | None |

## Changes that were considered and not made

- No new experiments, seeds, quantiles or datasets. These were outside the authorized scope; the
  follow-ups above need a protocol amendment before they run.
- No change to the title. Its question (does *better* forecasting buy better pacing) is now answered
  more directly by the text: only slightly, once a demand profile is used.
- No rewrite of Methods, Results tables or figures.

## Points for the author

1. **Length.** The revision adds about 890 words, mostly hedging and disclosure. If the venue limits
   pages, the easiest trims are the new Related Work sentences and the Protocol paragraph; keep the
   seed-variation and logging-policy statements.
2. **Release wording.** The paper now says "will release". Decide what is public (repository,
   archive for the 155 MB of large files) before submission and then make the sentence exact.
3. **Follow-up experiments.** The five open items above, in order of value for the paper's claims:
   seed-aware inference or more seeds; quantiles below the median plus a daily-total forecast; paired
   CIs on MASE; a stronger feedback baseline; a second dataset or an off-policy check.
4. **Venue-specific review.** Once a venue is chosen, rerun the review with its criteria bound
   (`criteria_binding_unavailable` applied this round), and spot-check the remaining references
   named in `references.bib` against publisher pages.

## Files changed in this round

- `paper/main.tex`, `paper/references.bib`, `paper/main.pdf`
- `scripts/07_paper_assets.py` (new macros only), `results/tables/numbers.tex` (26 macros added),
  `results/paper_assets.meta.json`, `CODE_VERSION`
- `PROTOCOL.md` (entries 15 and 16)
- `paper/revision/01_self_review.md`, `paper/revision/02_response_and_tracking.md`
