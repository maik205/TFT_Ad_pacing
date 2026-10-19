# Checkpoint 1: devil's-advocate review of the protocol

Reviewed: `protocol.md` v1.0. Date: 2026-10-17.

### Verdict: PASS after one revision (v1.1)

### Critical issues (block progression)
No critical issues identified.

### Major issues

**DA1-M1. The granularity criterion would admit the whole budget-constrained bidding literature.**
Almost every advertiser-side bidding paper with a budget uses per-impression pCTR/pCVR at decision time,
so "information below the campaign aggregate" is met trivially. The included set would be dominated by
papers that do not touch the planned idea (forecasting *who* arrives and *how well* they match).
*Resolution (applied, v1.1):* included studies are split into Tier 1 (user/segment supply forecasts or
learned user/campaign representations feeding pacing) and Tier 2 (per-impression value prediction only).
The novelty analysis rests on Tier 1.

**DA1-M2. The agreement sample puts real work on the author.**
At least 50 title/abstract decisions plus borderline full-text exclusions, screened blind. Without it the
review has a single AI screener and the "systematic" label is weaker. *Not resolved by the protocol; the
author decides at confirmation.* If declined, the report must say screening was single-screener throughout.

### Minor issues

- **DA1-m1.** Relevance-ranked truncation (300 per string on Semantic Scholar and arXiv) can miss studies
  that use unexpected vocabulary. Partly offset by OpenAlex Boolean retrieval and citation tracking. Stated
  as a limitation.
- **DA1-m2.** No native ACM DL / IEEE Xplore search. KDD, WWW, CIKM and WSDM are the core venues; DBLP
  title queries and OpenAlex coverage partly offset this. Stated as a limitation.
- **DA1-m3.** SQ-b (time-of-day modelling) may have very few Tier 1 studies; the background strand B2 will
  carry most of that evidence, and it is not systematic. The report must not present B2 findings with
  systematic-review certainty.
- **DA1-m4.** Industrial papers rarely report variance and often use proprietary data; most studies may
  rate High on D5 and Some/High on D6. Expected, not a protocol flaw, but the certainty statements will
  likely be low; that is a result, not a failure.
- **DA1-m5.** The custom appraisal checklist is unvalidated. Stated in Section 9.

### Strongest counter-argument to the review as designed
"The novelty question could be answered faster by a targeted search of 20 key papers." True for speed, but
a targeted search cannot show absence of prior work with any confidence; the systematic search is what
makes a "not studied together" claim defensible. The author chose this form.
