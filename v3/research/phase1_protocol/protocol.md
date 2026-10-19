# Protocol: User- and Segment-Level Prediction in Budget Pacing and Budgeted Ad Allocation — A Systematic Review

Protocol version: 1.0 (DRAFT, awaiting author confirmation)
Date: 2026-10-17
Registration: not registered. OSF registration is recommended before Phase 2 and is the author's decision.
Reporting guideline: PRISMA-P 2015 for this protocol, PRISMA 2020 for the report.

## Administrative information

| # | Name | Affiliation | Role |
|---|---|---|---|
| 1 | Mai Minh Đức | FPT University | Guarantor; protocol approval; second screener on the agreement sample (Section 7.3) |
| — | Claude (Anthropic), AI research assistant | — | Search execution, first-pass screening, extraction, appraisal, drafting. A tool, not an author |

Support: no external funding. Purpose: background and novelty check for a planned study (separate paper) on
audience-aware, personalized budget pacing from the ad-platform perspective.

### Amendments

| Date | Section | Change | Rationale |
|---|---|---|---|
| — | — | — | — |

Any change after confirmation is logged here with a date and reason. Searches are not rerun silently.

## 1. Rationale

Budget pacing spreads a campaign's budget over time. Most published controllers work on campaign-level
aggregates: spend so far, an aggregate traffic forecast, or a dual multiplier updated from realized spend.
Our planned study instead asks whether pacing improves when the platform models *who* will arrive and
*how well* each user matches each campaign at each time of day, through learned user and campaign
representations with a time-of-day-modulated affinity.

Before designing that study we need to know (a) which methods already feed user- or segment-level
predictions into pacing or budgeted allocation, (b) how they were evaluated, and (c) what evidence exists
that this beats campaign-level approaches. This defines the novelty gap and the baselines. We know of no
systematic review that answers this; Phase 2 will check for existing reviews (Section 5, block R).

## 2. Objectives and research questions

### Primary question (systematic core)

**PQ.** What methods use user-, segment- or impression-level predictions to inform budget pacing or
budget-constrained allocation of ad impressions to campaigns, how have they been evaluated, and what
evidence shows they improve delivery outcomes over campaign-level or forecast-free approaches?

Structured as PICOS, adapted for computational studies:

| Element | Definition |
|---|---|
| **P — problem setting** | Online advertising systems that serve or buy impressions for campaigns with budget constraints over a period (display, real-time bidding, sponsored search, feed or in-app ads, guaranteed delivery) |
| **I — intervention** | A pacing, delivery-control or budgeted-allocation method whose decisions use predictions below the campaign aggregate: per user, per audience segment, per impression or per user × campaign pair (e.g., user-level CTR/CVR or value predictions inside pacing, user- or segment-level supply forecasts, learned user or audience representations, time-aware user modelling) |
| **C — comparator** | Pacing or allocation using only campaign-level signals or none: even pacing, PID/feedback control, probabilistic throttling, aggregate-forecast pacing, dual or primal-dual methods without user-level prediction; or the same method with the user-level component removed |
| **O — outcomes** | Primary: delivered value under budget (clicks, conversions, revenue, utility), budget adherence (utilization, overspend, early depletion), delivery smoothness. Secondary: forecast accuracy, computational cost, evaluation design |
| **S — study design** | Empirical studies: online A/B tests, offline log replay, simulation, or off-policy evaluation, including algorithm papers that report such an evaluation |

### Secondary questions (sub-questions of the systematic core, answered from the same included set)

- **SQ-a (evaluation).** Which evaluation designs are used (online A/B, log replay, simulator, off-policy
  estimation), and how do the studies handle the counterfactual problem created by the logging policy?
- **SQ-b (time).** Which included methods model time-of-day or periodic variation in user behaviour or
  user–campaign affinity, and how?
- **SQ-c (gap).** Which combinations of {platform perspective, learned user and campaign representations,
  time-modulated affinity, supply forecasting, pacing multipliers} have not been studied together?

### Background strands (documented, not systematic)

These support the framing but are not screened under PRISMA and are not counted in the flow diagram.
Their queries and selected sources are recorded so they can be reproduced.

- **B1. Platform-side pacing and online allocation theory and practice**: online matching / AdWords,
  primal-dual and dual mirror descent budget pacing, guaranteed-delivery compact allocation plans, throttling
  vs bid modification, industrial multi-campaign pacing systems.
- **B2. Time-aware user and item representations**: temporal and periodic recommender models,
  time-of-day-aware embeddings, session and periodic interest models, and their use in ads.

### Dataset track (documentation audit, not a literature review)

**DQ.** Which public datasets support the planned study? Assessed against fixed requirements (Section 10)
using each dataset's official documentation and originating paper only. No dataset is downloaded or
analysed in this review.

## 3. Eligibility criteria (systematic core)

| Criterion | Include | Exclude |
|---|---|---|
| Topic | Pacing, budget smoothing, delivery control, or budget-constrained allocation of ad impressions/traffic to campaigns; or supply / inventory / traffic forecasting explicitly used to drive such allocation or pacing | CTR/CVR prediction with no budget or pacing component; bid optimization without a budget constraint over time; auction or mechanism design; bid shading for first-price auctions without pacing; reserve-price optimization |
| Granularity | Decisions or forecasts use information below the campaign aggregate (user, segment, impression, user × campaign) | Only campaign- or ad-group-level aggregate signals (these become comparators and background, not included studies) |
| Design | Reports an empirical evaluation (online, offline replay, simulation, or off-policy) with at least one comparator | Theory-only papers with no experiment (moved to background B1); position papers; tutorials |
| Publication type | Peer-reviewed journal or conference papers; arXiv or other preprints; technical reports from industry labs with full method descriptions | Patents; abstracts only; posters without a full paper; theses (unless no paper version exists); slides; blog posts |
| Date | 2005-01-01 to 2026-10-17 | Earlier (sponsored-search budget allocation research begins about 2005) |
| Language | English | Other languages (resource limit; stated as a limitation) |
| Duplicates | The most complete version (usually the archival paper over the preprint); versions are linked | Additional versions |

Two tiers of included studies (assigned at full text, synthesized separately):
- **Tier 1**: the method forecasts future arrivals or supply at user, segment or user × campaign level,
  or learns user and/or campaign representations that feed pacing or allocation decisions.
- **Tier 2**: the only sub-campaign information is a per-impression value prediction (e.g., pCTR, pCVR)
  used at decision time, as in most budget-constrained bidding papers.

Tier 2 is included so the review shows how common that design is, but the novelty analysis (SQ-c) rests on
Tier 1.

Borderline rule: a study is included when it meets every criterion; if the full text cannot settle
granularity or design, the study is excluded with the reason "insufficient information" and listed.

## 4. Information sources

| Source | Access | Role |
|---|---|---|
| OpenAlex | api.openalex.org (title/abstract search) | Main bibliographic database (broad, includes ACM, IEEE, Springer, arXiv) |
| Semantic Scholar | api.semanticscholar.org (relevance search + citation graph) | Second database; backward and forward citation tracking |
| arXiv | export.arxiv.org API | Preprints, industry-lab papers |
| DBLP | dblp.org search API | Computer-science venue coverage check |
| Google Scholar | web search | Supplementary check only, results recorded but not relied on for completeness |
| Citation tracking | Semantic Scholar | Backward (references) and forward (citations) of every included study, one round |

Not available to this review: ACM Digital Library, IEEE Xplore, Scopus and Web of Science native search
(no access from this environment). OpenAlex and Semantic Scholar index most of their content, but this is a
stated limitation.

## 5. Search strategy

Three concept blocks, combined as A AND B AND C. R is a separate search for existing reviews.

```
A (pacing / allocation):
  "budget pacing" OR pacing OR "budget smoothing" OR "budget allocation" OR "budget constrained"
  OR "budget-constrained" OR "delivery control" OR "guaranteed delivery" OR "online allocation"
  OR throttling OR "ad allocation" OR "allocation plan"
B (advertising):
  advertising OR advertisement OR "ad serving" OR "display ads" OR "real-time bidding" OR RTB
  OR "sponsored search" OR campaign OR advertiser
C (sub-campaign granularity):
  user OR users OR "user-level" OR audience OR segment OR personalized OR personalised
  OR "supply forecast" OR "traffic forecast" OR "inventory forecast" OR impression-level
  OR embedding OR "user behavior" OR "user behaviour"
R (existing reviews):
  (A) AND (B) AND (survey OR review OR overview)
```

Per source:
- **OpenAlex**: Boolean A AND B AND C on title + abstract; date filter as in Section 3. All records retrieved
  if the total is at most 3,000. Above that, the query is tightened by restricting block C to title +
  abstract phrase terms; the tightening and both counts are logged as an amendment.
- **Semantic Scholar and arXiv** do not support full Boolean search, so a fixed list of query strings is run
  (Appendix A), taking the first 300 relevance-ranked results per string. The truncation is a stated
  limitation.
- **DBLP**: title keyword queries (Appendix A), all results.
- Exact query strings, run dates and raw counts are saved in `phase2_search/search_log.md`.

Search peer review: the author checks the search strings at confirmation (simplified PRESS step).

## 6. Data management

- Records are exported to `phase2_search/records_raw.jsonl` (one line per record, with source and query).
- Deduplication: DOI match, then normalized title match (lowercase, punctuation removed, Jaccard ≥ 0.9 on
  word sets), checked by hand. Linked versions are kept in `dedup_map.csv`.
- All screening decisions with reasons are kept in `screening_ta.csv` and `screening_ft.csv`.

## 7. Selection process

1. **Pilot.** The first 50 records (random sample, fixed seed) are screened, criteria wording is clarified
   if needed, and any clarification is logged as an amendment before full screening.
2. **Title/abstract screening.** Every record is screened against Section 3. Uncertain records go to full text.
3. **Agreement sample.** A random 20% sample of title/abstract decisions (minimum 50 records) and all
   full-text exclusions that are marked borderline are given to the author for independent screening, blind
   to the first decision. Cohen's κ is reported. If κ < 0.6, the criteria are clarified (amendment) and the
   disagreeing category is re-screened.
4. **Full-text screening.** Every record passing title/abstract is read in full; exclusion reasons use a
   fixed list (wrong topic, campaign-level only, no empirical evaluation, wrong publication type,
   insufficient information, duplicate).
5. **Citation tracking.** One round of backward and forward tracking from the included set, screened the
   same way.

Independence limitation: apart from the agreement sample, screening is done by a single AI screener. The
report will state this.

## 8. Data items (extraction form, piloted on 3 studies)

| Category | Items |
|---|---|
| Metadata | Authors, year, venue, version links, affiliations (industry / academic), code or data availability |
| Setting | Ad channel, platform vs advertiser/DSP perspective, auction type if stated, budget horizon |
| Method | Pacing/allocation mechanism (throttling, bid modification, dual multiplier, allocation plan, RL, other); what sub-campaign prediction is used (user / segment / impression / user × campaign); prediction model; whether representations are learned; whether time-of-day or periodicity is modelled and how; whether supply or arrival is forecast |
| Evaluation | Design (online A/B, log replay, simulator, OPE); dataset (public name or proprietary); scale; split and leakage handling; how the logging-policy counterfactual is handled; comparators and whether they were tuned |
| Outcomes | Each reported outcome, its definition, the comparator and the reported effect (as stated, with variance if given) |
| Notes | Author-stated limitations, with section locators |

Missing items are recorded as "not reported"; authors are not contacted (time constraint).

## 9. Appraisal of included studies

RoB 2 and ROBINS-I are designed for clinical studies and do not fit algorithmic evaluations. We use a
custom appraisal checklist for computational experiments, defined here before any study is read. It is
**not a validated instrument**, and the report will say so.

| Domain | Low concern | High concern |
|---|---|---|
| D1. Data realism | Real production or public logs | Fully synthetic data, or real data with undocumented transformation |
| D2. Counterfactual validity | Online A/B test, or OPE with known propensities | Replay or simulator that scores policies using the same model the policy relies on, or that ignores logging-policy effects without discussion |
| D3. Comparator adequacy | Comparators are named, appropriate and tuned on the same data | Weak or untuned comparators only, or none from the campaign-level family |
| D4. Temporal validity | Train/test split in time; no future information in features or tuning | Random splits across time, or tuning on test |
| D5. Statistical reporting | Variance, repeated runs, or confidence intervals | Single run, point estimates only |
| D6. Reproducibility | Code and data available, or enough detail to reimplement | Key details missing |

Each domain is rated Low / Some concerns / High with a one-line justification and locator. Ratings are
used in the synthesis (Section 11) and in a sensitivity statement (results excluding High-concern studies
on D2 or D4).

## 10. Dataset track (DQ) method

Candidate datasets, fixed in advance: iPinYou RTB (raw logs), Taobao / Alimama display ad click data,
Criteo Attribution Modeling for Bidding, Avazu CTR, Open Bandit Dataset. Further datasets are added only if
they are cited by an included study as used for pacing or allocation, or found in the dataset searches in
Appendix A; each addition is logged.

Requirements matrix (columns fixed now):
1. Persistent user IDs across days, and any documented repeat rate
2. Local time or time-zone information in timestamps
3. Named (non-anonymized) categories for ads/campaigns or user interests
4. Cost, price or bid per impression
5. Bid opportunities logged, or only won impressions
6. Duration
7. Number of campaigns/advertisers, and whether users meet many campaigns (true many-to-many)
8. Logging propensities or randomized logging
9. Licence and access conditions

Rule: every cell cites an official source (dataset page, documentation, or originating paper) with a locator,
or reads "not documented". Nothing is filled from memory.

## 11. Synthesis

- **Meta-analysis** only if at least 3 included studies report the same outcome, on the same public dataset
  or under comparable designs, with variance. We expect this to be infeasible (different proprietary data and
  metrics), and the feasibility check will be reported either way.
- **Otherwise** a structured narrative synthesis following SWiM (Campbell et al., 2020): studies grouped by
  mechanism and by granularity of prediction; effect direction tabulated per outcome (better / no
  difference / worse than the comparator), with the D2/D4 sensitivity statement.
- **Certainty** is stated per outcome group with a GRADE-style qualitative statement (high / moderate /
  low / very low) based on appraisal, consistency, directness to our setting, and precision. GRADE was
  designed for health evidence; this is an adapted use and will be labelled as such.
- **Gap analysis** (SQ-c) as a matrix of included studies against the components of the planned method.

## 12. Meta-biases

- Publication bias: industrial papers report successful launches; negative results are rare. Formal tests
  (funnel plots) are not planned unless a meta-analysis is run with at least 10 studies. The bias will be
  discussed qualitatively.
- Selective reporting: compared between preprint and archival versions where both exist.

## 13. Phase plan and checkpoints

| Phase | Output | Checkpoint |
|---|---|---|
| 1 | This protocol | Devil's-advocate check 1; author confirmation |
| 2 | Search log, records, screening files, PRISMA flow, verified source list, appraisal table | Author agreement sample |
| 3 | Narrative synthesis, effect-direction table, certainty statements, gap matrix, dataset matrix | Devil's-advocate check 2 |
| 4 | PRISMA 2020 report in `v3/research/` | — |
| 5–6 | Editorial, ethics and devil's-advocate review; revision | Devil's-advocate check 3 |

## Appendix A. Query strings for sources without full Boolean support

Semantic Scholar and arXiv (each run separately; first 300 results each):
1. budget pacing user level advertising
2. budget pacing audience segment advertising
3. personalized budget pacing advertising
4. budget pacing traffic forecasting advertising
5. supply forecasting guaranteed delivery display advertising
6. inventory forecasting ad allocation user
7. budget constrained ad allocation user embedding
8. online ad allocation budget user arrival forecast
9. pacing multiplier user value prediction advertising
10. budget smoothing user response prediction real-time bidding
11. time-aware user interest advertising budget
12. survey budget pacing online advertising (block R)

DBLP title queries: `pacing advertis`, `budget pacing`, `budget allocation advertis user`,
`guaranteed delivery`, `inventory forecast advertis`, `supply forecast advertis`.

Dataset searches (dataset track only): Semantic Scholar and web search for "advertising dataset user
timestamp category budget", "real-time bidding dataset", "display advertising dataset user behavior",
"open bandit dataset propensity advertising".

## Appendix B. Background strand queries (not systematic)

- B1: AdWords online matching; primal-dual budget allocation advertising; dual mirror descent online
  allocation; budget pacing throttling bid modification; guaranteed delivery compact allocation plan;
  industrial pacing systems (smart pacing, multi-campaign pacing).
- B2: time-aware recommender systems; periodic user interest modeling; time-of-day user embeddings;
  temporal point process recommendation; time-aware user modeling in advertising.

Selection rule for background: seminal or most-cited works plus recent surveys, every one verified to
exist (title, authors, venue, URL). Unverifiable items are dropped, not hedged.

## Revision log

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2026-10-17 | Claude (draft) for Mai Minh Đức | Initial draft |
| 1.1 | 2026-10-17 | Claude (draft) | Added Tier 1 / Tier 2 split after devil's-advocate check 1 (issue DA1-M1) |
