# v4_ctr: click-probability model for the mgads ad server

This directory trains and exports the CTR model the mgads ad server ranks CPC campaigns with
(admob_platform ADR 0004, issue #194, FR-M4-07, BR-82, NFR-33). Everything here is trained and evaluated
on **synthetic** worlds (rule R1); nothing in it describes real traffic.

| Path | What it is |
|---|---|
| `synthetic/` | Fork of the frozen v3 generator with the served ad formats (banner, interstitial, rewarded). `FORK_SPEC.md` lists every difference from the frozen specification |
| `ctr/contract.py` | Input contract shared with the ad server: vocabularies, IAB category and country mappings, history length, outcome delay. Written into every exported manifest |
| `ctr/dataset.py` | Model inputs for every served impression of a world, from `observed/` only (rule R2) |
| `ctr/models.py` | `CadetCTR` (CADET adapted to the server's budget) and `LogRegCTR` (baseline) |
| `ctr/train.py` | Training on the train days, per-format Platt calibration on the validation days, predictions on the test days |
| `ctr/export.py` | ONNX export, manifest (version, sha256, contract, held-out metrics) and a golden file the server's tests check |
| `ctr/bench.py` | CPU latency of one scoring call |
| `evaluation/ctr_truth.py` | Comparison with the true click probability (the only code that reads `truth/`) |
| `RESULTS.md` | Results of the run that produced the served model |

Run everything with `run_all.bat` (generation, tests, datasets, training, evaluation, export, bench). On the
PC it takes about 25 minutes with the GPU.

## The model

`CadetCTR` follows CADET (arXiv 2602.11410) where it applies to this server:

- a decoder-only transformer over the user's last 32 served ads, newest first, followed by the candidates;
  each candidate attends to the history and to itself only, so all candidates of a request are scored in
  one call and none changes another's score;
- rotary position embeddings driven by each event's age in hours, not its index;
- self-gated attention (sigmoid gate on each attention output);
- session masking: the outcome of a served ad younger than one hour is an "unknown" token;
- the request context (format, app category, region, local hour and weekday, UTC hour) is added to every
  candidate token and fed again to a tower per ad format; the format is known before scoring, so it picks
  the tower (ADR 0004 decision 12);
- a per-format Platt scale fitted on validation data is part of the exported graph, because nothing after
  the model corrects its scale (ADR 0004 decision 7).

Size: d_model 64, 2 layers, 4 heads, 116,100 parameters, 515 KB as ONNX. The user id, app id and
placement id are not inputs: a per-id vector learned on a synthetic world would mean nothing for a real
id, so the user is represented by their served-ad history and the app by its category.
