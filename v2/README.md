# v2 — Forecast-Driven Daily Budget Pacing on Real Campaign Logs

Applied study. The design, metrics and every deviation are fixed in [PROTOCOL.md](PROTOCOL.md).
The earlier v1 material is in `../legacy/` and is invalid (see `../legacy/README.md`).

## Data

Criteo Attribution Modeling for Bidding dataset (Diemert et al., AdKDD 2017), CC BY-NC-SA 4.0:
https://huggingface.co/datasets/criteo/criteo-attribution-dataset

Put `criteo_attribution_dataset.tsv.gz` at the path given by `PATHS["raw"]` in `src/pacing/config.py`.

## Run

```bash
python3.12 -m venv ~/venvs/pacing && ~/venvs/pacing/bin/pip install -r requirements.txt
export PYTHONPATH=src
~/venvs/pacing/bin/python -m pytest -q tests
~/venvs/pacing/bin/python scripts/01_build_data.py      # panel, campaign list, impression store
scripts/run_all.sh                                      # forecasts, PID tuning, test simulation, tables
```

| Script | Output (under `PATHS["results"]`) |
|---|---|
| `01_build_data.py` | `data_summary.json` |
| `02_forecast.py MODEL --seed S` | `forecasts/MODEL_seedS.npz` (+ `.meta.json`) |
| `03_forecast_metrics.py` | `forecast_metrics.json`, `forecast_metrics_per_campaign.parquet` |
| `04_tune_pid.py` | `pid_tuned.json`, `pid_grid.csv` (validation days only) |
| `05_simulate_test.py` | `pacing_test_episodes.parquet` |
| `06_summarize.py` | `pacing_summary.csv`, `pacing_comparisons.csv`, `tables/*.tex` |

Every results file has a `.meta.json` recording its time, code version, arguments and configuration.

## Rule for the paper

Every number, table and figure in the paper must come from a file above. Tables are included
with `\input{}` from `results/tables/`; no numbers are typed by hand.
