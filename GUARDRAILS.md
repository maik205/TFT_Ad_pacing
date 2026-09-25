# Reproducibility guardrails

The cheap, high-ROI minimum for a credible research repo. Copy this folder's
contents into your repo root (preserving the `src/ tests/ scripts/` layout) and
wire them in. They turn a checklist into checks the machine runs for you.

## What each does

- **src/utils/seeding.py** — `set_seed()` seeds Python/NumPy/PyTorch (+CUDA) and
  requests deterministic algorithms; `log_run_config()` writes
  `config.json` (resolved config, git commit, library versions, seed) into each
  run directory. Call both at the start of every entry point.

- **src/data/splits.py** — `loso_splits()` / `grouped_kfold()` build
  subject-independent folds; `assert_no_subject_leakage()` raises if any subject
  lands in both train and test. Use these for every split; never random-split
  per-window data.

- **src/utils/checkpointing.py** — `save_checkpoint()` / `load_checkpoint()` and
  `BestTracker` to save the best-on-validation weights and evaluate the test set
  with exactly that checkpoint (never tune on test); `sha256_of()` for provenance.

- **src/utils/results.py** — `log_metrics()` writes `metrics.json` in the schema
  `make_tables.py` reads, and records the path + SHA-256 of the checkpoint that
  produced the numbers, so each table entry traces to specific weights.

- **tests/test_no_leakage.py** — fails if a split shares a subject. This guards
  against the single bug that can invalidate an entire paper.

- **tests/test_determinism.py** — same seed must give the same metrics. Point it
  at your real pipeline by implementing `run_once(seed) -> dict` in
  `src/train.py`; until then it checks the seeding utility itself.

- **scripts/make_tables.py** — reads `runs/*/metrics.json` and emits LaTeX + CSV
  tables (mean +/- std). No paper number should be typed by hand.

## Wire-in (one-time)

1. Entry point: `set_seed(cfg.seed); log_run_config(cfg, run_dir, cfg.seed)`.
2. Cross-validation: iterate `loso_splits(subject_ids)` (leakage-checked inside).
3. Training: track validation with `BestTracker`; on improvement,
   `save_checkpoint(state, run_dir, "best")`. Evaluate the test set by loading
   that best checkpoint.
4. After evaluation: `log_metrics(run_dir, method, dataset, metrics,
   checkpoint=best_path, seed=cfg.seed)` so `runs/<id>/metrics.json` carries the
   numbers and the weights' provenance.
5. Pre-commit smoke check: `pytest tests/` then `python scripts/make_tables.py`.

## Run

    pip install -r requirements.txt -r requirements-dev.txt
    pytest tests/
    python scripts/make_tables.py --runs runs --out paper/tables

`requirements.txt` ships UNPINNED (a dependency list, not versions). At project
start, pin each package to the current latest-stable from PyPI and record the
date + Python target in the file header — reproducibility needs exact pins, not a
bare list. `pytest tests/` doubles as an install smoke check: if the stack is
internally inconsistent, the tests fail fast.

## Deliberately NOT included (add only when justified)

CI on every push, a pinned lockfile/Dockerfile, and a verified one-command
reproduce. Worth it for a team or a public code release; unnecessary overhead for
a solo, pre-results project. Add them when you publish the repo, not before.
