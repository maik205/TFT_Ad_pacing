# Reproduction evidence (2026-09-25)

Artifacts produced by a re-run of the committed pipeline (`e5af244`) made in a separate
session on 2026-09-25, before S05. They are kept as evidence for S01 finding F14
(committed numbers not reproducible from the committed checkpoint; no seeds).

- `repro_log.txt` — console log: eval-only with `data/tft_best_checkpoint.pt`, then a retrain.
- `tft_best_checkpoint_repro.pt`, `tft_best_checkpoint_repro2.pt` — checkpoints from the re-runs.
- `data_after_rerun/` — `data/*.json` as overwritten by the re-run (originals restored in `data/`).
- `plots_after_rerun/`, `paper_plots_after_rerun/` — figures as overwritten by the re-run.
- `plots_original/` — copy of the figures taken during the re-run session.

These files are evidence only; no number in the manuscript may be taken from them.
