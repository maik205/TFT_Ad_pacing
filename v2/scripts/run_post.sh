#!/bin/zsh
# Steps after all forecasts are present in results/forecasts (Mac).
set -e
cd ~/pacing_v2/code
export PYTHONPATH=src
PY=~/venvs/pacing/bin/python
L=~/pacing_v2/logs
$PY scripts/01_build_data.py > $L/01_build_data.log 2>&1
$PY scripts/03_forecast_metrics.py > $L/03_forecast_metrics.log 2>&1
echo "$(date) forecast metrics done" >> $L/progress.log
$PY scripts/05_simulate_test.py > $L/05_simulate_test.log 2>&1
echo "$(date) test simulation done" >> $L/progress.log
$PY scripts/06_summarize.py > $L/06_summarize.log 2>&1
echo "$(date) ALL DONE" >> $L/progress.log
