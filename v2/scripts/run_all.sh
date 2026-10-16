#!/bin/zsh
# Full v2 run on the Mac. Logs go to ~/pacing_v2/logs/.
set -e
cd ~/pacing_v2/code
export PYTHONPATH=src
PY=~/venvs/pacing/bin/python
mkdir -p ~/pacing_v2/logs
L=~/pacing_v2/logs
# data_summary.json lives in results/; rebuild it so a fresh results/ is complete
$PY scripts/01_build_data.py > $L/01_build_data.log 2>&1
# PID tuning runs alongside the neural training (different cores)
$PY scripts/04_tune_pid.py > $L/04_tune_pid.log 2>&1 &
PID_JOB=$!
$PY scripts/02_forecast.py naive > $L/02_naive.log 2>&1
$PY scripts/02_forecast.py lightgbm > $L/02_lightgbm.log 2>&1
echo "$(date) naive+lightgbm done" >> $L/progress.log
for seed in 0 1 2; do
  for m in TFT NHITS; do
    echo "$(date) start $m seed $seed" >> $L/progress.log
    $PY scripts/02_forecast.py $m --seed $seed > $L/02_${m}_seed${seed}.log 2>&1
    echo "$(date) done  $m seed $seed" >> $L/progress.log
  done
done
wait $PID_JOB
echo "$(date) pid tuning done" >> $L/progress.log
$PY scripts/03_forecast_metrics.py > $L/03_forecast_metrics.log 2>&1
echo "$(date) forecast metrics done" >> $L/progress.log
$PY scripts/05_simulate_test.py > $L/05_simulate_test.log 2>&1
echo "$(date) test simulation done" >> $L/progress.log
$PY scripts/06_summarize.py > $L/06_summarize.log 2>&1
echo "$(date) ALL DONE" >> $L/progress.log
