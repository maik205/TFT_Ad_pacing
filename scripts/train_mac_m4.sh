#!/bin/bash
# Fast Training & Benchmarking Pipeline on Apple Silicon Mac (M4 / MPS)
set -e

echo "=========================================================="
echo "  TFT Ad Spend Pacing — Apple Silicon Mac M4 Accelerator  "
echo "=========================================================="

# 1. Setup virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment (venv)..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. Upgrade pip and install dependencies
echo "[2/4] Installing dependencies with native Apple Silicon MPS support..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Check data availability
if [ "$1" == "--reingest" ] || [ ! -f "data/train_hourly.parquet" ]; then
    echo "[3/4] Ingesting raw iPinYou bidding logs from local directory..."
    if [ ! -f "train.csv" ] || [ ! -f "test.csv" ]; then
        echo "Error: train.csv and test.csv not found in current folder."
        echo "Please ensure train.csv and test.csv are in $(pwd)"
        exit 1
    fi
    python data_ingestion.py --chunksize 1000000
else
    echo "[3/4] Using preprocessed data splits in data/ (pass --reingest to re-process raw CSVs)."
fi

# 4. Run TFT training and baseline simulations with MPS acceleration
echo "[4/4] Starting TFT Training & Pacing Simulations on Apple Silicon GPU (MPS)..."
export PYTHONPATH="."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
python -m pipeline.train_tft

# 5. Generate final publication plots from real out-of-sample prediction arrays
echo "Generating publication plots in plots/..."
python pipeline/plot_results.py

echo "=========================================================="
echo "  All runs completed! Results saved to data/ and plots/   "
echo "=========================================================="
