#!/bin/bash
# Fast Training & Benchmarking Pipeline on Apple Silicon Mac (M4 / MPS)
set -e

echo "=== Setting up TFT Ad Pacing on Apple Silicon Mac M4 ==="

# 1. Setup virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. Upgrade pip and install dependencies
echo "Installing dependencies with native Apple Silicon arm64 PyTorch..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Check data availability
if [ ! -f "data/train_hourly.parquet" ]; then
    echo "Preprocessed data not found. Checking raw CSV files..."
    if [ ! -f "train.csv" ] || [ ! -f "test.csv" ]; then
        echo "Raw CSV files not found locally."
        echo "Please download them using: python scripts/sync_data.py download --bucket YOUR_BUCKET"
        echo "Or copy train.csv and test.csv into this directory."
        exit 1
    fi
    echo "Running data ingestion and resampling pipeline..."
    python data_ingestion.py --chunksize 1000000
fi

# 4. Run TFT training and baseline simulations with MPS acceleration
echo "Starting TFT Training & Pacing Simulations on Apple Silicon GPU (MPS)..."
export PYTHONPATH="."
python -m pipeline.train_tft

# 5. Generate final publication plots
echo "Generating publication plots..."
python pipeline/plot_results.py

echo "=== Pipeline Completed Successfully! ==="
