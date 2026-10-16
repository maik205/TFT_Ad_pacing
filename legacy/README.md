# legacy/ — v1 material (INVALID, kept for the record only)

Nothing in this folder is a valid research result. Do not cite or reuse any number, table,
figure, claim or summary document from here.

An audit on 2026-10-16 found the following:
- **Data:** `data_ingestion.py` read iPinYou_x1, a label-encoded CTR benchmark with no
  timestamps, bid prices or paid prices. It collapsed the data into one weekly template and
  repeated that template with added noise, trend and injected shocks to fake 90 days.
  "Spend" was derived from encoded price-bucket IDs.
- **Ablations:** `pipeline/train_tft.py` produced the ablation table by multiplying the
  full model's metrics by constants. No ablated models were ever trained.
- **Figures:** `pipeline/plot_results.py` drew Fig. 1 (forecasts) and Fig. 5 (attention)
  from sine and Gaussian formulas.
- **Metrics:** MASE compared rows of different campaigns. The test budget leaked into the
  input features.
- **Documents:** `STATE_OF_THE_UNION.md`, `AGENT_HANDOFF.md` and the commit messages
  describe these results as genuine. They are not.

The original README follows, unchanged.

---

# Optimizing Programmatic Ad Spend Pacing via Multi-Horizon Temporal Fusion Transformers

**Target Venue:** ICIIT 2027 (ACM ICPS, Scopus/Ei indexed)  
**Track:** Real-Time Bidding (RTB), Demand-Side Platform (DSP) Budget Pacing & Time Series Forecasting  

---

## 1. Project Overview
This repository contains the complete research artifacts, mathematical formulations, and empirical training pipeline for:
- **Paper Title:** *"Optimizing Programmatic Ad Spend Pacing via Multi-Horizon Temporal Fusion Transformers"*
- **Forecasting Engine:** Multi-Horizon Temporal Fusion Transformer (TFT) with Gated Residual Networks (GRN), Variable Selection Networks (VSN), and Interpretable Multi-Head Self-Attention ($k=168\text{h}$ lookback, $\tau_{\max}=24\text{h}$ horizon).
- **Pacing Control Policy:** Closed-form non-linear quantile throttling factor:
  $$\alpha_{t+1}^{(i)} = \text{clip}\left(\alpha_t^{(i)} \cdot \frac{B^{(i)} - S_t^{(i)}}{\sum_{\tau=1}^{\tau_{\max}} \hat{y}_{t+\tau}^{(i)}(0.90)}, \; \alpha_{\min}, \; 1.0\right)$$
- **Spike Detection:** Dual-boundary dynamic trigger:
  $$y_t > \hat{y}_t(0.90) \quad \text{AND} \quad \left(y_t - \hat{y}_t(0.50)\right) > 1.5 \cdot \left(\frac{B}{T}\right)$$

---

## 2. Directory Structure

```
.
├── README.md
├── requirements.txt            # Cross-platform dependencies (Apple Silicon M4 / Linux / Windows)
├── .gitignore                  # Excludes 2GB raw CSVs, venv, and LaTeX build junk
├── data_ingestion.py           # Hourly resampling pipeline for 19.5M iPinYou logs
├── data/                       # Preprocessed hourly splits (train 60d, val 15d, test 15d)
├── paper/                      # Publication manuscript in official ACM SIGCONF template
│   ├── main.tex                # Full paper LaTeX source
│   ├── references.bib          # BibTeX bibliography
│   ├── main.pdf                # Compiled 6-page camera-ready manuscript
│   └── plots/                  # Embedded high-res figures
├── pipeline/                   # Machine learning and simulation code
│   ├── models/tft_model.py     # PyTorch TFT architecture
│   ├── baselines.py            # ULP, PID, and Lagged LightGBM baselines
│   ├── pacing_simulation.py    # Auction pacing simulation environment
│   ├── evaluate.py             # Statistical and economic metrics
│   ├── train_tft.py            # End-to-end training and evaluation pipeline
│   └── plot_results.py         # Publication-grade vector figure renderer
├── plots/                      # Output vector PDFs and 300 DPI PNGs
└── scripts/
    ├── sync_data.py            # Object store upload/download script (S3, R2, MinIO)
    └── train_mac_m4.sh         # One-click training script for Apple Silicon Mac M4
```

---

## 3. Large Dataset Storage (Separated Object Store)

The raw iPinYou bidding dataset files (`train.csv` ~1.6GB and `test.csv` ~429MB) are intentionally excluded from Git via `.gitignore` to keep the repository lightweight and fast.

### Upload Raw Data to Object Storage (from source machine):
Set your bucket credentials (S3, Cloudflare R2, MinIO, or Railway Bucket):
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export S3_BUCKET_NAME="your-rtb-bucket"
# For Cloudflare R2 / MinIO: export S3_ENDPOINT_URL="https://<account_id>.r2.cloudflarestorage.com"

python scripts/sync_data.py upload --bucket your-rtb-bucket
```

### Download Raw Data (onto Mac M4):
```bash
python scripts/sync_data.py download --bucket your-rtb-bucket
```

*(Note: The preprocessed 90-day hourly aggregated splits in `data/*.parquet` are already included in the repo (~1.5MB total), so you can train immediately without re-processing the 19.5M raw auction logs if desired!)*

---

## 4. Running Training & Simulations on Apple Silicon Mac M4

On your Mac M4, PyTorch automatically leverages the **Metal Performance Shaders (MPS)** hardware acceleration for high-throughput training:

```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd TFT_Ad_pacing

# One-click setup and execution
chmod +x scripts/train_mac_m4.sh
./scripts/train_mac_m4.sh
```

Or manually:
```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training and pacing simulations (auto-detects MPS)
export PYTHONPATH="."
python -m pipeline.train_tft

# 4. Generate final publication figures
python pipeline/plot_results.py
```

---

## 5. Compiling the LaTeX Paper
The paper is written using the official ACM SIGCONF template (`acmart`):
```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Output manuscript is generated at `paper/main.pdf`.
