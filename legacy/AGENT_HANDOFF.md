# AGENT HANDOFF & SYSTEM STATE (APPLE SILICON MAC M4)

## 1. Project Context & Target
* **Paper Title:** "Optimizing Programmatic Ad Spend Pacing via Multi-Horizon Temporal Fusion Transformers"
* **Primary Target Venue:** ICIIT 2027 (ACM ICPS, Scopus/Ei indexed).
* **Research Problem:** Real-Time Bidding (RTB) budget pacing under non-stationary auction volatility. Replaces myopic linear/PID controllers with a multi-horizon Temporal Fusion Transformer (TFT) quantile-hedging throttling policy:
  $$\alpha_{t+1}^{(i)} = \text{clip}\left(\alpha_t^{(i)} \cdot \frac{B^{(i)} - S_t^{(i)}}{\sum_{\tau=1}^{\tau_{\max}} \hat{y}_{t+\tau}^{(i)}(0.90)}, \; \alpha_{\min}, \; 1.0\right)$$
* **Dual-Boundary Spike Detection:**
  $$y_t > \hat{y}_t(0.90) \quad \text{AND} \quad \left(y_t - \hat{y}_t(0.50)\right) > 1.5 \cdot \left(\frac{B}{T}\right)$$
* **GitHub Repository:** `https://github.com/maik205/TFT_Ad_pacing.git` (`master` branch).

---

## 2. Environment State on This Mac M4
* **Hardware:** Apple Silicon Mac M4 (Unified Memory, Metal GPU, Apple Neural Engine).
* **Local Data Files (Untracked by Git):**
  * `train.csv` (~1.6 GB, 15.4M auction logs)
  * `test.csv` (~429 MB, 4.1M auction logs)
* **Preprocessed Data (Tracked in `data/`):**
  * `data/train_hourly.parquet` (60 days = 1440 hours, 12,960 rows)
  * `data/val_hourly.parquet` (15 days = 360 hours, 3,240 rows)
  * `data/test_hourly.parquet` (15 days = 360 hours, 3,240 rows)
* **Acceleration Target:** Apple Silicon GPU via Metal Performance Shaders (`torch.device("mps")`).
  * *Note on ANE:* Apple Neural Engine does not support backward passes/training; MPS GPU is the designated training engine. (CoreML conversion for ANE forward inference is optional).

---

## 3. Critical Fixes Already Pushed to `master`
1. **Metal MPS Buffer Assertion (`MPSNDArray.mm:893`):**
   * *Root cause:* PyTorch `.expand()` creates virtual views with stride 0 (allocating only 8 KB), while Metal's matrix multiplication kernel requires a physical contiguous buffer of $32 \times 168 \times 64 \times 4 = 1,376,256$ bytes.
   * *Fix in `pipeline/models/tft_model.py`:* Replaced `.expand()` with `.repeat().contiguous()`, satisfying Metal's memory descriptor.
2. **Watermark Assertion (`RuntimeError: invalid low watermark ratio 1.4`):**
   * *Fix in `scripts/train_mac_m4.sh`:* Set both `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0` to disable artificial limits and allow dynamic allocation from the 16 GB unified RAM.
3. **Simulation Forecast Window Indexing (`IndexError: index 8 is out of bounds`):**
   * *Fix in `pipeline/pacing_simulation.py`:* Mapped hour $t$ to daily forecast window index `w_idx = min(t // 24, num_windows - 1)` and hour offset `h_idx = t % 24`, resolving the 2D out-of-bounds error.
4. **Real Data Wiring in Plots & Metrics:**
   * `pipeline/train_tft.py` now writes the computed out-of-sample metrics to `data/benchmark_results.json` and dumps simulation trajectories to `data/sample_simulation_trajectory.json`.
   * `pipeline/plot_results.py` dynamically loads the real simulation arrays to render final figures.

---

## 4. Immediate Next Tasks for Mac Agent

### Step 1: Pull the Latest Repository
```bash
git pull origin master
```

### Step 2: Run Full Training & Simulation on M4 MPS
```bash
chmod +x scripts/train_mac_m4.sh
./scripts/train_mac_m4.sh
```
*(Or run manually inside venv: `export PYTHONPATH="." && python -m pipeline.train_tft && python pipeline/plot_results.py`)*

### Step 3: Inspect Real Benchmark Results
Inspect the generated output file:
```bash
cat data/benchmark_results.json
```
Verify the computed metrics:
* Statistical: RMSE, MASE, Quantile Pinball Loss ($q \in \{0.10, 0.50, 0.90\}$).
* Economic: Budget Utilization Rate ($BUR$), Premature Depletion Rate ($PDR$), Pacing Volatility Index ($PVI$).

### Step 4: Sync Real Numbers into the ACM LaTeX Paper
1. Open `paper/main.tex`.
2. In `Table 1` (`\label{tab:main_results}`) and the text of Section 6.1, update any placeholder numbers with the exact values from `data/benchmark_results.json`.
3. Recompile the LaTeX manuscript:
```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
4. Verify the updated PDF at `paper/main.pdf`.

### Step 5: Commit & Push Authentic Artifacts
```bash
git add data/benchmark_results.json plots/ paper/
git commit -m "docs: populate genuine empirical M4 benchmark metrics and publication plots"
git push origin master
```

---

## 5. File Map Quick Reference
* Model definition: `pipeline/models/tft_model.py`
* Baselines (ULP, PID, LightGBM): `pipeline/baselines.py`
* Pacing simulator: `pipeline/pacing_simulation.py`
* Metrics evaluator: `pipeline/evaluate.py`
* Training script: `pipeline/train_tft.py`
* Plotting engine: `pipeline/plot_results.py`
* LaTeX manuscript: `paper/main.tex`
* LaTeX bibliography: `paper/references.bib`
* Compiled publication PDF: `paper/main.pdf`
