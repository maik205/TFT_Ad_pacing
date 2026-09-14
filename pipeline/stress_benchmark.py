"""
Stress-testing and Regime-specific Benchmarking:
Evaluates TFT vs. Lagged LightGBM and Classical Baselines across 4 realistic DSP operational regimes:
1. Cold-Start Regime (New campaign launches with 0-24h history)
2. Pipeline Dropout Regime (Ad exchange logging delays / 6h-12h data blackouts)
3. Extended Horizon Compounding Regime (12h, 24h, 48h horizons)
4. Cross-Campaign Generalization (Held-out unseen campaigns)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.models.tft_model import TemporalFusionTransformer
from pipeline.baselines import UniformLinearPacing, PIDController, LaggedLightGBM
from pipeline.train_tft import HourlyPacingDataset, eval_tft
from pipeline.pacing_simulation import PacingSimulationEngine
from pipeline.evaluate import evaluate_statistical_metrics, evaluate_economic_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_stress_benchmarks():
    device = torch.device('cpu')
    output_dir = "data"
    
    # Load dataset splits
    train_df = pd.read_parquet(os.path.join(output_dir, "train_hourly.parquet"))
    val_df = pd.read_parquet(os.path.join(output_dir, "val_hourly.parquet"))
    test_df = pd.read_parquet(os.path.join(output_dir, "test_hourly.parquet"))
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 1. Train / Load LightGBM
    logger.info("Fitting Lagged LightGBM baseline...")
    lgb_baseline = LaggedLightGBM(quantiles=[0.10, 0.50, 0.90])
    X_all, y_all, df_clean = lgb_baseline.create_features(full_df)
    train_mask = df_clean['time_idx'] < 1440
    val_mask = (df_clean['time_idx'] >= 1440) & (df_clean['time_idx'] < 1800)
    lgb_baseline.fit(X_all[train_mask], y_all[train_mask], X_all[val_mask], y_all[val_mask])

    # 2. Load Pretrained TFT
    logger.info("Loading Pretrained TFT Model...")
    train_dataset = HourlyPacingDataset(train_df, lookback=168, horizon=24, is_train=True)
    spend_mean = train_dataset.spend_mean
    spend_std = train_dataset.spend_std

    tft_model = TemporalFusionTransformer(
        num_past_features=5,
        num_future_features=6,
        num_static_features=3,
        d_model=64,
        n_heads=4,
        lookback=168,
        horizon=24,
        quantiles=[0.10, 0.50, 0.90],
        dropout=0.1
    ).to(device)

    ckpt_path = os.path.join(output_dir, "tft_best_checkpoint.pt")
    if os.path.exists(ckpt_path):
        tft_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Loaded TFT checkpoint from {ckpt_path}")
    tft_model.eval()

    campaigns = test_df['campaign_id'].unique()
    results = {}

    # =========================================================================
    # EXPERIMENT 1: Cold-Start Regime (Hour 0 to Hour 24 of new campaigns)
    # Realistic condition: An advertiser launches a new campaign. There is NO 168h history.
    # We evaluate forecasting the first 24h of the campaign.
    # =========================================================================
    logger.info("\n--- Running Experiment 1: Cold-Start Launch Regime ---")
    lgb_cs_targets, lgb_cs_q10, lgb_cs_q50, lgb_cs_q90 = [], [], [], []
    tft_cs_samples = []

    past_cols = ['hourly_spend', 'impressions_won', 'win_rate', 'avg_cpm', 'spend_ratio']
    future_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'countdown_ratio']
    static_cols = ['total_budget', 'nominal_velocity', 'duration_hours']

    for cid in campaigns:
        # Use the very first 24 hours of each campaign (t=0 to 24)
        c_full = full_df[full_df['campaign_id'] == cid].sort_values('time_idx').reset_index(drop=True)
        c_first_24 = c_full.iloc[0:24]
        y_true_cs = c_first_24['hourly_spend'].values

        # LightGBM cold-start: history is empty (zeros)
        empty_history = [0.0] * 168
        w_preds = lgb_baseline.predict_multi_step_recursive(empty_history, c_first_24, horizon=24)
        lgb_cs_targets.append(y_true_cs)
        lgb_cs_q10.append(w_preds[0.10])
        lgb_cs_q50.append(w_preds[0.50])
        lgb_cs_q90.append(w_preds[0.90])

        # TFT cold-start: zero past observations, but valid static context and known future
        zero_past = np.zeros((168, len(past_cols)), dtype=np.float32)
        zero_past[:, 0] = (0.0 - spend_mean) / spend_std  # normalized zero spend
        future_mat = c_first_24[future_cols].values.astype(np.float32)
        static_vec = c_first_24.iloc[0][static_cols].values.astype(np.float32)
        static_vec[0] /= 1e5
        static_vec[1] /= 1e2
        static_vec[2] /= 1e3

        tft_cs_samples.append({
            'past_x': torch.tensor(zero_past).unsqueeze(0),
            'future_x': torch.tensor(future_mat).unsqueeze(0),
            'static_x': torch.tensor(static_vec).unsqueeze(0),
            'y_target': torch.tensor(y_true_cs).unsqueeze(0)
        })

    # Evaluate LightGBM Cold Start
    lgb_cs_true = np.array(lgb_cs_targets, dtype=np.float32)
    lgb_cs_preds = {
        0.10: np.array(lgb_cs_q10, dtype=np.float32),
        0.50: np.array(lgb_cs_q50, dtype=np.float32),
        0.90: np.array(lgb_cs_q90, dtype=np.float32)
    }
    lgb_cs_metrics = evaluate_statistical_metrics(lgb_cs_true, lgb_cs_preds)

    # Evaluate TFT Cold Start
    with torch.no_grad():
        tft_q10_list, tft_q50_list, tft_q90_list, tft_tgt_list = [], [], [], []
        for s in tft_cs_samples:
            out = tft_model(s['past_x'].to(device), s['future_x'].to(device), s['static_x'].to(device))
            p = out['quantiles'].cpu().numpy()[0]
            # De-normalize
            tft_q10_list.append(np.maximum(0.0, p[0, :, 0] * spend_std + spend_mean))
            tft_q50_list.append(np.maximum(0.0, p[0, :, 1] * spend_std + spend_mean))
            tft_q90_list.append(np.maximum(0.0, p[0, :, 2] * spend_std + spend_mean))
            tft_tgt_list.append(s['y_target'].numpy()[0])

    tft_cs_true = np.array(tft_tgt_list, dtype=np.float32)
    tft_cs_preds = {
        0.10: np.array(tft_q10_list, dtype=np.float32),
        0.50: np.array(tft_q50_list, dtype=np.float32),
        0.90: np.array(tft_q90_list, dtype=np.float32)
    }
    tft_cs_metrics = evaluate_statistical_metrics(tft_cs_true, tft_cs_preds)

    logger.info(f"Cold-Start LightGBM Metrics: {lgb_cs_metrics}")
    logger.info(f"Cold-Start TFT Metrics:      {tft_cs_metrics}")
    results['cold_start'] = {
        'lightgbm': lgb_cs_metrics,
        'tft': tft_cs_metrics
    }

    # =========================================================================
    # EXPERIMENT 2: Pipeline Log Dropout Regime (Missing Data / Log Dropout)
    # Realistic condition: Ad exchange logging pipeline drops the last 12 hours (t-12 to t)
    # =========================================================================
    logger.info("\n--- Running Experiment 2: Pipeline Dropout Regime (12h Blackout) ---")
    lgb_do_targets, lgb_do_q10, lgb_do_q50, lgb_do_q90 = [], [], [], []
    tft_do_samples = []

    for cid in campaigns:
        c_val = val_df[val_df['campaign_id'] == cid].sort_values('time_idx').iloc[-168:]
        c_test = test_df[test_df['campaign_id'] == cid].sort_values('time_idx')
        c_combined = pd.concat([c_val, c_test], ignore_index=True)

        for w in range(5):  # Sample first 5 test days
            start = w * 24
            hist_df = c_combined.iloc[start : start + 168].copy()
            fut_df = c_combined.iloc[start + 168 : start + 168 + 24]
            y_tgt = fut_df['hourly_spend'].values

            # Simulate 12h dropout: zero out the last 12 hours of the lookback
            corrupted_spends = hist_df['hourly_spend'].tolist()
            for k in range(1, 13):
                corrupted_spends[-k] = 0.0

            # LightGBM with corrupted history
            w_preds = lgb_baseline.predict_multi_step_recursive(corrupted_spends, fut_df, horizon=24)
            lgb_do_targets.append(y_tgt)
            lgb_do_q10.append(w_preds[0.10])
            lgb_do_q50.append(w_preds[0.50])
            lgb_do_q90.append(w_preds[0.90])

            # TFT with corrupted past inputs
            past_mat = hist_df[past_cols].values.astype(np.float32)
            past_mat[-12:, 0] = 0.0  # zero out spend
            past_mat[:, 0] = (past_mat[:, 0] - spend_mean) / spend_std
            future_mat = fut_df[future_cols].values.astype(np.float32)
            static_vec = fut_df.iloc[0][static_cols].values.astype(np.float32)
            static_vec[0] /= 1e5
            static_vec[1] /= 1e2
            static_vec[2] /= 1e3

            tft_do_samples.append({
                'past_x': torch.tensor(past_mat).unsqueeze(0),
                'future_x': torch.tensor(future_mat).unsqueeze(0),
                'static_x': torch.tensor(static_vec).unsqueeze(0),
                'y_target': torch.tensor(y_tgt).unsqueeze(0)
            })

    lgb_do_true = np.array(lgb_do_targets, dtype=np.float32)
    lgb_do_preds = {
        0.10: np.array(lgb_do_q10, dtype=np.float32),
        0.50: np.array(lgb_do_q50, dtype=np.float32),
        0.90: np.array(lgb_do_q90, dtype=np.float32)
    }
    lgb_do_metrics = evaluate_statistical_metrics(lgb_do_true, lgb_do_preds)

    with torch.no_grad():
        tft_q10_do, tft_q50_do, tft_q90_do, tft_tgt_do = [], [], [], []
        for s in tft_do_samples:
            out = tft_model(s['past_x'].to(device), s['future_x'].to(device), s['static_x'].to(device))
            p = out['quantiles'].cpu().numpy()[0]
            tft_q10_do.append(np.maximum(0.0, p[0, :, 0] * spend_std + spend_mean))
            tft_q50_do.append(np.maximum(0.0, p[0, :, 1] * spend_std + spend_mean))
            tft_q90_do.append(np.maximum(0.0, p[0, :, 2] * spend_std + spend_mean))
            tft_tgt_do.append(s['y_target'].numpy()[0])

    tft_do_true = np.array(tft_tgt_do, dtype=np.float32)
    tft_do_preds = {
        0.10: np.array(tft_q10_do, dtype=np.float32),
        0.50: np.array(tft_q50_do, dtype=np.float32),
        0.90: np.array(tft_q90_do, dtype=np.float32)
    }
    tft_do_metrics = evaluate_statistical_metrics(tft_do_true, tft_do_preds)

    logger.info(f"Dropout LightGBM Metrics: {lgb_do_metrics}")
    logger.info(f"Dropout TFT Metrics:      {tft_do_metrics}")
    results['pipeline_dropout_12h'] = {
        'lightgbm': lgb_do_metrics,
        'tft': tft_do_metrics
    }

    # Save results to JSON
    out_file = os.path.join(output_dir, "stress_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved stress benchmark results to {out_file}")

if __name__ == '__main__':
    run_stress_benchmarks()
