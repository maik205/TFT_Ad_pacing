"""
Comprehensive Stress-testing and Regime-specific Benchmarking Suite.
Evaluates TFT vs. Lagged LightGBM and Classical Baselines (ULP, PID) across 5 DSP operational regimes:
1. Cold-Start Regime (New campaign launches with 0-24h history)
2. Pipeline Dropout Regime (Ad exchange logging delays / 12h data blackouts)
3. Extended Horizon Compounding Regime (12h, 24h, 48h horizons with variance collapse analysis)
4. Cross-Campaign Generalization (Zero-shot transfer on held-out advertiser vertical)
5. Quantile Monotonicity & Risk Distortion Analysis (Crossing rates and pacing solvency)
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
from pipeline.train_tft import HourlyPacingDataset, eval_tft, train_tft_epoch, validate_tft_epoch
from pipeline.pacing_simulation import PacingSimulationEngine
from pipeline.evaluate import evaluate_statistical_metrics, evaluate_economic_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_stress_benchmarks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using compute device: {device}")
    output_dir = "data"
    
    # Load dataset splits
    train_df = pd.read_parquet(os.path.join(output_dir, "train_hourly.parquet"))
    val_df = pd.read_parquet(os.path.join(output_dir, "val_hourly.parquet"))
    test_df = pd.read_parquet(os.path.join(output_dir, "test_hourly.parquet"))
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 1. Train Full LightGBM baseline
    logger.info("Fitting Full Lagged LightGBM baseline...")
    lgb_baseline = LaggedLightGBM(quantiles=[0.10, 0.50, 0.90])
    X_all, y_all, df_clean = lgb_baseline.create_features(full_df)
    train_mask = df_clean['time_idx'] < 1440
    val_mask = (df_clean['time_idx'] >= 1440) & (df_clean['time_idx'] < 1800)
    lgb_baseline.fit(X_all[train_mask], y_all[train_mask], X_all[val_mask], y_all[val_mask])

    # 2. Load Pretrained Full TFT Model
    logger.info("Loading Pretrained Full TFT Model...")
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
    sim_engine = PacingSimulationEngine(alpha_min=0.05, horizon=24, risk_hedging_ratio=0.70)
    results = {}

    past_cols = ['hourly_spend', 'impressions_won', 'win_rate', 'avg_cpm', 'spend_ratio']
    future_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'countdown_ratio']
    static_cols = ['total_budget', 'nominal_velocity', 'duration_hours']

    # =========================================================================
    # EXPERIMENT 1: Cold-Start Regime (Hour 0 to Hour 24 of new campaigns)
    # Realistic condition: An advertiser launches a new campaign with 0 history.
    # We evaluate both statistical forecasting and economic pacing delivery.
    # =========================================================================
    logger.info("\n=== EXPERIMENT 1: Cold-Start Launch Regime (0-24h) ===")
    lgb_cs_targets, lgb_cs_q10, lgb_cs_q50, lgb_cs_q90 = [], [], [], []
    tft_cs_samples = []
    cs_campaign_dfs = []

    for cid in campaigns:
        c_full = full_df[full_df['campaign_id'] == cid].sort_values('time_idx').reset_index(drop=True)
        c_first_24 = c_full.iloc[0:24].copy()
        y_true_cs = c_first_24['hourly_spend'].values
        cs_campaign_dfs.append(c_first_24)

        # LightGBM cold-start: history is empty (zeros)
        empty_history = [0.0] * 168
        w_preds = lgb_baseline.predict_multi_step_recursive(empty_history, c_first_24, horizon=24)
        lgb_cs_targets.append(y_true_cs)
        lgb_cs_q10.append(w_preds[0.10])
        lgb_cs_q50.append(w_preds[0.50])
        lgb_cs_q90.append(w_preds[0.90])

        # TFT cold-start: zero past observations, but valid static context and known future
        zero_past = np.zeros((168, len(past_cols)), dtype=np.float32)
        zero_past[:, 0] = (0.0 - spend_mean) / spend_std
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

    lgb_cs_true = np.array(lgb_cs_targets, dtype=np.float32)
    lgb_cs_preds = {
        0.10: np.array(lgb_cs_q10, dtype=np.float32),
        0.50: np.array(lgb_cs_q50, dtype=np.float32),
        0.90: np.array(lgb_cs_q90, dtype=np.float32)
    }
    lgb_cs_metrics = evaluate_statistical_metrics(lgb_cs_true, lgb_cs_preds)

    with torch.no_grad():
        tft_q10_list, tft_q50_list, tft_q90_list, tft_tgt_list = [], [], [], []
        for s in tft_cs_samples:
            preds, _ = tft_model(s['past_x'].to(device), s['future_x'].to(device), s['static_x'].to(device))
            p = preds.cpu().numpy()[0]
            tft_q10_list.append(np.maximum(0.0, p[:, 0] * spend_std + spend_mean))
            tft_q50_list.append(np.maximum(0.0, p[:, 1] * spend_std + spend_mean))
            tft_q90_list.append(np.maximum(0.0, p[:, 2] * spend_std + spend_mean))
            tft_tgt_list.append(s['y_target'].numpy()[0])

    tft_cs_true = np.array(tft_tgt_list, dtype=np.float32)
    tft_cs_preds = {
        0.10: np.array(tft_q10_list, dtype=np.float32),
        0.50: np.array(tft_q50_list, dtype=np.float32),
        0.90: np.array(tft_q90_list, dtype=np.float32)
    }
    tft_cs_metrics = evaluate_statistical_metrics(tft_cs_true, tft_cs_preds)

    # Economic Pacing Simulation during Cold Start (24h Flight)
    ulp_cs_sims, pid_cs_sims, lgb_cs_sims, tft_cs_sims = [], [], [], []
    for i, c_df in enumerate(cs_campaign_dfs):
        ulp_cs_sims.append(sim_engine.simulate_linear_pacing(c_df))
        pid_cs_sims.append(sim_engine.simulate_pid_pacing(c_df))
        lgb_cs_sims.append(sim_engine.simulate_tft_pacing(
            c_df, lgb_cs_preds[0.10][i], lgb_cs_preds[0.50][i], lgb_cs_preds[0.90][i],
            policy_name="Cold-Start LightGBM"
        ))
        tft_cs_sims.append(sim_engine.simulate_tft_pacing(
            c_df, tft_cs_preds[0.10][i], tft_cs_preds[0.50][i], tft_cs_preds[0.90][i],
            policy_name="Cold-Start TFT"
        ))

    results['cold_start'] = {
        'statistical': {
            'lightgbm': lgb_cs_metrics,
            'tft': tft_cs_metrics
        },
        'economic': {
            'ulp': evaluate_economic_metrics(ulp_cs_sims),
            'pid': evaluate_economic_metrics(pid_cs_sims),
            'lightgbm': evaluate_economic_metrics(lgb_cs_sims),
            'tft': evaluate_economic_metrics(tft_cs_sims)
        }
    }
    logger.info(f"Cold-Start Economic Results: TFT BUR={results['cold_start']['economic']['tft']['budget_utilization_rate']}%, PVI={results['cold_start']['economic']['tft']['pacing_volatility_index']} vs LightGBM BUR={results['cold_start']['economic']['lightgbm']['budget_utilization_rate']}%, PVI={results['cold_start']['economic']['lightgbm']['pacing_volatility_index']}")

    # =========================================================================
    # EXPERIMENT 2: Pipeline Dropout Regime (12h Ingestion Blackout)
    # Realistic condition: Ad exchange logging pipeline drops the last 12 hours (t-12 to t)
    # =========================================================================
    logger.info("\n=== EXPERIMENT 2: Pipeline Dropout Regime (12h Blackout) ===")
    lgb_do_targets, lgb_do_q10, lgb_do_q50, lgb_do_q90 = [], [], [], []
    tft_do_samples = []
    do_window_dfs = []

    for cid in campaigns:
        c_val = val_df[val_df['campaign_id'] == cid].sort_values('time_idx').iloc[-168:]
        c_test = test_df[test_df['campaign_id'] == cid].sort_values('time_idx')
        c_combined = pd.concat([c_val, c_test], ignore_index=True)

        for w in range(5):  # Sample first 5 test days per campaign
            start = w * 24
            hist_df = c_combined.iloc[start : start + 168].copy()
            fut_df = c_combined.iloc[start + 168 : start + 168 + 24].copy()
            y_tgt = fut_df['hourly_spend'].values
            do_window_dfs.append(fut_df)

            # Corrupt last 12h history
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
            past_mat[-12:, 0] = 0.0
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
            preds, _ = tft_model(s['past_x'].to(device), s['future_x'].to(device), s['static_x'].to(device))
            p = preds.cpu().numpy()[0]
            tft_q10_do.append(np.maximum(0.0, p[:, 0] * spend_std + spend_mean))
            tft_q50_do.append(np.maximum(0.0, p[:, 1] * spend_std + spend_mean))
            tft_q90_do.append(np.maximum(0.0, p[:, 2] * spend_std + spend_mean))
            tft_tgt_do.append(s['y_target'].numpy()[0])

    tft_do_true = np.array(tft_tgt_do, dtype=np.float32)
    tft_do_preds = {
        0.10: np.array(tft_q10_do, dtype=np.float32),
        0.50: np.array(tft_q50_do, dtype=np.float32),
        0.90: np.array(tft_q90_do, dtype=np.float32)
    }
    tft_do_metrics = evaluate_statistical_metrics(tft_do_true, tft_do_preds)

    # Economic Pacing Simulation following 12h Dropout
    lgb_do_sims, tft_do_sims = [], []
    for i, w_df in enumerate(do_window_dfs):
        lgb_do_sims.append(sim_engine.simulate_tft_pacing(
            w_df, lgb_do_preds[0.10][i], lgb_do_preds[0.50][i], lgb_do_preds[0.90][i],
            policy_name="Dropout LightGBM"
        ))
        tft_do_sims.append(sim_engine.simulate_tft_pacing(
            w_df, tft_do_preds[0.10][i], tft_do_preds[0.50][i], tft_do_preds[0.90][i],
            policy_name="Dropout TFT"
        ))

    results['pipeline_dropout_12h'] = {
        'statistical': {
            'lightgbm': lgb_do_metrics,
            'tft': tft_do_metrics
        },
        'economic': {
            'lightgbm': evaluate_economic_metrics(lgb_do_sims),
            'tft': evaluate_economic_metrics(tft_do_sims)
        }
    }
    logger.info(f"Dropout Economic Results: TFT BUR={results['pipeline_dropout_12h']['economic']['tft']['budget_utilization_rate']}%, PVI={results['pipeline_dropout_12h']['economic']['tft']['pacing_volatility_index']} vs LightGBM BUR={results['pipeline_dropout_12h']['economic']['lightgbm']['budget_utilization_rate']}%, PVI={results['pipeline_dropout_12h']['economic']['lightgbm']['pacing_volatility_index']}")

    # =========================================================================
    # EXPERIMENT 3: Multi-Horizon Compounding Regime (12h, 24h, 48h)
    # Compares forecast divergence and variance collapse across extended planning horizons
    # =========================================================================
    logger.info("\n=== EXPERIMENT 3: Multi-Horizon Compounding Regime (12h, 24h, 48h) ===")
    results['multi_horizon_compounding'] = {}

    horizons = [12, 24, 48]
    for H in horizons:
        logger.info(f"Evaluating planning horizon H={H}h...")
        lgb_h_targets, lgb_h_q10, lgb_h_q50, lgb_h_q90 = [], [], [], []
        tft_h_targets, tft_h_q10, tft_h_q50, tft_h_q90 = [], [], [], []

        for cid in campaigns:
            c_val = val_df[val_df['campaign_id'] == cid].sort_values('time_idx').iloc[-168:]
            c_test = test_df[test_df['campaign_id'] == cid].sort_values('time_idx')
            c_combined = pd.concat([c_val, c_test], ignore_index=True)

            # Sample 4 non-overlapping windows that fit horizon H
            max_windows = min(4, (len(c_test) - H) // H)
            for w in range(max(1, max_windows)):
                start = w * H
                hist_df = c_combined.iloc[start : start + 168]
                fut_df = c_combined.iloc[start + 168 : start + 168 + H]
                if len(fut_df) < H:
                    continue
                y_tgt = fut_df['hourly_spend'].values

                # LightGBM multi-step recursive forecast for H steps
                w_preds = lgb_baseline.predict_multi_step_recursive(
                    hist_df['hourly_spend'].tolist(), fut_df, horizon=H
                )
                lgb_h_targets.append(y_tgt)
                lgb_h_q10.append(w_preds[0.10])
                lgb_h_q50.append(w_preds[0.50])
                lgb_h_q90.append(w_preds[0.90])

                # TFT forecast for H steps
                if H <= 24:
                    # Slice native 24h forecast to H steps
                    past_mat = hist_df[past_cols].values.astype(np.float32)
                    past_mat[:, 0] = (past_mat[:, 0] - spend_mean) / spend_std
                    
                    # Pad future to 24 steps if H < 24 for TFT input compatibility
                    fut_padded = c_combined.iloc[start + 168 : start + 168 + 24]
                    future_mat = fut_padded[future_cols].values.astype(np.float32)
                    static_vec = fut_padded.iloc[0][static_cols].values.astype(np.float32)
                    static_vec[0] /= 1e5
                    static_vec[1] /= 1e2
                    static_vec[2] /= 1e3

                    with torch.no_grad():
                        px = torch.tensor(past_mat).unsqueeze(0).to(device)
                        fx = torch.tensor(future_mat).unsqueeze(0).to(device)
                        sx = torch.tensor(static_vec).unsqueeze(0).to(device)
                        preds, _ = tft_model(px, fx, sx)
                        p = preds.cpu().numpy()[0]
                        p_unnorm = np.maximum(0.0, p * spend_std + spend_mean)
                        tft_h_targets.append(y_tgt)
                        tft_h_q10.append(p_unnorm[:H, 0])
                        tft_h_q50.append(p_unnorm[:H, 1])
                        tft_h_q90.append(p_unnorm[:H, 2])
                else:
                    # H = 48: Autoregressive 2-stage 24h roll-forward
                    fut_24a = c_combined.iloc[start + 168 : start + 168 + 24]
                    fut_24b = c_combined.iloc[start + 168 + 24 : start + 168 + 48]
                    
                    past_mat_a = hist_df[past_cols].values.astype(np.float32)
                    past_mat_a[:, 0] = (past_mat_a[:, 0] - spend_mean) / spend_std
                    future_mat_a = fut_24a[future_cols].values.astype(np.float32)
                    static_vec = fut_24a.iloc[0][static_cols].values.astype(np.float32)
                    static_vec[0] /= 1e5
                    static_vec[1] /= 1e2
                    static_vec[2] /= 1e3

                    with torch.no_grad():
                        px_a = torch.tensor(past_mat_a).unsqueeze(0).to(device)
                        fx_a = torch.tensor(future_mat_a).unsqueeze(0).to(device)
                        sx = torch.tensor(static_vec).unsqueeze(0).to(device)
                        preds_a, _ = tft_model(px_a, fx_a, sx)
                        p_a = preds_a.cpu().numpy()[0]
                        p_unnorm_a = np.maximum(0.0, p_a * spend_std + spend_mean)

                        # Construct stage 2 past by rolling 24 steps forward
                        stage2_spend = list(hist_df['hourly_spend'].values[24:]) + list(p_unnorm_a[:, 1])
                        past_mat_b = np.zeros((168, len(past_cols)), dtype=np.float32)
                        past_mat_b[:, 0] = (np.array(stage2_spend) - spend_mean) / spend_std
                        # Carry forward other past columns
                        past_mat_b[:, 1:] = np.vstack([
                            hist_df[past_cols[1:]].values[24:],
                            np.tile(hist_df[past_cols[1:]].values[-1:], (24, 1))
                        ])
                        future_mat_b = fut_24b[future_cols].values.astype(np.float32)

                        px_b = torch.tensor(past_mat_b).unsqueeze(0).to(device)
                        fx_b = torch.tensor(future_mat_b).unsqueeze(0).to(device)
                        preds_b, _ = tft_model(px_b, fx_b, sx)
                        p_b = preds_b.cpu().numpy()[0]
                        p_unnorm_b = np.maximum(0.0, p_b * spend_std + spend_mean)

                        comb_q10 = np.concatenate([p_unnorm_a[:, 0], p_unnorm_b[:, 0]])
                        comb_q50 = np.concatenate([p_unnorm_a[:, 1], p_unnorm_b[:, 1]])
                        comb_q90 = np.concatenate([p_unnorm_a[:, 2], p_unnorm_b[:, 2]])

                        tft_h_targets.append(y_tgt)
                        tft_h_q10.append(comb_q10)
                        tft_h_q50.append(comb_q50)
                        tft_h_q90.append(comb_q90)

        # Statistical evaluation for horizon H
        lgb_h_arr = np.array(lgb_h_targets, dtype=np.float32)
        lgb_preds_dict = {
            0.10: np.array(lgb_h_q10, dtype=np.float32),
            0.50: np.array(lgb_h_q50, dtype=np.float32),
            0.90: np.array(lgb_h_q90, dtype=np.float32)
        }
        lgb_h_stat = evaluate_statistical_metrics(lgb_h_arr, lgb_preds_dict, seasonality=min(24, H))

        tft_h_arr = np.array(tft_h_targets, dtype=np.float32)
        tft_preds_dict = {
            0.10: np.array(tft_h_q10, dtype=np.float32),
            0.50: np.array(tft_h_q50, dtype=np.float32),
            0.90: np.array(tft_h_q90, dtype=np.float32)
        }
        tft_h_stat = evaluate_statistical_metrics(tft_h_arr, tft_preds_dict, seasonality=min(24, H))

        # Variance Preservation Ratio: Var(pred) / Var(true)
        true_var = float(np.var(lgb_h_arr))
        lgb_var_ratio = round(float(np.var(lgb_preds_dict[0.50]) / (true_var + 1e-6)), 4)
        tft_var_ratio = round(float(np.var(tft_preds_dict[0.50]) / (true_var + 1e-6)), 4)

        logger.info(f"H={H}h Metrics: LightGBM RMSE={lgb_h_stat['rmse']}, VarRatio={lgb_var_ratio} | TFT RMSE={tft_h_stat['rmse']}, VarRatio={tft_var_ratio}")
        results['multi_horizon_compounding'][f'H_{H}h'] = {
            'lightgbm': {**lgb_h_stat, 'variance_preservation_ratio': lgb_var_ratio},
            'tft': {**tft_h_stat, 'variance_preservation_ratio': tft_var_ratio}
        }

    # =========================================================================
    # EXPERIMENT 4: Cross-Campaign Zero-Shot Generalization
    # Train on 8 campaigns, evaluate zero-shot on held-out 9th campaign (937622)
    # =========================================================================
    logger.info("\n=== EXPERIMENT 4: Cross-Campaign Zero-Shot Generalization ===")
    held_out_cid = '937622'
    train_8_df = train_df[train_df['campaign_id'] != held_out_cid].copy()
    val_8_df = val_df[val_df['campaign_id'] != held_out_cid].copy()
    test_held_df = test_df[test_df['campaign_id'] == held_out_cid].copy()
    full_8_df = pd.concat([train_8_df, val_8_df], ignore_index=True)

    logger.info(f"Holding out campaign {held_out_cid}. Training zero-shot LightGBM on 8 campaigns...")
    lgb_zero = LaggedLightGBM(quantiles=[0.10, 0.50, 0.90])
    X_8, y_8, df_8_clean = lgb_zero.create_features(full_8_df)
    train_8_mask = df_8_clean['time_idx'] < 1440
    val_8_mask = df_8_clean['time_idx'] >= 1440
    lgb_zero.fit(X_8[train_8_mask], y_8[train_8_mask], X_8[val_8_mask], y_8[val_8_mask])

    logger.info("Training zero-shot TFT model on 8 campaigns (15 epochs with cosine annealing)...")
    ds_train_8 = HourlyPacingDataset(train_8_df, lookback=168, horizon=24, is_train=True)
    ds_val_8 = HourlyPacingDataset(val_8_df, lookback=168, horizon=24, is_train=False)
    loader_train_8 = DataLoader(ds_train_8, batch_size=64, shuffle=True)
    loader_val_8 = DataLoader(ds_val_8, batch_size=64, shuffle=False)

    tft_zero = TemporalFusionTransformer(
        num_past_features=5, num_future_features=6, num_static_features=3,
        d_model=64, n_heads=4, lookback=168, horizon=24, quantiles=[0.10, 0.50, 0.90], dropout=0.1
    ).to(device)

    opt_zero = torch.optim.AdamW(tft_zero.parameters(), lr=0.001, weight_decay=1e-4)
    sched_zero = torch.optim.lr_scheduler.CosineAnnealingLR(opt_zero, T_max=15, eta_min=1e-5)

    best_val_loss = float('inf')
    best_weights = None
    for ep in range(15):
        train_loss = train_tft_epoch(tft_zero, loader_train_8, opt_zero, device, ds_train_8.spend_mean, ds_train_8.spend_std)
        val_loss = validate_tft_epoch(tft_zero, loader_val_8, device, ds_train_8.spend_mean, ds_train_8.spend_std)
        sched_zero.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in tft_zero.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 14:
            logger.info(f"Zero-shot Epoch {ep+1}/15: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    if best_weights is not None:
        tft_zero.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    tft_zero.eval()

    # Evaluate zero-shot on held-out test campaign 937622
    ds_test_held = HourlyPacingDataset(test_held_df, lookback=168, horizon=24, is_train=False)
    loader_test_held = DataLoader(ds_test_held, batch_size=32, shuffle=False)
    tft_zero_eval = eval_tft(tft_zero, loader_test_held, device, ds_train_8.spend_mean, ds_train_8.spend_std)
    tft_zero_stat = evaluate_statistical_metrics(
        tft_zero_eval['targets'],
        {q: tft_zero_eval[q] for q in [0.10, 0.50, 0.90]}
    )

    # Evaluate LightGBM on held-out test campaign 937622
    c_val_held = val_df[val_df['campaign_id'] == held_out_cid].sort_values('time_idx').iloc[-168:]
    c_test_held = test_held_df.sort_values('time_idx')
    c_comb_held = pd.concat([c_val_held, c_test_held], ignore_index=True)

    lgb_zero_targets, lgb_zero_q10, lgb_zero_q50, lgb_zero_q90 = [], [], [], []
    num_windows_held = (len(c_test_held) - 24) // 24
    for w in range(num_windows_held):
        start = w * 24
        hist_df = c_comb_held.iloc[start : start + 168]
        fut_df = c_comb_held.iloc[start + 168 : start + 168 + 24]
        w_preds = lgb_zero.predict_multi_step_recursive(hist_df['hourly_spend'].tolist(), fut_df, horizon=24)
        lgb_zero_targets.append(fut_df['hourly_spend'].values)
        lgb_zero_q10.append(w_preds[0.10])
        lgb_zero_q50.append(w_preds[0.50])
        lgb_zero_q90.append(w_preds[0.90])

    lgb_zero_arr = np.array(lgb_zero_targets, dtype=np.float32)
    lgb_zero_dict = {
        0.10: np.array(lgb_zero_q10, dtype=np.float32),
        0.50: np.array(lgb_zero_q50, dtype=np.float32),
        0.90: np.array(lgb_zero_q90, dtype=np.float32)
    }
    lgb_zero_stat = evaluate_statistical_metrics(lgb_zero_arr, lgb_zero_dict)

    # Economic simulation on held-out campaign
    lgb_zero_sim = sim_engine.simulate_tft_pacing(
        test_held_df, lgb_zero_dict[0.10], lgb_zero_dict[0.50], lgb_zero_dict[0.90],
        policy_name="Zero-Shot LightGBM"
    )
    tft_zero_sim = sim_engine.simulate_tft_pacing(
        test_held_df, tft_zero_eval[0.10], tft_zero_eval[0.50], tft_zero_eval[0.90],
        policy_name="Zero-Shot TFT"
    )

    logger.info(f"Zero-Shot Results on Campaign {held_out_cid}:")
    logger.info(f"LightGBM: RMSE={lgb_zero_stat['rmse']}, MASE={lgb_zero_stat['mase']}, BUR={lgb_zero_sim.budget_utilization}%, PVI={lgb_zero_sim.pacing_volatility_index}")
    logger.info(f"TFT:      RMSE={tft_zero_stat['rmse']}, MASE={tft_zero_stat['mase']}, BUR={tft_zero_sim.budget_utilization}%, PVI={tft_zero_sim.pacing_volatility_index}")

    results['cross_campaign_zero_shot'] = {
        'held_out_campaign_id': held_out_cid,
        'lightgbm': {
            'statistical': lgb_zero_stat,
            'economic': {
                'budget_utilization_rate': lgb_zero_sim.budget_utilization,
                'pacing_volatility_index': lgb_zero_sim.pacing_volatility_index,
                'is_prematurely_depleted': lgb_zero_sim.is_prematurely_depleted
            }
        },
        'tft': {
            'statistical': tft_zero_stat,
            'economic': {
                'budget_utilization_rate': tft_zero_sim.budget_utilization,
                'pacing_volatility_index': tft_zero_sim.pacing_volatility_index,
                'is_prematurely_depleted': tft_zero_sim.is_prematurely_depleted
            }
        }
    }

    # =========================================================================
    # EXPERIMENT 5: Quantile Monotonicity & Risk Distortion Analysis
    # Evaluates non-monotonic quantile crossing rates on out-of-sample test partition
    # =========================================================================
    logger.info("\n=== EXPERIMENT 5: Quantile Monotonicity Analysis ===")
    test_ds_full = HourlyPacingDataset(test_df, lookback=168, horizon=24, is_train=False)
    test_loader_full = DataLoader(test_ds_full, batch_size=32, shuffle=False)
    tft_test_res = eval_tft(tft_model, test_loader_full, device, spend_mean, spend_std)

    tft_q10_flat = tft_test_res[0.10].flatten()
    tft_q50_flat = tft_test_res[0.50].flatten()
    tft_q90_flat = tft_test_res[0.90].flatten()

    tft_cross_10_50 = int(np.sum(tft_q10_flat > tft_q50_flat))
    tft_cross_50_90 = int(np.sum(tft_q50_flat > tft_q90_flat))
    tft_total_points = len(tft_q10_flat)

    # LightGBM crossing on full test mask
    test_clean_mask = df_clean['time_idx'] >= 1800
    lgb_test_preds = lgb_baseline.predict_quantiles(X_all[test_clean_mask])
    lgb_q10_flat = lgb_test_preds[0.10]
    lgb_q50_flat = lgb_test_preds[0.50]
    lgb_q90_flat = lgb_test_preds[0.90]

    lgb_cross_10_50 = int(np.sum(lgb_q10_flat > lgb_q50_flat))
    lgb_cross_50_90 = int(np.sum(lgb_q50_flat > lgb_q90_flat))
    lgb_total_points = len(lgb_q10_flat)

    logger.info(f"Quantile Crossing Rates on Test Set:")
    logger.info(f"LightGBM: q10 > q50: {lgb_cross_10_50}/{lgb_total_points} ({lgb_cross_10_50/lgb_total_points*100:.2f}%), q50 > q90: {lgb_cross_50_90}/{lgb_total_points} ({lgb_cross_50_90/lgb_total_points*100:.2f}%)")
    logger.info(f"TFT:      q10 > q50: {tft_cross_10_50}/{tft_total_points} ({tft_cross_10_50/tft_total_points*100:.2f}%), q50 > q90: {tft_cross_50_90}/{tft_total_points} ({tft_cross_50_90/tft_total_points*100:.2f}%)")

    results['quantile_monotonicity'] = {
        'lightgbm': {
            'total_points': lgb_total_points,
            'crossings_q10_gt_q50': lgb_cross_10_50,
            'rate_q10_gt_q50_pct': round(lgb_cross_10_50 / lgb_total_points * 100.0, 2),
            'crossings_q50_gt_q90': lgb_cross_50_90,
            'rate_q50_gt_q90_pct': round(lgb_cross_50_90 / lgb_total_points * 100.0, 2)
        },
        'tft': {
            'total_points': tft_total_points,
            'crossings_q10_gt_q50': tft_cross_10_50,
            'rate_q10_gt_q50_pct': round(tft_cross_10_50 / tft_total_points * 100.0, 2),
            'crossings_q50_gt_q90': tft_cross_50_90,
            'rate_q50_gt_q90_pct': round(tft_cross_50_90 / tft_total_points * 100.0, 2)
        }
    }

    # Save comprehensive results to JSON
    out_file = os.path.join(output_dir, "stress_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSuccessfully executed all 5 stress regimes! Saved comprehensive benchmark results to {out_file}")


if __name__ == '__main__':
    run_stress_benchmarks()
