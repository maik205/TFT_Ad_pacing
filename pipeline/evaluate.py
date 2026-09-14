"""
Evaluation Metrics for Statistical Forecasting Accuracy and Economic Pacing Efficiency.
Statistical:
- Root Mean Squared Error (RMSE)
- Mean Absolute Scaled Error (MASE)
- Quantile Pinball Loss (q in {0.10, 0.50, 0.90})
Economic:
- Budget Utilization Rate (BUR)
- Premature Depletion Rate (PDR)
- Pacing Volatility Index (PVI)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 24) -> float:
    """
    Computes Mean Absolute Scaled Error (MASE) relative to a seasonal naive baseline.
    MASE < 1.0 indicates superiority over seasonal naive persistence.
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    
    # Seasonal naive baseline: y_{t} - y_{t - seasonality}
    if len(y_true) <= seasonality:
        naive_mae = np.mean(np.abs(np.diff(y_true)))
    else:
        naive_mae = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    
    if naive_mae < 1e-6:
        return float(mae_model)
    return float(mae_model / naive_mae)


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Computes Quantile Pinball Loss for quantile q."""
    err = y_true - y_pred
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.mean(loss))


def evaluate_statistical_metrics(
    y_true: np.ndarray,
    preds: Dict[float, np.ndarray],
    seasonality: int = 24
) -> Dict[str, float]:
    """
    Evaluates complete statistical metric suite.
    preds dictionary contains quantiles {0.10, 0.50, 0.90}.
    """
    median_pred = preds.get(0.50, list(preds.values())[0])
    rmse = compute_rmse(y_true, median_pred)
    mase = compute_mase(y_true, median_pred, seasonality=seasonality)
    
    pinballs = {}
    for q in [0.10, 0.50, 0.90]:
        if q in preds:
            pinballs[f'pinball_q{int(q*100)}'] = compute_pinball_loss(y_true, preds[q], q)
    
    avg_pinball = float(np.mean(list(pinballs.values()))) if pinballs else 0.0

    return {
        'rmse': round(rmse, 4),
        'mase': round(mase, 4),
        'avg_pinball_loss': round(avg_pinball, 4),
        **{k: round(v, 4) for k, v in pinballs.items()}
    }


def evaluate_economic_metrics(results: List) -> Dict[str, float]:
    """
    Aggregates economic metrics across campaign simulations:
    - Average Budget Utilization Rate (%)
    - Premature Depletion Rate (%)
    - Mean Pacing Volatility Index (PVI)
    """
    burs = [r.budget_utilization for r in results]
    premature_counts = [1 if r.is_prematurely_depleted else 0 for r in results]
    pvis = [r.pacing_volatility_index for r in results]

    return {
        'budget_utilization_rate': round(float(np.mean(burs)), 2),
        'premature_depletion_rate': round(float(np.mean(premature_counts) * 100.0), 2),
        'pacing_volatility_index': round(float(np.mean(pvis)), 4)
    }
