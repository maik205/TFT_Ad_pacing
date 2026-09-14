"""
Benchmark Baselines for Programmatic Ad Spend Pacing:
1. Uniform Linear Pacing (ULP)
2. PID Feedback Controller
3. Lagged LightGBM (GBDT Quantile Regression)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple, List


class UniformLinearPacing:
    """
    Uniform Linear Pacing (ULP):
    Allocates budget at a constant nominal velocity B / T.
    Adjusts alpha_t proportionally based on deviation from target velocity.
    """
    def __init__(self, total_budget: float, duration_hours: int, alpha_min: float = 0.05):
        self.B = total_budget
        self.T = duration_hours
        self.alpha_min = alpha_min
        self.nominal_velocity = total_budget / duration_hours
        self.alpha = 1.0

    def step(self, t: int, current_spend: float, cum_spend: float) -> float:
        remaining_budget = max(0.0, self.B - cum_spend)
        remaining_time = max(1, self.T - t)
        target_velocity = remaining_budget / remaining_time

        if current_spend > 0:
            ratio = target_velocity / current_spend
            self.alpha = np.clip(self.alpha * ratio, self.alpha_min, 1.0)
        return self.alpha


class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller for Ad Spend Pacing:
    Error e_t = target_cum_spend - actual_cum_spend
    alpha_{t+1} = alpha_t + K_p * e_t + K_i * sum(e) + K_d * (e_t - e_{t-1})
    """
    def __init__(
        self,
        total_budget: float,
        duration_hours: int,
        Kp: float = 0.0008,
        Ki: float = 0.0001,
        Kd: float = 0.0004,
        alpha_min: float = 0.05
    ):
        self.B = total_budget
        self.T = duration_hours
        self.nominal_velocity = total_budget / duration_hours
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.alpha_min = alpha_min
        self.alpha = 1.0
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, t: int, cum_spend: float) -> float:
        target_cum_spend = self.nominal_velocity * (t + 1)
        error = target_cum_spend - cum_spend

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        adjustment = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.alpha = float(np.clip(self.alpha + adjustment, self.alpha_min, 1.0))
        return self.alpha


class LaggedLightGBM:
    """
    Autoregressive Lagged LightGBM for Multi-Horizon Hourly Spend Forecasting.
    Uses lags up to 168h, rolling statistics, calendar indicators, and quantile regression.
    """
    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
        self.quantiles = quantiles
        self.models: Dict[float, lgb.Booster] = {}
        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engineers autoregressive lagged and rolling features.
        """
        df = df.copy().sort_values(['campaign_id', 'time_idx']).reset_index(drop=True)
        
        # Lag features (1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 168h)
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
        for lag in lags:
            df[f'spend_lag_{lag}'] = df.groupby('campaign_id')['hourly_spend'].shift(lag)
        
        # Rolling window statistics (6h, 24h, 168h)
        for w in [6, 24, 168]:
            df[f'spend_roll_mean_{w}'] = df.groupby('campaign_id')['hourly_spend'].shift(1).rolling(w).mean()
            df[f'spend_roll_std_{w}'] = df.groupby('campaign_id')['hourly_spend'].shift(1).rolling(w).std()
            df[f'spend_roll_max_{w}'] = df.groupby('campaign_id')['hourly_spend'].shift(1).rolling(w).max()

        # Calendar features
        features = [
            'hour_of_day', 'hour_sin', 'hour_cos',
            'day_of_week', 'day_sin', 'day_cos',
            'is_weekend', 'time_to_end', 'countdown_ratio',
            'win_rate', 'avg_cpm', 'nominal_velocity'
        ] + [f'spend_lag_{lag}' for lag in lags] + \
          [f'spend_roll_mean_{w}' for w in [6, 24, 168]] + \
          [f'spend_roll_std_{w}' for w in [6, 24, 168]] + \
          [f'spend_roll_max_{w}' for w in [6, 24, 168]]

        self.feature_names = features
        
        # Drop rows with NaN caused by 168h lags
        valid_mask = df['spend_lag_168'].notna()
        df_clean = df[valid_mask].copy()

        X = df_clean[features]
        y = df_clean['hourly_spend']
        return X, y, df_clean

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Trains separate LightGBM regressors for each quantile.
        """
        for q in self.quantiles:
            params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'boosting_type': 'gbdt',
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
                'random_state': 42
            }
            dtrain = lgb.Dataset(X_train, label=y_train)
            valid_sets = [dtrain]
            if X_val is not None and y_val is not None:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                valid_sets.append(dval)

            booster = lgb.train(params, dtrain, valid_sets=valid_sets, num_boost_round=200)
            self.models[q] = booster

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Returns predictions for each configured quantile.
        """
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X[self.feature_names])
        return preds
