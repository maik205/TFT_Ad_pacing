"""
Pacing Simulation Engine and Throttling Policy Implementation.
Simulates real-world RTB budget pacing under:
1. Uniform Linear Pacing (ULP)
2. PID Feedback Controller
3. Lagged LightGBM (Point Forecast vs Quantile Forecast)
4. Proposed Multi-Horizon TFT Pacing with Dual-Boundary Spike Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimulationResult:
    campaign_id: str
    policy_name: str
    total_budget: float
    total_spend: float
    budget_utilization: float
    is_prematurely_depleted: bool
    depletion_hour: int
    pacing_volatility_index: float
    hourly_alphas: List[float]
    hourly_spends: List[float]
    cumulative_spends: List[float]
    spike_triggers: List[int]


class PacingSimulationEngine:
    def __init__(self, alpha_min: float = 0.05, horizon: int = 24):
        self.alpha_min = alpha_min
        self.horizon = horizon

    def simulate_linear_pacing(self, df_campaign: pd.DataFrame) -> SimulationResult:
        """Simulates Uniform Linear Pacing (ULP)."""
        B = df_campaign['total_budget'].iloc[0]
        T = len(df_campaign)
        nominal_velocity = B / T
        alpha = 1.0

        alphas = []
        spends = []
        cum_spends = []
        cum_spend = 0.0
        depleted = False
        depletion_hour = T

        for t in range(T):
            row = df_campaign.iloc[t]
            nominal_spend = row['hourly_spend']

            if cum_spend >= B:
                # Fully depleted
                if not depleted:
                    depleted = True
                    depletion_hour = t
                spend_t = 0.0
                alpha = self.alpha_min
            else:
                # Spend is scaled by bid multiplier alpha
                realized_spend = min(nominal_spend * alpha, B - cum_spend)
                spend_t = realized_spend
                cum_spend += spend_t

                # Linear adjustment towards nominal velocity
                rem_budget = B - cum_spend
                rem_time = max(1, T - (t + 1))
                target_v = rem_budget / rem_time
                if spend_t > 0:
                    alpha = float(np.clip(alpha * (target_v / spend_t), self.alpha_min, 1.0))

            alphas.append(alpha)
            spends.append(spend_t)
            cum_spends.append(cum_spend)

        pvi = float(np.sqrt(np.mean(np.diff([1.0] + alphas) ** 2)))
        premature = depleted and (depletion_hour < (T - 12))

        return SimulationResult(
            campaign_id=str(df_campaign['campaign_id'].iloc[0]),
            policy_name="Uniform Linear Pacing",
            total_budget=B,
            total_spend=cum_spend,
            budget_utilization=round(cum_spend / B * 100.0, 2),
            is_prematurely_depleted=premature,
            depletion_hour=depletion_hour,
            pacing_volatility_index=round(pvi, 4),
            hourly_alphas=alphas,
            hourly_spends=spends,
            cumulative_spends=cum_spends,
            spike_triggers=[]
        )

    def simulate_pid_pacing(
        self,
        df_campaign: pd.DataFrame,
        Kp: float = 0.0008,
        Ki: float = 0.0001,
        Kd: float = 0.0004
    ) -> SimulationResult:
        """Simulates PID Feedback Controller Pacing."""
        B = df_campaign['total_budget'].iloc[0]
        T = len(df_campaign)
        nominal_velocity = B / T
        alpha = 1.0
        integral = 0.0
        prev_error = 0.0

        alphas = []
        spends = []
        cum_spends = []
        cum_spend = 0.0
        depleted = False
        depletion_hour = T

        for t in range(T):
            row = df_campaign.iloc[t]
            nominal_spend = row['hourly_spend']

            if cum_spend >= B:
                if not depleted:
                    depleted = True
                    depletion_hour = t
                spend_t = 0.0
                alpha = self.alpha_min
            else:
                realized_spend = min(nominal_spend * alpha, B - cum_spend)
                spend_t = realized_spend
                cum_spend += spend_t

                # PID error tracking
                target_cum = nominal_velocity * (t + 1)
                error = target_cum - cum_spend
                integral += error
                deriv = error - prev_error
                prev_error = error

                adj = Kp * error + Ki * integral + Kd * deriv
                alpha = float(np.clip(alpha + adj, self.alpha_min, 1.0))

            alphas.append(alpha)
            spends.append(spend_t)
            cum_spends.append(cum_spend)

        pvi = float(np.sqrt(np.mean(np.diff([1.0] + alphas) ** 2)))
        premature = depleted and (depletion_hour < (T - 12))

        return SimulationResult(
            campaign_id=str(df_campaign['campaign_id'].iloc[0]),
            policy_name="PID Controller",
            total_budget=B,
            total_spend=cum_spend,
            budget_utilization=round(cum_spend / B * 100.0, 2),
            is_prematurely_depleted=premature,
            depletion_hour=depletion_hour,
            pacing_volatility_index=round(pvi, 4),
            hourly_alphas=alphas,
            hourly_spends=spends,
            cumulative_spends=cum_spends,
            spike_triggers=[]
        )

    def simulate_tft_pacing(
        self,
        df_campaign: pd.DataFrame,
        preds_q10: np.ndarray,
        preds_q50: np.ndarray,
        preds_q90: np.ndarray,
        enable_spike_detection: bool = True
    ) -> SimulationResult:
        """
        Simulates Proposed Multi-Horizon TFT Pacing:
        alpha_{t+1} = clip(alpha_t * (B - S_t) / sum(y_hat_90), alpha_min, 1.0)
        + Dual-Boundary Spike Detection Trigger
        """
        B = df_campaign['total_budget'].iloc[0]
        T = len(df_campaign)
        nominal_velocity = B / T
        alpha = 1.0

        alphas = []
        spends = []
        cum_spends = []
        cum_spend = 0.0
        depleted = False
        depletion_hour = T
        spike_triggers = []

        for t in range(T):
            row = df_campaign.iloc[t]
            nominal_spend = row['hourly_spend']

            if cum_spend >= B:
                if not depleted:
                    depleted = True
                    depletion_hour = t
                spend_t = 0.0
                alpha = self.alpha_min
            else:
                realized_spend = min(nominal_spend * alpha, B - cum_spend)
                spend_t = realized_spend
                cum_spend += spend_t

                # Forecasted 24h horizon trajectory
                # preds_q90 is shaped (T, 24) or (T,)
                if len(preds_q90.shape) == 2:
                    future_sum_90 = float(np.sum(preds_q90[t, :min(self.horizon, T - t)]))
                    median_pred = float(preds_q50[t, 0])
                    upper_pred = float(preds_q90[t, 0])
                else:
                    future_sum_90 = float(np.sum(preds_q90[t:min(t + self.horizon, T)]))
                    median_pred = float(preds_q50[t])
                    upper_pred = float(preds_q90[t])

                rem_budget = max(0.0, B - cum_spend)

                # Proposed Quantile Throttling Equation
                if future_sum_90 > 0:
                    alpha = float(np.clip(alpha * (rem_budget / future_sum_90), self.alpha_min, 1.0))

                # Dual-Boundary Spike Detection Condition
                if enable_spike_detection and t > 0:
                    condition_a = (spend_t > upper_pred)
                    condition_b = ((spend_t - median_pred) > 1.5 * nominal_velocity)
                    if condition_a and condition_b:
                        spike_triggers.append(t)
                        # Emergency reserve throttling
                        if spend_t > 0:
                            emergency_alpha = alpha * (median_pred / spend_t)
                            alpha = float(np.clip(emergency_alpha, self.alpha_min, 1.0))

            alphas.append(alpha)
            spends.append(spend_t)
            cum_spends.append(cum_spend)

        pvi = float(np.sqrt(np.mean(np.diff([1.0] + alphas) ** 2)))
        premature = depleted and (depletion_hour < (T - 12))

        return SimulationResult(
            campaign_id=str(df_campaign['campaign_id'].iloc[0]),
            policy_name="Proposed Multi-Horizon TFT",
            total_budget=B,
            total_spend=cum_spend,
            budget_utilization=round(cum_spend / B * 100.0, 2),
            is_prematurely_depleted=premature,
            depletion_hour=depletion_hour,
            pacing_volatility_index=round(pvi, 4),
            hourly_alphas=alphas,
            hourly_spends=spends,
            cumulative_spends=cum_spends,
            spike_triggers=spike_triggers
        )
