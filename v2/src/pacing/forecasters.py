"""Rolling-origin probabilistic forecasters.

Every forecaster returns F with shape (n_campaigns, n_origins, HORIZON, len(QUANTILES)):
F[c, i, h, k] is the q_k-quantile forecast of campaign c's spend in hour origins[i] + h,
made with information from hours < origins[i] only.
"""

import time

import numpy as np
import pandas as pd

from . import config as C
from .data import calendar

Q = np.array(C.QUANTILES)
STEPS = np.arange(C.HORIZON)


def test_origins():
    return np.arange(C.VAL_END, C.TEST_END)


# ---------------------------------------------------------------- seasonal naive

def seasonal_naive(Y, origins, n_days=7):
    """Quantiles over the same hour of the previous `n_days` days."""
    k = np.arange(1, n_days + 1)
    idx = origins[:, None, None] + STEPS[None, :, None] - 24 * k[None, None, :]
    assert (idx < origins[:, None, None]).all() and idx.min() >= 0
    past = Y[:, idx]                                   # (n, no, H, n_days)
    return np.moveaxis(np.quantile(past, Q, axis=-1), 0, -1)


# ---------------------------------------------------------------- LightGBM (direct)

LGB_FEATURES = (
    [f"lag_{l}" for l in range(1, 25)]
    + ["seas_1", "seas_2", "seas_7", "seas_mean7", "roll24_mean", "roll24_std", "roll168_mean",
       "step", "hod_sin", "hod_cos", "dow_sin", "dow_cos", "log_scale", "ctr", "cv"]
)


def lgb_design(Yn, origins, stat):
    """Feature rows for every (campaign, origin, step). Yn is spend divided by the campaign's
    training mean hourly spend. Only hours < origin enter the features, except the calendar
    phase of the target hour, which is known in advance."""
    n, no, H = Yn.shape[0], len(origins), C.HORIZON
    o = origins[:, None]                               # (no, 1)
    tgt = o + STEPS[None, :]                           # (no, H) target hours
    cols = {}
    for l in range(1, 25):
        cols[f"lag_{l}"] = np.broadcast_to(Yn[:, origins - l][:, :, None], (n, no, H))
    for name, d in (("seas_1", 1), ("seas_2", 2), ("seas_7", 7)):
        cols[name] = Yn[:, tgt - 24 * d]
    cols["seas_mean7"] = np.mean(np.stack([Yn[:, tgt - 24 * d] for d in range(1, 8)]), axis=0)
    w24 = Yn[:, origins[:, None] - np.arange(1, 25)[None, :]]       # (n, no, 24)
    w168 = Yn[:, origins[:, None] - np.arange(1, 169)[None, :]]
    for name, v in (("roll24_mean", w24.mean(-1)), ("roll24_std", w24.std(-1)), ("roll168_mean", w168.mean(-1))):
        cols[name] = np.broadcast_to(v[:, :, None], (n, no, H))
    cols["step"] = np.broadcast_to(STEPS[None, None, :], (n, no, H))
    cal = calendar(tgt.ravel())
    for c in cal.columns:
        cols[c] = np.broadcast_to(cal[c].to_numpy().reshape(no, H)[None], (n, no, H))
    for name, v in (("log_scale", np.log(stat.mean_hourly_spend.to_numpy())),
                    ("ctr", stat.ctr.to_numpy()), ("cv", stat.cv.to_numpy())):
        cols[name] = np.broadcast_to(v[:, None, None], (n, no, H))
    X = np.stack([np.asarray(cols[f], dtype=np.float32).ravel() for f in LGB_FEATURES], axis=1)
    valid = tgt < Yn.shape[1]
    y = np.where(valid, Yn[:, np.minimum(tgt, Yn.shape[1] - 1)], np.nan).astype(np.float32).ravel()
    return X, y


def lightgbm_forecast(Y, stat, seed=0, train_step=2):
    import lightgbm as lgb

    scale = stat.mean_hourly_spend.to_numpy()[:, None]
    Yn = Y / scale
    tr_o = np.arange(C.LOOKBACK, C.TRAIN_END - C.HORIZON + 1, train_step)   # targets < TRAIN_END
    va_o = np.arange(C.TRAIN_END, C.VAL_END - C.HORIZON + 1, train_step)    # targets in val
    Xtr, ytr = lgb_design(Yn, tr_o, stat)
    Xva, yva = lgb_design(Yn, va_o, stat)
    Xte, _ = lgb_design(Yn, test_origins(), stat)
    t0 = time.time()
    preds, best_iters = [], {}
    for q in C.QUANTILES:
        params = dict(objective="quantile", alpha=q, learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, seed=seed, verbose=-1, num_threads=0)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=LGB_FEATURES)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iters[q] = bst.best_iteration
        preds.append(bst.predict(Xte, num_iteration=bst.best_iteration))
    fit_s = time.time() - t0
    n, no = Y.shape[0], len(test_origins())
    F = np.stack(preds, axis=-1).reshape(n, no, C.HORIZON, len(C.QUANTILES)) * scale[:, :, None, None]
    return np.clip(F, 0, None), {"fit_seconds": fit_s, "best_iterations": best_iters,
                                 "n_train_rows": int(len(ytr)), "n_val_rows": int(len(yva))}


# ---------------------------------------------------------------- neuralforecast (TFT, NHITS)

def nf_frames(panel, campaigns, stat):
    sub = panel[panel.campaign.isin(campaigns)].copy()
    st = stat.loc[sub.campaign]
    df = pd.DataFrame({
        "unique_id": sub.campaign.astype(str).to_numpy(),
        "ds": sub.hour.to_numpy(),
        "y": sub.spend.to_numpy() / st.mean_hourly_spend.to_numpy(),
        "imps": sub.imps.to_numpy() / st.mean_imps.to_numpy(),
        "clicks": sub.clicks.to_numpy() / st.mean_imps.to_numpy(),
    })
    df = pd.concat([df.reset_index(drop=True), calendar(df.ds).reset_index(drop=True)], axis=1)
    static_df = pd.DataFrame({
        "unique_id": [str(c) for c in campaigns],
        "log_scale": np.log(stat.mean_hourly_spend.to_numpy()),
        "ctr": stat.ctr.to_numpy(), "cv": stat.cv.to_numpy(),
    })
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True), static_df


def _quantile_columns(cols, alias):
    """Map neuralforecast MQLoss output columns to quantile levels (handles both naming schemes)."""
    out = {}
    for c in cols:
        if not c.startswith(alias + "-") and c != alias:
            continue
        tail = c[len(alias) + 1:] if c != alias else "median"
        if tail.startswith("ql"):
            out[round(float(tail[2:]), 4)] = c
        elif tail == "median":
            out[0.5] = c
        elif tail.startswith("lo-"):
            out[round(0.5 - float(tail[3:]) / 200, 4)] = c
        elif tail.startswith("hi-"):
            out[round(0.5 + float(tail[3:]) / 200, 4)] = c
    missing = [q for q in C.QUANTILES if round(q, 4) not in out]
    assert not missing, f"no output column for quantiles {missing} in {list(cols)}"
    return [out[round(q, 4)] for q in C.QUANTILES]


def neural_forecast(kind, panel, campaigns, stat, seed, accelerator="cpu", max_steps=3000):
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS, TFT

    df, static_df = nf_frames(panel, campaigns, stat)
    hist, futr, stat_cols = ["imps", "clicks"], ["hod_sin", "hod_cos", "dow_sin", "dow_cos"], ["log_scale", "ctr", "cv"]
    common = dict(h=C.HORIZON, input_size=C.LOOKBACK, loss=MQLoss(quantiles=list(C.QUANTILES)),
                  hist_exog_list=hist, futr_exog_list=futr, stat_exog_list=stat_cols,
                  max_steps=max_steps, val_check_steps=50, early_stop_patience_steps=10,
                  learning_rate=1e-3, batch_size=64, windows_batch_size=256,
                  scaler_type="robust", random_seed=seed, alias=kind,
                  accelerator=accelerator, enable_progress_bar=False, logger=False)
    if kind == "TFT":
        model = TFT(hidden_size=64, dropout=0.1, **common)
    elif kind == "NHITS":
        model = NHITS(**common)
    else:
        raise ValueError(kind)
    nf = NeuralForecast(models=[model], freq=1)
    test_size = C.N_HOURS - C.VAL_END
    t0 = time.time()
    # trains on hours < TRAIN_END, early-stops on [TRAIN_END, VAL_END), then forecasts
    # every origin in [VAL_END, N_HOURS - HORIZON] without refitting
    cv = nf.cross_validation(df=df, static_df=static_df, n_windows=None, test_size=test_size,
                             val_size=C.VAL_END - C.TRAIN_END, step_size=1, refit=False)
    wall = time.time() - t0
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    qcols = _quantile_columns(cv.columns, kind)
    cv["origin"] = cv["cutoff"] + 1
    cv["step"] = cv["ds"] - cv["origin"]
    origins = test_origins()
    cid = {str(c): i for i, c in enumerate(campaigns)}
    F = np.full((len(campaigns), len(origins), C.HORIZON, len(C.QUANTILES)), np.nan)
    cv = cv[cv.origin.isin(origins)]
    F[cv.unique_id.map(cid).to_numpy(), cv.origin.to_numpy() - origins[0], cv.step.to_numpy()] = cv[qcols].to_numpy()
    # The last test origin (719) has no full 24 h of future hours in the data, so
    # cross_validation stops at 718; reuse origin 718 shifted by one hour (logged in PROTOCOL 10).
    last = len(origins) - 1
    if np.isnan(F[:, last]).all():
        F[:, last, :-1] = F[:, last - 1, 1:]
        F[:, last, -1] = F[:, last - 1, -1]
    assert not np.isnan(F).any(), "missing neural forecasts"
    F = F * stat.mean_hourly_spend.to_numpy()[:, None, None, None]
    try:
        steps = int(nf.models[0].trainer.global_step)
    except Exception:
        steps = None
    return np.clip(F, 0, None), {"wall_seconds_fit_and_predict": wall, "train_steps": steps}
