"""World configuration. Every default here is taken from GENERATOR_SPEC.md (frozen 2026-10-17).

A world is fully determined by (config, world seed); `config_hash` goes into every output's metadata.
Values not fixed by the spec are marked IMPL and listed in the spec's deviation log (Section 16).
"""
import copy
import hashlib
import json

TZ_MIXES = {
    "single": {7: 1.0},
    "4zone": {7: 0.4, 8: 0.2, 1: 0.2, -5: 0.2},
    "24zone": {o: 1 / 24 for o in range(-11, 13)},
}

DEFAULT = {
    # Section 1
    "days": 28,
    # Section 3
    "tz_mix": "4zone",
    "regions_per_zone": 3,  # IMPL: several region codes per zone
    "p_missing": 0.20,
    "p_wrong": 0.05,
    # Section 5
    "n_users": 50_000,
    "alpha": 2.0,
    "prevalence": [0.30, 0.15, 0.20, 0.20, 0.15],
    "eps_sd": 0.3,
    "drift_sd": 0.05,
    "lambda_median": 3.0,
    "lambda_sigma": 1.0,
    "late_join_share": 0.3,
    "mean_lifetime_days": 14.0,  # None = infinite
    # Section 6
    "n_campaigns": 40,
    "second_cat_share": 0.2,
    "v_sigma": 0.4,
    "q_sd": 0.3,
    "rhos": [0.3, 0.5, 0.7],
    # Section 7
    "extra_slots_mean": 0.5,
    "slot_gap_s": 60,  # IMPL: spacing of slots within a visit
    "p_video": 0.4,
    "p_feed": 0.6,
    "preroll_durations": [6, 15, 30],
    "preroll_probs": [0.3, 0.4, 0.3],
    "infeed_dur_logmean": 4.871,  # KuaiRand-Pure AD views, measured 2026-10-17
    "infeed_dur_logsd": 0.616,
    "context_uniform_share": 0.3,
    # Section 7a
    "eta0": -0.5,
    "eta1": 0.8,
    "h0": 0.01,  # deviation 1 (was 0.03): calibrated to KuaiRand in-feed watch fractions
    "night_mult": 3.0,
    "autoplay_mean": 8.0,
    "engaged_to_casual": 0.1,
    "a_amb": 0.10,
    "absent_loop_mean": 1.0,  # IMPL: in-feed absent watch fraction = 1 + Exp(mean) loops    # Section 8
    "gamma": 1.0,
    "delta": 0.5,
    "phi": 0.4,
    "s_state": [0.0, -1.0, -5.0],
    "target_ctr": 0.01,
    # Section 9
    "c5_sigma": 0.865,  # iPinYou season 2 paying price, measured 2026-10-17
    "c3_ext_sigma": 0.5,
    # Section 10
    "epsilon": 0.10,
    "tau": 1.0,
    "delta_ctx": 1.0,
    "none_score": 1.0,  # IMPL: fixed score of the "no ad" option
    "logging_rho": 0.5,  # IMPL: the logging policy paces to the C1 budget at this rho
    "pacing_steps_per_hour": 10,  # IMPL: throttle re-evaluated every 6 minutes
    # Section 12
    "train_days": 21,
    # filled by calibrate.py (Section 7a.2 fit, Section 13 gamma_real)
    "infeed": None,
}

STATES = ("engaged", "casual", "absent")
FORMATS = ("display", "preroll", "infeed")


def make_config(overrides=None, calibration=None):
    cfg = copy.deepcopy(DEFAULT)
    if calibration:
        cfg["infeed"] = calibration["infeed"]
        if calibration.get("gamma_real") is not None:
            cfg["gamma_real"] = calibration["gamma_real"]
    for k, v in (overrides or {}).items():
        if k not in cfg and k != "gamma_real":
            raise KeyError(f"unknown config key {k}")
        cfg[k] = v
    if cfg["gamma"] == "gamma_real":
        cfg["gamma"] = cfg["gamma_real"]
    return cfg


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
