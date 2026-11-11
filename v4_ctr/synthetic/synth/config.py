"""World configuration of the ad-format fork (FORK_SPEC.md). Defaults not listed in the fork's deviation
log are those of the frozen v3 specification (BASE_SPEC_v3_frozen.md, 2026-10-17).

A world is fully determined by (config, world seed); `config_hash` goes into every output's metadata.
Values not fixed by either specification are marked IMPL.
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
    # F1: banner / interstitial / rewarded share of slots of an attentive visit. An absent visit never
    # opens a rewarded slot (opt-in), so its slots split banner : interstitial in the same ratio.
    "format_mix": [0.5, 0.3, 0.2],
    "interstitial_durations": [6, 15, 30],  # pre-roll mechanics of v3
    "interstitial_probs": [0.3, 0.4, 0.3],
    "rewarded_durations": [15, 30],  # F2
    "rewarded_probs": [0.5, 0.5],
    "context_uniform_share": 0.3,
    # Section 7a
    "eta0": -0.5,
    "eta1": 0.8,
    "h0": 0.01,  # deviation 1 (was 0.03): calibrated to KuaiRand in-feed watch fractions
    "night_mult": 3.0,
    "autoplay_mean": 8.0,
    "engaged_to_casual": 0.1,
    "a_amb": 0.10,
    # F2: completion of a rewarded view by state (engaged, casual); rewarded views are never skippable
    "p_complete_rewarded": [0.97, 0.90],
    # Section 8
    "gamma": 1.0,
    "delta": 0.5,
    "phi": 0.4,
    "s_state": [0.0, -1.0, -5.0],
    "target_ctr": 0.01,
    # F3: per-format click terms (banner, interstitial, rewarded): a logit offset and a factor on the
    # affinity term. Design choices, not calibrated; see FORK_SPEC.md
    "s_format": [0.0, 0.4, -0.4],
    "gamma_format": [1.0, 1.0, 0.5],
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
}

STATES = ("engaged", "casual", "absent")
FORMATS = ("banner", "interstitial", "rewarded")


def make_config(overrides=None, calibration=None):
    cfg = copy.deepcopy(DEFAULT)
    if calibration and calibration.get("gamma_real") is not None:
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
