"""Integrity tests of spec Section 15 on a small world. Run: python -m pytest tests -q

Test 9 (fits hit targets) runs on full-size worlds via tests/check_world.py.
"""
import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from synth.config import make_config
from synth.generate import build, main as generate_main
from synth.mi import synthetic_mi_excess
from synth.opportunities import local_time
from synth.world import ARCHS, CATS, rhythm_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(ROOT, "calibration.json")
SMALL = {"n_users": 2000, "n_campaigns": 12}


def calib():
    return json.load(open(CALIB))


def small_cfg(**over):
    return make_config({**SMALL, **over}, calib())


@pytest.fixture(scope="module")
def world():
    return build(small_cfg(), seed=1)


# 1. isolation
def test_isolation():
    allowed = {"synth_truth", "evaluation", "tests"}
    bad = []
    for f in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, ROOT).replace("\\", "/")
        top = rel.split("/")[0]
        if top in allowed:
            continue
        src = open(f, encoding="utf-8").read()
        if re.search(r"^\s*(from|import)\s+synth_truth", src, re.M):
            bad.append(f"{rel}: imports synth_truth")
        # only the generator's writer may name truth files
        if top != "synth" and re.search(r"truth_\w+\.(parquet|npy|json)|[\\/]truth[\\/\"']", src):
            bad.append(f"{rel}: references out/truth")
    assert not bad, bad


# 2. determinism
def test_determinism(tmp_path):
    hashes = []
    for run in ("a", "b"):
        out = tmp_path / run
        generate_main(["--out", str(out), "--seed", "3", "--calibration", CALIB] +
                      [f"--set={k}={v}" for k, v in SMALL.items()])
        hashes.append({os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
                       for p in sorted(glob.glob(str(out / "observed" / "*.parquet")))})
    assert hashes[0] == hashes[1]
    assert len(hashes[0]) == 6


# 3. planted structure
def test_rhythms_reproduced(world):
    users, opp = world["users"], world["opp"]
    R = rhythm_table().reshape(5, 2, 24, -1).mean(axis=3)
    vu, vt = opp["visit_user"], opp["visit_ts"]
    _, dt, _, lh = local_time(vt, users["tz"][vu])
    main = users["w"].argmax(axis=1)
    alive = np.clip(np.minimum(users["leave"], 28) - users["join"], 0, None)
    peaks = []
    for a in range(5):
        grp = main == a
        sel = grp[vu] & (dt == 0)
        assert sel.sum() > 1000, (ARCHS[a], sel.sum())
        h = np.bincount(lh[sel].astype(int), minlength=24).astype(float)
        # expected weekday histogram of this group: activity-weighted mixture of its users' rhythms
        wt = users["lam"][grp] * alive[grp]
        expected = (wt[:, None] * (users["w"][grp] @ R[:, 0, :])).sum(axis=0)
        r = np.corrcoef(h, expected)[0, 1]
        assert r > 0.97, (ARCHS[a], r)
        peaks.append(int(np.argmax(h)))
    assert len(set(peaks)) >= 4, peaks  # groups peak at different local hours


def test_interest_semantics(world):
    """Office workers' affinity for enterprise software is higher at weekday work hours than late."""
    users, opp = world["users"], world["opp"]
    vu, vt, it = opp["visit_user"], opp["visit_ts"], opp["visit_interest"]
    _, dt, _, lh = local_time(vt, users["tz"][vu])
    office = users["w"][vu, 0] > 0.7
    ent = it[:, CATS.index("enterprise_software")]
    work = office & (dt == 0) & (lh >= 10) & (lh < 16)
    late = office & (lh >= 21)
    assert ent[work].mean() - ent[late].mean() > 1.0
    gamer = users["w"][vu, 1] > 0.7
    vg = it[:, CATS.index("video_games")]
    assert vg[gamer & ((lh >= 22) | (lh < 4))].mean() - vg[gamer & (lh >= 7) & (lh < 9)].mean() > 1.0


def test_context_follows_interest(world):
    """Empirical context frequencies match the planted sampling distribution."""
    cfg, opp = small_cfg(), world["opp"]
    it = opp["visit_interest"][opp["visit"]]
    z = cfg["gamma"] * it
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p = (1 - cfg["context_uniform_share"]) * p + cfg["context_uniform_share"] / len(CATS)
    emp = np.bincount(opp["context"], minlength=len(CATS)) / opp["context"].size
    assert np.abs(emp - p.mean(axis=0)).max() < 0.005


# 4. negative control
def test_negative_control():
    cfg = small_cfg(gamma=0.0, tz_mix="single")
    w = build(cfg, seed=2)
    imp, users, opp = w["imp"], w["users"], w["opp"]
    z = imp["logit_no_beta0"] - (w["camp"]["q"][imp["campaign"]] + cfg["delta"] * imp["match"]
                                 - cfg["phi"] * np.log1p(imp["n_uc"]) + np.array(cfg["s_state"])[imp["state"]])
    assert np.abs(z).max() < 1e-9  # no affinity term left: pCTR has no hour dependence via interest
    _, _, _, lh = local_time(opp["ts"], users["tz"][opp["user"]])
    for lo, hi in [(0, 6), (9, 17), (20, 24)]:
        sel = (lh >= lo) & (lh < hi)
        share = np.bincount(opp["context"][sel], minlength=len(CATS)) / sel.sum()
        assert np.abs(share - 1 / len(CATS)).max() < 0.01
    ex = synthetic_mi_excess(cfg, users, opp)[0]
    assert abs(ex) < 0.003, ex


# 5. propensities
def test_propensities(world):
    log = world["log"]
    p, ne = log["propensity"], log["n_eligible"]
    cfg = small_cfg()
    assert ((p > 0) & (p <= 1)).all()
    assert (p >= cfg["epsilon"] / (ne + 1) - 1e-12).all()
    # importance-weight identity: E[1 / p(a)] = number of actions available
    ratio = np.mean(1 / p / (ne + 1))
    assert abs(ratio - 1) < 0.03, ratio


# 6. calibration of CTR
def test_ctr_calibration(world):
    out = world["out"]
    assert abs(out["pctr"].mean() / small_cfg()["target_ctr"] - 1) < 1e-6
    n = out["clicked"].size
    se = np.sqrt(0.01 * 0.99 / n)
    assert abs(out["clicked"].mean() - 0.01) < 4 * se + 0.02 * 0.01


# 7. engagement sanity
def test_engagement_rates(world):
    imp, out = world["imp"], world["out"]
    st = imp["state"]
    for s, target in enumerate([0.6, 0.2, 0.03]):
        sel = st == s
        rate = out["interacted"][sel].mean()
        assert abs(rate - target) < 4 * np.sqrt(target * (1 - target) / sel.sum()) + 1e-3, (s, rate)
    # absent click rate vs engaged, same campaigns and context: bounded by exp(s_absent) = exp(-5) ~ 0.67%
    ratio = out["pctr"][st == 2].mean() / out["pctr"][st == 0].mean()
    assert ratio < 0.01, ratio


# 8. ambiguity
def test_ambiguity(world):
    imp, out = world["imp"], world["out"]
    st, dwell = imp["state"], out["post_dwell_s"]
    ts = np.quantile(dwell, np.linspace(0.05, 0.95, 91))
    acc = [((dwell > t) == (st == 2)).mean() for t in ts]
    t_best = ts[int(np.argmax(acc))]
    eng = st == 0
    wrong = (dwell[eng] > t_best).mean()
    assert wrong >= 0.8 * small_cfg()["a_amb"], wrong


def test_oracle_matches_logged_pctr(tmp_path):
    from synth_truth.oracle import Oracle  # evaluation-side use, allowed in tests
    out = tmp_path / "w"
    generate_main(["--out", str(out), "--seed", "4", "--calibration", CALIB] +
                  [f"--set={k}={v}" for k, v in SMALL.items()])
    orc = Oracle(str(out))
    ti = pd.read_parquet(out / "truth" / "truth_impressions.parquet")
    imp = pd.read_parquet(out / "observed" / "synth_impressions.parquet")
    camp = imp.set_index("opp_id").loc[ti.opp_id, "campaign_id"].to_numpy().astype(int)
    p = orc.pctr(ti.opp_id.to_numpy(), camp, ti.n_uc.to_numpy())
    assert np.allclose(p, ti.pctr.to_numpy(), rtol=1e-6)
    eng = pd.read_parquet(out / "observed" / "synth_engagement.parquet")
    e = orc.expected(ti.opp_id.to_numpy(), camp, ti.n_uc.to_numpy())
    vid = e.cost_C6.notna() & orc.format[ti.opp_id.to_numpy()] > 0
    pc = orc.p_complete(ti.opp_id.to_numpy(), camp)
    comp = eng.set_index("opp_id").loc[ti.opp_id, "completed"].astype(float).to_numpy()
    v = ~np.isnan(pc)
    assert abs(np.nanmean(comp[v]) - pc[v].mean()) < 0.01
