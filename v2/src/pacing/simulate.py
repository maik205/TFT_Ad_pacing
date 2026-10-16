"""Run policies over campaign-day episodes with common random numbers."""

import multiprocessing as mp

import numpy as np

from . import config as C
from . import metrics as M
from .env import run_episode, uniforms
from .policies import Hindsight

_STATE = {}


def _init(store, budgets_fn_args, policy_factory):
    _STATE.update(store=store, budgets=budgets_fn_args, factory=policy_factory)


def _run_campaign(args):
    c, days, seeds = args
    store, factory, daily = _STATE["store"], _STATE["factory"], _STATE["budgets"]
    policies = factory()
    out = []
    for rho in C.RHOS:
        B = rho * daily[c]
        for day in days:
            ep = store.episode(c, day, B)
            demand = ep.demand
            ref = Hindsight()
            for s in seeds:
                u = uniforms(ep, s)
                r_ref = run_episode(ep, ref, u)
                ref.reset(ep)
                ref_profile = ref.a * demand
                for pol in [ref] + policies:
                    r = r_ref if pol is ref else run_episode(ep, pol, u)
                    out.append({
                        "policy": pol.name, "rho": rho, "campaign": c, "day": day, "seed": s,
                        "budget": B, "demand": float(demand.sum()),
                        "spend": r["spend"], "clicks": r["clicks"], "convs": r["convs"],
                        "ref_clicks": r_ref["clicks"], "ref_convs": r_ref["convs"],
                        "depletion_hour": r["depletion_hour"], "pvi": r["pvi"],
                        "profile_dev": M.profile_deviation(r["hourly_spend"], ref_profile, B),
                    })
    return out


def simulate(store, daily_train_spend, policy_factory, campaigns_idx, days, seeds, processes=None):
    """policy_factory() must build fresh policy objects (called once per worker task)."""
    tasks = [(c, days, seeds) for c in campaigns_idx]
    # fork shares the impression store with workers instead of pickling it per worker
    with mp.get_context("fork").Pool(processes, initializer=_init,
                                     initargs=(store, daily_train_spend, policy_factory)) as pool:
        chunks = pool.map(_run_campaign, tasks, chunksize=4)
    return [row for chunk in chunks for row in chunk]
