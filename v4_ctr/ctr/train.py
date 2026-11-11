"""Train a CTR model on one world's train split, fit its calibration on the validation split, and write
predictions for the test split of every world given (observed data only, rule R2).

    python -m ctr.train --model cadet --train data/default_s0.npz --eval data/default_s1.npz ... --out runs/cadet_s0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics
from .models import MODELS

FIELDS = ["ctx", "hist_cat", "hist_fmt", "hist_click", "hist_age_h", "cand_cat", "cand_fatigue"]
SPLIT = {"train": 0, "val": 1, "test": 2}


def to_device(data, idx, dev):
    out = {}
    for f in FIELDS:
        a = torch.from_numpy(np.ascontiguousarray(data[f][idx]))
        out[f] = a.to(dev)
    out["y"] = torch.from_numpy(data["clicked"][idx].astype(np.float32)).to(dev)
    return out


def batch_args(t, sl):
    """Batched model arguments, one candidate per row."""
    return (t["ctx"][sl].long(), t["hist_cat"][sl].long(), t["hist_fmt"][sl].long(), t["hist_click"][sl].long(),
            t["hist_age_h"][sl].float(), t["cand_cat"][sl].long().unsqueeze(1), t["cand_fatigue"][sl].float().unsqueeze(1))


@torch.no_grad()
def predict_logits(model, t, bs=16384):
    model.eval()
    n = t["y"].shape[0]
    out = torch.empty(n, device=t["y"].device)
    for s in range(0, n, bs):
        out[s:s + bs] = model.logits(*batch_args(t, slice(s, s + bs)))[:, 0]
    return out


def fit_platt(model, logits, y, fmt):
    """Per-format Platt scale on validation logits (LBFGS on log loss)."""
    for f in torch.unique(fmt).tolist():
        m = fmt == f
        a = torch.ones(1, device=y.device, requires_grad=True)
        b = torch.zeros(1, device=y.device, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits[m] * a + b, y[m])
            loss.backward()
            return loss

        opt.step(closure)
        model.platt.a[f] = a.detach()[0]
        model.platt.b[f] = b.detach()[0]


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dict(np.load(args.train))
    tr = to_device(data, data["split"] == SPLIT["train"], dev)
    va = to_device(data, data["split"] == SPLIT["val"], dev)
    model = MODELS[args.model]().to(dev)
    lr = args.lr or (2e-3 if args.model == "cadet" else 1e-2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    n = tr["y"].shape[0]
    steps = args.epochs * ((n + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state, log = float("inf"), None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(0, n, args.bs):
            idx = perm[s:s + args.bs]
            z = model.logits(*batch_args(tr, idx))[:, 0]
            loss = F.binary_cross_entropy_with_logits(z, tr["y"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * idx.numel()
        vz = predict_logits(model, va)
        vl = F.binary_cross_entropy_with_logits(vz, va["y"]).item()
        log.append({"epoch": ep, "train_logloss": tot / n, "val_logloss": vl, "seconds": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    vz = predict_logits(model, va)
    fit_platt(model, vz, va["y"], va["ctx"][:, 0].long())
    return model, log


@torch.no_grad()
def evaluate(model, path, dev):
    data = dict(np.load(path))
    out = {}
    preds = {}
    for name in ("val", "test"):
        sel = data["split"] == SPLIT[name]
        t = to_device(data, sel, dev)
        z = predict_logits(model, t)
        fmt = t["ctx"][:, 0].long()
        p = torch.sigmoid(model.platt(z, fmt)).cpu().numpy()
        y = data["clicked"][sel]
        out[name] = metrics.report(y, p, data["ctx"][sel, 0])
        if name == "test":
            preds = {"opp_id": data["opp_id"][sel], "pred": p, "clicked": y, "format": data["ctx"][sel, 0]}
    return out, preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    model, log = train(args)
    dev = next(model.parameters()).device
    torch.save({"model": args.model, "state": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "model.pt"))
    results = {"synthetic": True, "model": args.model, "train_world": args.train, "training": log,
               "platt": {"a": model.platt.a.tolist(), "b": model.platt.b.tolist()},
               "params": sum(p.numel() for p in model.parameters()), "eval": {}}
    for path in [args.train, *args.eval]:
        name = os.path.splitext(os.path.basename(path))[0]
        res, preds = evaluate(model, path, dev)
        results["eval"][name] = res
        np.savez(os.path.join(args.out, f"pred_{name}.npz"), **preds)
        print(name, json.dumps(res["test"]["all"]), flush=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
