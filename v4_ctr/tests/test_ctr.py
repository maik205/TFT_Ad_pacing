"""Checks for the CTR pipeline. Run from v4_ctr: python -m pytest tests -q"""
import glob
import os
import re

import numpy as np
import torch

from ctr import contract as K
from ctr.models import CadetCTR, LogRegCTR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ctr_code_never_reads_truth():
    """Rule R2: model code does not import the oracle or open out/truth; only evaluation/ may."""
    bad = []
    for f in glob.glob(os.path.join(ROOT, "ctr", "*.py")):
        src = open(f, encoding="utf-8").read()
        if re.search(r"^\s*(from|import)\s+synth_truth", src, re.M) or re.search(r"truth_\w+\.(parquet|npy|json)|[\\/]truth[\\/\"']", src):
            bad.append(os.path.basename(f))
    assert not bad, bad


def _request(rng, n):
    L = K.HISTORY_LEN
    hist_fmt = rng.integers(0, 4, (1, L))
    return [torch.tensor([[2, 5, 7, 13, 3, 6]]),
            torch.from_numpy(rng.integers(0, K.N_CAT_TOKENS, (1, L, 2))),
            torch.from_numpy(hist_fmt),
            torch.from_numpy(rng.integers(0, 4, (1, L))),
            torch.from_numpy(rng.uniform(0, 100, (1, L)).astype(np.float32)),
            torch.from_numpy(rng.integers(2, K.N_CAT_TOKENS, (n, 2))),
            torch.from_numpy(rng.integers(0, 4, n).astype(np.float32))]


def test_candidates_are_scored_independently():
    """A candidate's probability must not depend on which other candidates share the request; otherwise
    one model call per request would rank differently from one call per candidate."""
    rng = np.random.default_rng(0)
    for cls in (CadetCTR, LogRegCTR):
        torch.manual_seed(0)
        m = cls().eval()
        if cls is LogRegCTR:
            for p in m.parameters():
                torch.nn.init.normal_(p, std=0.1)
        args = _request(rng, 6)
        with torch.no_grad():
            together = m(*args)
            alone = torch.cat([m(*args[:5], args[5][i:i + 1], args[6][i:i + 1]) for i in range(6)])
        assert torch.allclose(together, alone, atol=1e-6), cls.__name__


def test_padding_does_not_change_scores():
    """Padded history slots (format 0) carry no information, whatever their other fields hold."""
    rng = np.random.default_rng(1)
    torch.manual_seed(0)
    m = CadetCTR().eval()
    args = _request(rng, 3)
    args[2][0, 20:] = 0
    noisy = [a.clone() for a in args]
    noisy[1][0, 20:] = 7
    noisy[3][0, 20:] = 2
    noisy[4][0, 20:] = 55.0
    with torch.no_grad():
        assert torch.allclose(m(*args), m(*noisy), atol=1e-6)
