"""CTR models with the contract's inputs (contract.py). Both take one request's inputs with N candidates
and return N calibrated click probabilities, so the ad server runs either through the same code.

CadetCTR adapts CADET (arXiv 2602.11410) to the ad server's budget (ADR 0004):
- decoder-only transformer over the user's served-ad history followed by the candidates;
- history tokens attend to themselves and older history; each candidate attends to the history and to
  itself only, so candidates of one request are scored independently in one pass;
- timestamp-based rotary position embeddings: the rotation angle is the event's age in hours, not its
  index, so gaps between served ads count;
- self-gated attention: each attention output is multiplied by a sigmoid gate of its input;
- session masking: outcomes younger than OUTCOME_DELAY_S are a separate "unknown" token (contract.py);
- context conditioning and one tower per ad format: the request context is added to every candidate token
  and fed again to the format's tower; the tower is chosen by the request's format, which the server knows
  before scoring (ADR 0004 decision 12). The paper's position towers have no counterpart here.
- calibration: one Platt scale per format, fitted on validation data, is part of the exported graph,
  because nothing after the model corrects its scale (ADR 0004 decision 7).

LogRegCTR is the baseline: logistic regression on the same inputs with explicit crosses.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import contract as K

N_FMT = 1 + len(K.FORMATS)
N_REG = K.N_REGION_TOKENS
N_CAT = K.N_CAT_TOKENS


class PlattHead(nn.Module):
    """Per-format Platt scale (a, b) applied to a logit; identity until fitted."""

    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.ones(N_FMT))
        self.register_buffer("b", torch.zeros(N_FMT))

    def forward(self, logit, fmt):
        return logit * self.a[fmt] + self.b[fmt]


def _unbatched(ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue):
    """Serving shape (1 request, N candidates) to the batched shape (B=1, M=N)."""
    return ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat.unsqueeze(0), cand_fatigue.unsqueeze(0)


class ContextEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fmt = nn.Embedding(N_FMT, d)
        self.cat = nn.Embedding(N_CAT, d)
        self.region = nn.Embedding(N_REG, d)
        self.lhour = nn.Embedding(25, d)
        self.wday = nn.Embedding(8, d)
        self.uhour = nn.Embedding(25, d)

    def forward(self, ctx):  # [B, 6] -> [B, d]
        return (self.fmt(ctx[:, 0]) + self.cat(ctx[:, 1]) + self.region(ctx[:, 2]) + self.lhour(ctx[:, 3])
                + self.wday(ctx[:, 4]) + self.uhour(ctx[:, 5]))


def rotary(x, pos_h, inv_period):
    """Rotate pairs of x [B, H, T, dh] by angle pos_h [B, T] * inv_period [dh/2]."""
    ang = pos_h[:, None, :, None] * inv_period[None, None, None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.stack([r1, r2], dim=-1).flatten(-2)


class GatedBlock(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        # periods from 1 hour to 2 weeks, geometric
        n = self.dh // 2
        periods = torch.exp(torch.linspace(math.log(1.0), math.log(24.0 * 14), n))
        self.register_buffer("inv_period", 2 * math.pi / periods)

    def forward(self, x, pos_h, allowed):
        B, T, D = x.shape
        y = self.ln1(x)
        q, k, v = self.qkv(y).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, pos_h, self.inv_period), rotary(k, pos_h, self.inv_period)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~allowed[:, None], -1e9)
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + torch.sigmoid(self.gate(y)) * self.out(o)
        return x + self.ff(self.ln2(x))


class CadetCTR(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, ffn=128, tower=64):
        super().__init__()
        self.ctx = ContextEmbedding(d)
        self.cat = nn.Embedding(N_CAT, d, padding_idx=0)
        self.fmt = nn.Embedding(N_FMT, d, padding_idx=0)
        self.click = nn.Embedding(4, d, padding_idx=0)
        self.age = nn.Linear(1, d)
        self.fat = nn.Linear(1, d)
        self.cand_type = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([GatedBlock(d, heads, ffn) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.towers = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, tower), nn.GELU(), nn.Linear(tower, 1))
                                     for _ in range(N_FMT)])
        self.platt = PlattHead()
        # Tokens the synthetic world never produces (a category outside the vocabulary, an app without a
        # category) start and stay at zero, so at serve time they add nothing instead of a random vector.
        with torch.no_grad():
            self.cat.weight[K.CAT_OTHER].zero_()
            self.ctx.cat.weight[K.CAT_PAD].zero_()
            self.ctx.cat.weight[K.CAT_OTHER].zero_()

    def logits(self, ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue):
        """Batched: ctx [B,6], hist_* [B,L(,2)], cand_cat [B,M,2], cand_fatigue [B,M] -> logits [B,M]."""
        B, L = hist_fmt.shape
        M = cand_cat.shape[1]
        c = self.ctx(ctx)
        hmask = hist_fmt > 0
        h = (self.cat(hist_cat).sum(2) + self.fmt(hist_fmt) + self.click(hist_click)
             + self.age(torch.log1p(hist_age_h).unsqueeze(-1)))
        h = h * hmask.unsqueeze(-1)
        k = (self.cat(cand_cat).sum(2) + self.fat(torch.log1p(cand_fatigue).unsqueeze(-1))
             + c.unsqueeze(1) + self.cand_type)
        x = torch.cat([h, k], dim=1)
        pos = torch.cat([-hist_age_h, torch.zeros(B, M, dtype=hist_age_h.dtype, device=x.device)], dim=1)
        allowed = self.mask(hmask, M)
        for blk in self.blocks:
            x = blk(x, pos, allowed)
        s = self.ln(x[:, L:])  # [B, M, d]
        z = torch.cat([s, c.unsqueeze(1).expand(B, M, c.shape[-1])], dim=-1)
        all_f = torch.cat([t(z) for t in self.towers], dim=-1)  # [B, M, N_FMT]
        sel = F.one_hot(ctx[:, 0], N_FMT).to(all_f.dtype)  # [B, N_FMT]
        return (all_f * sel.unsqueeze(1)).sum(-1)

    @staticmethod
    def mask(hmask, M):
        """allowed[b, q, k] for T = L + M tokens (history newest first, then candidates)."""
        B, L = hmask.shape
        dev = hmask.device
        T = L + M
        qi = torch.arange(T, device=dev)[:, None]
        ki = torch.arange(T, device=dev)[None, :]
        key_valid = torch.cat([hmask, torch.ones(B, M, dtype=torch.bool, device=dev)], dim=1)  # [B, T]
        hist_q = qi < L
        hist_k = ki < L
        # history -> same or older history; candidate -> all history and itself
        rule = (hist_q & hist_k & (ki >= qi)) | (~hist_q & hist_k) | (qi == ki)
        return rule[None] & (key_valid[:, None, :] | (qi == ki)[None])

    def forward(self, ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue):
        """Serving signature (contract.INPUTS): one request, N candidates -> pctr [N]."""
        args = _unbatched(ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue)
        z = self.logits(*args)[0]
        return torch.sigmoid(self.platt(z, ctx[0, 0]))


class LogRegCTR(nn.Module):
    """Logistic regression with crosses: candidate category x page context, x local hour, x format; history
    interest in the candidate's categories (share of recent ads and clicks); fatigue."""

    N_HIST = 5

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.ctx_w = nn.ModuleList([nn.Embedding(n, 1) for n in (N_FMT, N_CAT, N_REG, 25, 8, 25)])
        self.cat_w = nn.Embedding(N_CAT, 1, padding_idx=0)
        self.x_ctx = nn.Embedding(N_CAT * N_CAT, 1)
        self.x_hour = nn.Embedding(N_CAT * 25, 1)
        self.x_fmt = nn.Embedding(N_CAT * N_FMT, 1)
        self.hist = nn.Linear(self.N_HIST, 1, bias=False)
        self.fat = nn.Linear(1, 1, bias=False)
        for e in [*self.ctx_w, self.cat_w, self.x_ctx, self.x_hour, self.x_fmt]:
            nn.init.zeros_(e.weight)
        self.platt = PlattHead()

    def logits(self, ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue):
        B, M, _ = cand_cat.shape
        z = self.bias + sum(w(ctx[:, i]) for i, w in enumerate(self.ctx_w)).squeeze(-1)  # [B]
        z = z.unsqueeze(1).expand(B, M)
        cvalid = (cand_cat > 0).to(torch.float32)  # [B, M, 2]
        z = z + (self.cat_w(cand_cat).squeeze(-1) * cvalid).sum(-1)
        cc = cand_cat
        z = z + (self.x_ctx(cc * N_CAT + ctx[:, 1, None, None]).squeeze(-1) * cvalid).sum(-1)
        z = z + (self.x_hour(cc * 25 + ctx[:, 3, None, None]).squeeze(-1) * cvalid).sum(-1)
        z = z + (self.x_fmt(cc * N_FMT + ctx[:, 0, None, None]).squeeze(-1) * cvalid).sum(-1)
        # history: per candidate, how much of the recent history shares its categories
        hmask = (hist_fmt > 0).to(torch.float32)  # [B, L]
        n_hist = hmask.sum(-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        clicked = (hist_click == K.CLICK_YES).to(torch.float32)  # [B, L]
        same = ((hist_cat[:, None, :, :, None] == cand_cat[:, :, None, None, :])
                & (cand_cat[:, :, None, None, :] > 0)).any(-1).any(-1).to(torch.float32)  # [B, M, L]
        same = same * hmask[:, None, :]
        f = torch.stack([
            same.sum(-1) / n_hist,
            (same * clicked[:, None, :]).sum(-1),
            clicked.sum(-1, keepdim=True).expand(B, M),
            hmask.sum(-1, keepdim=True).expand(B, M) / K.HISTORY_LEN,
            torch.log1p((same * clicked[:, None, :]).sum(-1)),
        ], dim=-1)
        z = z + self.hist(f).squeeze(-1)
        # Event ages carry no weight here, but the input stays in the exported graph so both models share
        # one input contract and one serving path.
        z = z + 0.0 * hist_age_h.sum()
        return z + self.fat(torch.log1p(cand_fatigue).unsqueeze(-1)).squeeze(-1)

    def forward(self, ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue):
        args = _unbatched(ctx, hist_cat, hist_fmt, hist_click, hist_age_h, cand_cat, cand_fatigue)
        z = self.logits(*args)[0]
        return torch.sigmoid(self.platt(z, ctx[0, 0]))


MODELS = {"cadet": CadetCTR, "logreg": LogRegCTR}
