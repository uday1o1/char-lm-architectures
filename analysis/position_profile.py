"""Loss as a function of position inside the evaluation window, for every char_models and context run.

Evaluation (as in nanolm) cuts the validation text into non-overlapping windows of the training
length T, and each window starts from an empty context. The prediction at position p (p = 0..T-1)
is therefore made from exactly p+1 characters, so the per-position loss shows how much each model
gains from more context. Averaged over positions, it reproduces the logged val_bpb.

    python analysis/position_profile.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common import CHAR_MODELS, SEEDS, load, metrics, save, windows
from nanolm.data import count_bytes
from nanolm.device import autocast_ctx

CONTEXT = [f"{a}_T{t}" for a in ("lstm", "gpt_modern") for t in (64, 128, 512)]


@torch.no_grad()
def profile(sweep: str, name: str, seed: int) -> dict:
    cfg, model, _, dinfo, data, luts = load(sweep, name, seed)
    T = cfg.train.seq_len
    win = windows(data.val, T).to(dinfo.device)
    nats = torch.zeros(T, dtype=torch.float64, device=dinfo.device)
    nbytes = torch.zeros(T, dtype=torch.float64, device=dinfo.device)
    for b in range(0, len(win), 32):
        x, y = win[b: b + 32, :-1], win[b: b + 32, 1:]
        with autocast_ctx(dinfo):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        nats += loss.view_as(y).double().sum(0)
        nbytes += count_bytes(x, y, luts).double().sum(0)
    bpb = (nats / math.log(2) / nbytes).tolist()
    total = (nats.sum() / math.log(2) / nbytes.sum()).item()
    return {"T": T, "n_windows": len(win), "bpb_by_position": bpb, "bpb": total,
            "logged_val_bpb": metrics(sweep, name, seed)["val_bpb"]}


def main():
    out = {}
    for sweep, names in (("char_models", CHAR_MODELS), ("context", CONTEXT)):
        for name in names:
            for seed in SEEDS:
                r = profile(sweep, name, seed)
                out[f"{name}_s{seed}"] = r
                print(f"{name} s{seed}: bpb {r['bpb']:.4f} (logged {r['logged_val_bpb']:.4f})")
    save("position_profile", out)


if __name__ == "__main__":
    main()
