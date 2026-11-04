"""Bits per byte at a fixed amount of context, on the same characters for every model.

nanolm's evaluation cuts the validation text into windows of the training length T, so a model
with a longer T is scored on a different mix of short and long contexts. Here every model scores
the same 16,384 validation characters, each predicted from exactly c preceding characters
(c = 1, 2, 4, ..., up to the model's training length). That separates two questions: how much a
model gains from more context, and whether training on longer windows helps at equal context.

    python analysis/context_eval.py
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from common import CHAR_MODELS, SEEDS, load, save
from nanolm.data import count_bytes
from nanolm.device import autocast_ctx

N_TARGETS, MAX_C = 16384, 512
CONTEXTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
RUNS = [("char_models", n) for n in CHAR_MODELS] + [
    ("context", f"{a}_T{t}") for a in ("lstm", "gpt_modern") for t in (64, 128, 512)]


@torch.no_grad()
def at_context(model, val, targets, c, luts, dinfo) -> float:
    nats = nbytes = 0.0
    for b in range(0, len(targets), 256):
        j = torch.as_tensor(targets[b:b + 256])
        win = torch.stack([torch.as_tensor(val[k - c:k + 1]) for k in j.tolist()]).to(dinfo.device)
        x, y = win[:, :-1], win[:, -1]
        with autocast_ctx(dinfo):
            logits = model(x)[:, -1]
        nats += F.cross_entropy(logits.float(), y, reduction="sum").item()
        nbytes += count_bytes(x[:, -1], y, luts).double().sum().item()
    return nats / math.log(2) / nbytes


def main():
    out = {}
    for sweep, name in RUNS:
        for seed in SEEDS:
            cfg, model, _, dinfo, data, luts = load(sweep, name, seed)
            val = data.val.astype("int64")
            targets = np.linspace(MAX_C, len(val) - 1, N_TARGETS).astype(int)   # same for every model
            T = cfg.train.seq_len
            out[f"{name}_s{seed}"] = r = {"T": T, "bpb": {c: at_context(model, val, targets, c, luts, dinfo)
                                                         for c in CONTEXTS if c <= T}}
            print(name, seed, {c: round(v, 4) for c, v in r["bpb"].items()})
    save("context_eval", out)


if __name__ == "__main__":
    main()
