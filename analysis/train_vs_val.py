"""Generalisation gap of each final model: bits-per-byte on training text against validation text,
both measured the same way (eval mode, so no dropout; non-overlapping windows of the training
length). The training sample is the first stretch of the training split with as many tokens as
the validation split.

    python analysis/train_vs_val.py
"""
from __future__ import annotations

import numpy as np

from common import CHAR_MODELS, SEEDS, load, save
from nanolm.evaluate import evaluate

RUNS = [("char_models", n) for n in CHAR_MODELS] + [("lstm512_dropout", n) for n in
                                                    ("lstm_512_drop0.2", "lstm_512_drop0.3")]


def main():
    out = {}
    for sweep, name in RUNS:
        for seed in SEEDS:
            cfg, model, _, dinfo, data, luts = load(sweep, name, seed)
            T = cfg.train.seq_len
            train = np.asarray(data.train[0][: data.val.size])
            tr = evaluate(model, train, T, 32, luts, dinfo)
            va = evaluate(model, data.val_windows(T), T, 32, luts, dinfo)
            out[f"{name}_s{seed}"] = {"train_bpb": tr["bpb"], "val_bpb": va["bpb"], "gap": va["bpb"] - tr["bpb"]}
            print(name, seed, out[f"{name}_s{seed}"])
    save("train_vs_val", out)


if __name__ == "__main__":
    main()
