"""Recompute perplexity, bits/token and bits-per-byte from the logged mean loss and the token
and byte counts in metrics.json, and compare the loss of untrained models with ln V.

    python analysis/metrics_by_hand.py        (reads run records only; no GPU)
"""
from __future__ import annotations

import math

from common import CHAR_MODELS, SEEDS, metrics, save


def by_hand(m: dict) -> dict:
    loss = m["val_loss"]                         # mean nats per predicted token
    bits_per_token = loss / math.log(2)
    tokens_per_byte = m["val_tokens"] / m["val_bytes"]
    return {"loss_nats": loss, "ppl": math.exp(loss), "bits_per_token": bits_per_token,
            "bpb": bits_per_token * tokens_per_byte,
            "logged_ppl": m["val_ppl"], "logged_bpb": m["val_bpb"],
            "val_tokens": m["val_tokens"], "val_bytes": m["val_bytes"]}


def main():
    worked = by_hand(metrics("char_models", "gpt_modern", 1337))
    # the same identities on every char_models run (35), to show they hold in general
    ppl_gaps, bpb_gaps = [], []
    for name in CHAR_MODELS:
        for s in SEEDS:
            r = by_hand(metrics("char_models", name, s))
            ppl_gaps.append(abs(r["ppl"] - r["logged_ppl"]) / r["logged_ppl"])
            bpb_gaps.append(abs(r["bpb"] - r["logged_bpb"]))
    untrained = []
    for tok in ("char", "byte", "word2048", "gpt2"):
        for s in (1337, 42):
            m = metrics("untrained", f"init_{tok}", s)
            # vocabulary size = rows of the (tied) embedding: params minus non-embedding params / dim
            V = (m["params"] - m["params_non_embedding"]) // 256
            r = by_hand(m)
            untrained.append({"tokenizer": m["tokenizer"], "seed": s, "V": V, "lnV": math.log(V),
                              "loss": r["loss_nats"], "loss_minus_lnV": r["loss_nats"] - math.log(V),
                              "bpb": r["bpb"], "uniform_bpb": math.log2(V) * r["val_tokens"] / r["val_bytes"],
                              "bytes_per_token": r["val_bytes"] / r["val_tokens"], "steps": m["steps"]})
    save("metrics_by_hand", {"worked_example": worked, "n_runs": len(bpb_gaps),
                             "max_relative_ppl_gap": max(ppl_gaps), "max_abs_bpb_gap": max(bpb_gaps),
                             "untrained": untrained})
    print(worked)
    for u in untrained:
        print(u)


if __name__ == "__main__":
    main()
