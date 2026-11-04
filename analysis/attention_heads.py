"""What the heads of the modern character Transformer attend to, measured with the same
attention_maps() function that `python -m nanolm.viz attention` draws.

For every head, averaged over all query positions of all validation windows:
  prev        weight on the previous character (position t-1)
  word_start  weight on the first letter of the current word, for queries at least two letters
              after it (so the target is never also the previous character); words that start
              before the window are skipped, since their first letter is not visible
  line_start  weight on the most recent newline, for queries at least two characters after it
  distance    mean look-back distance, sum_s A[t,s] (t-s)
Each score is compared with a uniform causal head, which puts 1/(t+1) on every position <= t.

Induction: on a random 40-letter string repeated twice, weight from positions t = 40..78 of the
second copy on position t-39, the character that followed the current one last time.

Ablation: validation bpb with one head's output zeroed (before the output projection).

    python analysis/attention_heads.py
"""
from __future__ import annotations

import random
import string

import numpy as np
import torch

from common import SEEDS, TEXTS, load, save, windows
from nanolm.evaluate import evaluate
from nanolm.viz import attention_maps

REPEAT_LEN = 40


def repeated(seed: int) -> str:
    rng = random.Random(seed)
    r = "".join(rng.choice(string.ascii_lowercase) for _ in range(REPEAT_LEN))
    return r + r


def features(chars: list[str], maps: list[np.ndarray]) -> dict:
    """Sums of each score over valid queries, per layer and head, plus the uniform reference."""
    T = len(chars)
    t = np.arange(T)
    ws = np.full(T, -1)                       # first letter of the word containing t
    nl = np.full(T, -1)                       # last newline strictly before t
    for i in range(T):
        if chars[i].isalpha() or chars[i] == "'":
            ws[i] = ws[i - 1] if i > 0 and ws[i - 1] >= 0 else i
        nl[i] = nl[i - 1] if i > 0 else -1
        if i > 0 and chars[i - 1] == "\n":
            nl[i] = i - 1
    q_word = np.where((ws >= 1) & (ws < t - 1))[0]      # target is not t-1, which the prev score covers
    q_line = np.where((nl >= 0) & (nl < t - 1))[0]
    q_all = t[1:]
    out = {"n": {"prev": len(q_all), "word_start": len(q_word), "line_start": len(q_line)},
           "uniform": {"prev": float((1 / (q_all + 1)).sum()), "word_start": float((1 / (q_word + 1)).sum()),
                       "line_start": float((1 / (q_line + 1)).sum()), "distance": float((q_all / 2).sum())}}
    for name in ("prev", "word_start", "line_start", "distance"):
        out[name] = []
    for A in maps:                             # A: [H, T, T]
        out["prev"].append(A[:, q_all, q_all - 1].sum(-1))
        out["word_start"].append(A[:, q_word, ws[q_word]].sum(-1))
        out["line_start"].append(A[:, q_line, nl[q_line]].sum(-1))
        out["distance"].append((A[:, q_all, :] * (q_all[:, None] - t[None, :]).clip(min=0)).sum((-1, -2)))
    for name in ("prev", "word_start", "line_start", "distance"):
        out[name] = np.stack(out[name])        # [L, H]
    return out


def normalise(acc: dict) -> dict:
    """Means over valid queries; None where a text has no such query (e.g. no newline)."""
    count = lambda k: acc["n"].get(k, acc["n"]["prev"])
    res = {k: (acc[k] / count(k)).tolist() if count(k) else None
           for k in ("prev", "word_start", "line_start", "distance")}
    res["uniform"] = {k: v / count(k) if count(k) else None for k, v in acc["uniform"].items()}
    res["n"] = dict(acc["n"])
    return res


def merge(a: dict | None, b: dict) -> dict:
    if a is None:
        return b
    for k in ("prev", "word_start", "line_start", "distance"):
        a[k] = a[k] + b[k]
    for k in a["n"]:
        a["n"][k] += b["n"][k]
    for k in a["uniform"]:
        a["uniform"][k] += b["uniform"][k]
    return a


@torch.no_grad()
def analyse(seed: int) -> dict:
    cfg, model, tok, dinfo, data, luts = load("char_models", "gpt_modern", seed)
    model.float()
    T = cfg.train.seq_len
    res = {}
    acc = None                                 # validation set, all windows
    for w in windows(data.val, T):
        x = w[:-1]
        acc = merge(acc, features([tok.decode([i]) for i in x.tolist()], attention_maps(model, x[None].to(dinfo.device))))
    res["validation"] = normalise(acc)
    for kind, text in TEXTS.items():
        ids = tok.encode(text)
        maps = attention_maps(model, torch.tensor([ids], device=dinfo.device))
        res[kind] = normalise(features(list(text), maps))
        res[kind]["maps"] = [m.round(4).tolist() for m in maps]         # [L][H][T][T], for figures.py
    # induction on repeated random strings
    scores, first, second = [], [], []
    for k in range(100):
        text = repeated(1000 + k)
        idx = torch.tensor([tok.encode(text)], device=dinfo.device)
        maps = attention_maps(model, idx)
        q = np.arange(REPEAT_LEN, 2 * REPEAT_LEN - 1)
        scores.append(np.stack([A[:, q, q - REPEAT_LEN + 1].mean(-1) for A in maps]))
        logp = torch.log_softmax(model(idx[:, :-1]).float(), -1)[0]
        nll = -logp.gather(-1, idx[0, 1:, None])[:, 0] / np.log(2)
        first.append(nll[: REPEAT_LEN - 1].mean().item())
        second.append(nll[REPEAT_LEN - 1:].mean().item())
    res["induction"] = {"score": np.mean(scores, 0).tolist(),
                        "uniform": float(np.mean(1 / (np.arange(REPEAT_LEN, 2 * REPEAT_LEN - 1) + 1))),
                        "bits_first_copy": float(np.mean(first)), "bits_second_copy": float(np.mean(second))}
    # ablation: zero one head's output before the output projection
    base = evaluate(model, data.val_windows(T), T, 32, luts, dinfo)["bpb"]
    hd = model.blocks[0].attn.hd
    abl = []
    for l, blk in enumerate(model.blocks):
        row = []
        for h in range(blk.attn.H):
            def pre(_m, args, h=h):
                y = args[0].clone()
                y[..., h * hd:(h + 1) * hd] = 0
                return (y,)
            hk = blk.attn.proj.register_forward_pre_hook(pre)
            row.append(evaluate(model, data.val_windows(T), T, 32, luts, dinfo)["bpb"] - base)
            hk.remove()
        abl.append(row)
    res["ablation"] = {"base_bpb": base, "delta_bpb": abl}
    return res


def main():
    out = {}
    for seed in SEEDS:
        out[f"s{seed}"] = analyse(seed)
        v = out[f"s{seed}"]["validation"]
        print(seed, "max prev", np.max(v["prev"]), "max word_start", np.max(v["word_start"]),
              "max line_start", np.max(v["line_start"]), "max induction", np.max(out[f"s{seed}"]["induction"]["score"]))
    save("attention_heads", out)


if __name__ == "__main__":
    main()
