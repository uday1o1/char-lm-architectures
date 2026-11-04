"""Sample diversity (fraction of distinct character 4-grams) against temperature and
top-p, with the model fixed, so its validation bpb is fixed. Sampling uses nanolm's sample_next().

Per setting and model seed: 16 samples of 512 characters, each continuing a 32-character prompt
taken from the validation text (the same prompts and random streams for every setting). Each
sample is also scored by the model itself at temperature 1 (bits per character), which shows what
the extra diversity costs in likelihood. Human reference: 512-character stretches of validation text.

    python analysis/diversity.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common import SEEDS, load, save
from nanolm.evaluate import sample_next

N, PROMPT, LENGTH = 16, 32, 512
TEMPERATURES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
TOP_P = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


def distinct4(s: str) -> float:
    grams = [s[i:i + 4] for i in range(len(s) - 3)]
    return len(set(grams)) / len(grams)


@torch.no_grad()
def sample(model, prompts, ctx, temperature, top_p, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = prompts.clone()
    for _ in range(LENGTH):
        logits = model(idx[:, -ctx:])[:, -1, :]
        idx = torch.cat([idx, sample_next(logits, temperature, 0, top_p, g)], dim=1)
    return idx


@torch.no_grad()
def bits_per_char(model, idx, ctx):
    """Model's own bits per generated character, scored with the last `ctx` characters as context."""
    total = 0.0
    for t in range(PROMPT, idx.size(1)):
        logp = F.log_softmax(model(idx[:, max(0, t - ctx):t])[:, -1].float(), -1)
        total += -logp.gather(-1, idx[:, t:t + 1]).sum().item()
    return total / (idx.size(0) * (idx.size(1) - PROMPT)) / math.log(2)


def main():
    out = {"settings": [], "human": []}
    for seed in SEEDS:
        cfg, model, tok, dinfo, data, _ = load("char_models", "gpt_modern", seed)
        model.float()
        ctx = cfg.train.seq_len
        val = data.val.astype("int64")
        starts = [i * (len(val) // N) for i in range(N)]
        prompts = torch.tensor([val[s:s + PROMPT] for s in starts], device=dinfo.device)
        human = [tok.decode(val[s + PROMPT:s + PROMPT + LENGTH].tolist()) for s in starts]
        out["human"].append({"seed": seed, "distinct4": [distinct4(h) for h in human]})
        settings = [("temperature", t, t, 1.0) for t in TEMPERATURES] + [("top_p", p, 1.0, p) for p in TOP_P]
        for kind, value, temperature, top_p in settings:
            idx = sample(model, prompts, ctx, temperature, top_p, seed=7)
            texts = [tok.decode(row[PROMPT:].tolist()) for row in idx]
            out["settings"].append({"seed": seed, "kind": kind, "value": value,
                                    "distinct4": [distinct4(s) for s in texts],
                                    "model_bits_per_char": bits_per_char(model, idx, ctx),
                                    "example": texts[0][:200]})
            print(seed, kind, value, sum(out["settings"][-1]["distinct4"]) / N, out["settings"][-1]["model_bits_per_char"])
    save("diversity", out)


if __name__ == "__main__":
    main()
