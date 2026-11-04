"""How strongly the prediction at the end of a window depends on input characters k steps back:
the norm of d(loss at the last position) / d(input embedding at position T-1-k), averaged over
validation windows. For a vanilla RNN this gradient passes through k recurrent Jacobians and
should shrink quickly with k; gates (GRU, LSTM) and attention should keep it larger.

    python analysis/input_sensitivity.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from common import CHAR_MODELS, SEEDS, load, save, windows

N_WINDOWS = 256


def sensitivity(name: str, seed: int) -> list[float]:
    cfg, model, _, dinfo, data, _ = load("char_models", name, seed)
    model.float()
    emb = model.embed if hasattr(model, "embed") else model.tok_emb
    store = {}

    def hook(_m, _inp, out):                  # make the embedding output a leaf we can differentiate
        out = out.detach().requires_grad_(True)
        store["e"] = out
        return out

    h = emb.register_forward_hook(hook)
    win = windows(data.val, cfg.train.seq_len, N_WINDOWS).to(dinfo.device)
    total = torch.zeros(cfg.train.seq_len, dtype=torch.float64, device=dinfo.device)
    # cuDNN RNNs only differentiate in training mode; the native kernels work in eval mode
    with torch.backends.cudnn.flags(enabled=False):
        for b in range(0, len(win), 32):
            x, y = win[b: b + 32, :-1], win[b: b + 32, 1:]
            logits = model(x)
            F.cross_entropy(logits[:, -1].float(), y[:, -1], reduction="sum").backward()
            g = store["e"].grad.double().norm(dim=-1).sum(0)       # [T], summed over windows
            total += g.flip(0)                                    # index k = distance back from the end
            model.zero_grad(set_to_none=True)
    h.remove()
    return (total / len(win)).tolist()


def main():
    out = {}
    for name in CHAR_MODELS:
        for seed in SEEDS:
            out[f"{name}_s{seed}"] = s = sensitivity(name, seed)
            print(f"{name} s{seed}: k=0 {s[0]:.3e}  k=10 {s[10]:.3e}  k=50 {s[50]:.3e}  k=200 {s[200]:.3e}")
    save("input_sensitivity", out)


if __name__ == "__main__":
    main()
