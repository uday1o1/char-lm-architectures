"""The loss of a model at initialisation (no optimizer step at all), split into its two
parts, loss = mean logsumexp(z) - mean z_target. For logits that are Gaussian across the
vocabulary with mean m and variance s^2 and unrelated to the target, the first part is about
ln V + m + s^2 / 2 and the second about m, so the loss would be about ln V + s^2 / 2.

The same seed rebuilds exactly the weights the `untrained` sweep started from (train.py seeds
before building the model and draws no other torch random numbers first), so the step-0 loss here
and the logged loss after one optimizer step come from the same initial weights; the script
records the largest parameter difference to the saved one-step checkpoint (about 1e-5, one AdamW
step at learning rate 1e-5).

    python analysis/untrained_init.py         (GPU recommended for the GPT-2 vocabulary)
"""
from __future__ import annotations

import math

import torch

from common import load, save
from nanolm.device import seed_everything
from nanolm.evaluate import evaluate
from nanolm.models import build_model


@torch.no_grad()
def main():
    rows = []
    for tok in ("char", "byte", "word2048", "gpt2"):
        for seed in (1337, 42):
            cfg, stepped, _, dinfo, data, luts = load("untrained", f"init_{tok}", seed)
            seed_everything(seed)
            model = build_model(cfg.model).to(dinfo.device).eval()
            T = cfg.train.seq_len
            r = evaluate(model, data.val_windows(T), T, 32, luts, dinfo)
            x = torch.as_tensor(data.val[: 32 * T].astype("int64")).view(32, T).to(dinfo.device)
            z = model(x).float()                                   # [32, T, V], fp32
            var = z.var(dim=-1).mean().item()                      # spread across the vocabulary
            y = torch.as_tensor(data.val[1: 32 * T + 1].astype("int64")).view(32, T).to(dinfo.device)
            lse = torch.logsumexp(z, -1).mean().item()
            z_target = z.gather(-1, y[..., None]).mean().item()
            moved = [(a - b).abs() for a, b in zip(stepped.parameters(), model.parameters())]
            V = cfg.model.vocab_size
            rows.append({"tokenizer": cfg.data.tokenizer, "seed": seed, "V": V, "lnV": math.log(V),
                         "loss_init": r["loss"], "loss_minus_lnV": r["loss"] - math.log(V),
                         "logit_var": var, "logit_mean": z.mean().item(), "lnV_plus_half_var": math.log(V) + var / 2,
                         "sample_logsumexp": lse, "sample_z_target": z_target, "sample_loss": lse - z_target,
                         "bpb_init": r["bpb"],
                         "one_step_max_abs_param_change": max(m.max().item() for m in moved),
                         "one_step_mean_abs_param_change": (sum(m.sum().item() for m in moved)
                                                            / sum(m.numel() for m in moved))})
            print(rows[-1])
    save("untrained_init", rows)


if __name__ == "__main__":
    main()
