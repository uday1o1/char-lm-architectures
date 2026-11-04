"""Size of the differences between GRUScratch and nn.GRU (fp64 and fp32), and
between nn.GRU and the reset-before-product variant.

    python analysis/gru_check.py
"""
import torch
import torch.nn as nn

from common import save
from gru_scratch import GRUScratch


def diffs(dtype, reset_after=True, seed=0):
    torch.manual_seed(seed)
    ref = nn.GRU(256, 256, num_layers=2, batch_first=True).to(dtype)
    mine = GRUScratch(256, 256, 2, reset_after).to(dtype).load_from(ref)
    x = torch.randn(8, 128, 256, dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    (o1, h1), (o2, h2) = ref(x), mine(x2)
    o1.square().sum().backward()
    o2.square().sum().backward()
    return {"max_abs_output": (o1 - o2).abs().max().item(), "max_abs_final_state": (h1 - h2).abs().max().item(),
            "max_abs_input_grad": (x.grad - x2.grad).abs().max().item(), "output_scale": o1.abs().max().item()}


def main():
    torch.set_num_threads(4)
    out = {"fp64": diffs(torch.float64), "fp32": diffs(torch.float32),
           "fp64_reset_before_product": diffs(torch.float64, reset_after=False)}
    print(out)
    save("gru_check", out)


if __name__ == "__main__":
    main()
