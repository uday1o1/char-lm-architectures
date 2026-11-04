"""GRUScratch must compute the same outputs, final states and gradients as torch.nn.GRU.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from gru_scratch import GRUScratch  # noqa: E402


def pair(reset_after=True, layers=2, D=7, H=11):
    torch.manual_seed(0)
    ref = nn.GRU(D, H, num_layers=layers, batch_first=True).double()
    return ref, GRUScratch(D, H, layers, reset_after).double().load_from(ref)


def test_forward_and_gradients_match():
    ref, mine = pair()
    x = torch.randn(4, 30, 7, dtype=torch.float64, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    out_ref, h_ref = ref(x)
    out, h = mine(x2)
    assert torch.allclose(out, out_ref, atol=1e-12, rtol=0)
    assert torch.allclose(h, h_ref, atol=1e-12, rtol=0)
    (out_ref.square().sum() + h_ref.sum()).backward()
    (out.square().sum() + h.sum()).backward()
    assert torch.allclose(x2.grad, x.grad, atol=1e-10, rtol=0)
    for i, cell in enumerate(mine.cells):
        assert torch.allclose(cell.W_h.weight.grad, getattr(ref, f"weight_hh_l{i}").grad, atol=1e-10, rtol=0)
        assert torch.allclose(cell.W_i.bias.grad, getattr(ref, f"bias_ih_l{i}").grad, atol=1e-10, rtol=0)


def test_reset_before_product_is_a_different_function():
    ref, mine = pair(reset_after=False)
    x = torch.randn(4, 30, 7, dtype=torch.float64)
    assert (mine(x)[0] - ref(x)[0]).abs().max() > 1e-3
