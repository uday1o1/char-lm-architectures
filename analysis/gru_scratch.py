"""A GRU written out gate by gate, with the same equations and parameter layout as torch.nn.GRU.

    r  = sigma(W_ir x + b_ir + W_hr h + b_hr)        reset gate
    z  = sigma(W_iz x + b_iz + W_hz h + b_hz)        update gate
    n  = tanh (W_in x + b_in + r * (W_hn h + b_hn))  candidate state
    h' = (1 - z) * n + z * h

nn.GRU applies the reset gate after the recurrent matrix product, r * (W_hn h + b_hn). The
original formulation (Cho et al., 2014) applies it before, W_hn (r * h); set reset_after=False
for that variant, which computes a different function from the same weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUCellScratch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, reset_after: bool = True):
        super().__init__()
        self.hidden, self.reset_after = hidden, reset_after
        self.W_i = nn.Linear(in_dim, 3 * hidden)      # rows ordered r, z, n as in nn.GRU
        self.W_h = nn.Linear(hidden, 3 * hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        H = self.hidden
        xr, xz, xn = self.W_i(x).chunk(3, dim=-1)
        if self.reset_after:
            hr, hz, hn = self.W_h(h).chunk(3, dim=-1)
            r = torch.sigmoid(xr + hr)
            z = torch.sigmoid(xz + hz)
            n = torch.tanh(xn + r * hn)
        else:
            W, b = self.W_h.weight, self.W_h.bias
            r = torch.sigmoid(xr + h @ W[:H].T + b[:H])
            z = torch.sigmoid(xz + h @ W[H:2 * H].T + b[H:2 * H])
            n = torch.tanh(xn + (r * h) @ W[2 * H:].T + b[2 * H:])
        return (1 - z) * n + z * h


class GRUScratch(nn.Module):
    """Multi-layer, batch-first GRU. forward(x [B, T, D]) -> (outputs [B, T, H], h_n [L, B, H])."""

    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, reset_after: bool = True):
        super().__init__()
        self.cells = nn.ModuleList([GRUCellScratch(in_dim if i == 0 else hidden, hidden, reset_after)
                                    for i in range(num_layers)])

    def forward(self, x: torch.Tensor):
        finals = []
        for cell in self.cells:
            h = x.new_zeros(x.size(0), cell.hidden)
            outs = []
            for t in range(x.size(1)):
                h = cell(x[:, t], h)
                outs.append(h)
            x = torch.stack(outs, dim=1)
            finals.append(h)
        return x, torch.stack(finals)

    @torch.no_grad()
    def load_from(self, gru: nn.GRU) -> "GRUScratch":
        for i, cell in enumerate(self.cells):
            cell.W_i.weight.copy_(getattr(gru, f"weight_ih_l{i}"))
            cell.W_i.bias.copy_(getattr(gru, f"bias_ih_l{i}"))
            cell.W_h.weight.copy_(getattr(gru, f"weight_hh_l{i}"))
            cell.W_h.bias.copy_(getattr(gru, f"bias_hh_l{i}"))
        return self
