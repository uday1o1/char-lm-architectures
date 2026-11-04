"""Shared helpers: locate runs, load a checkpoint with its data, write results.

Every script runs from the repository root after `source scripts/env.sh`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "sweeps"
RESULTS = ROOT / "results"
SEEDS = [1337, 42, 1, 2, 3]
CHAR_MODELS = ["rnn", "rnn_512", "gru", "lstm", "lstm_512", "gpt_textbook", "gpt_modern"]
# texts drawn with nanolm.viz and scored by analysis/attention_heads.py
TEXTS = {
    "dialogue": "ROMEO:\nBut soft! what light\nJULIET:\nAy me!",
    "prose": "She opened her laptop and wrote it all down.",
}


def run_dir(sweep: str, name: str, seed: int) -> Path:
    return RUNS / sweep / f"{name}_s{seed}"


def metrics(sweep: str, name: str, seed: int) -> dict:
    return json.loads((run_dir(sweep, name, seed) / "metrics.json").read_text())


def load(sweep: str, name: str, seed: int, device: str = "auto"):
    """Model in eval mode, its tokenizer, device info, token data and byte lookup tables."""
    from nanolm.data import TokenData, byte_lut_tensors, prepare_dataset
    from nanolm.generate import load_run
    cfg, model, tok, dinfo, _ = load_run(str(run_dir(sweep, name, seed)), device=device)
    meta = prepare_dataset(cfg.data)
    data = TokenData(meta, sampler=cfg.data.sampler, seed=0)
    return cfg, model, tok, dinfo, data, byte_lut_tensors(meta, dinfo.device)


def windows(tokens, T: int, max_windows: int = 0):
    """Non-overlapping (T+1)-token windows from the start of `tokens`, as in nanolm's evaluate()."""
    import torch
    t = torch.as_tensor(tokens.astype("int64"))
    n = (t.numel() - 1) // T
    if max_windows:
        n = min(n, max_windows)
    return torch.stack([t[i * T: i * T + T + 1] for i in range(n)])


def save(name: str, obj) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"[saved] results/{name}.json")
