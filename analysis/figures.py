"""Draw every figure except figures/attention_*.png (scripts/attention_maps.sh) from results/*.json
(no GPU).

    python analysis/figures.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CHAR_MODELS, RESULTS, ROOT, SEEDS, TEXTS

FIG = ROOT / "figures"
LABEL = {"rnn": "RNN 2x256", "rnn_512": "RNN 2x512", "gru": "GRU 2x256", "lstm": "LSTM 2x256",
         "lstm_512": "LSTM 2x512", "gpt_textbook": "Transformer, GPT-2 style", "gpt_modern": "Transformer, modern",
         "lstm_512_drop0.2": "LSTM 2x512, dropout 0.2", "lstm_512_drop0.3": "LSTM 2x512, dropout 0.3"}
COLOR = {"rnn": "#8c8c8c", "rnn_512": "#4d4d4d", "gru": "#1baf7a", "lstm": "#2a78d6", "lstm_512": "#6250d6",
         "gpt_textbook": "#eda100", "gpt_modern": "#e34948", "lstm_512_drop0.2": "#9a8ee8", "lstm_512_drop0.3": "#c6bff3"}
J = lambda n: json.loads((RESULTS / f"{n}.json").read_text())
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e4", "legend.frameon": False, "lines.linewidth": 1.6})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] figures/{name}.png")


def learning_curves(S):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for n in CHAR_MODELS:
        c = S["curves"][n]
        x, m, s = np.array(c["tokens"]) / 1e6, np.array(c["mean"]), np.array(c["std"])
        ax.plot(x, m, color=COLOR[n], label=LABEL[n])
        ax.fill_between(x, m - s, m + s, color=COLOR[n], alpha=0.2, lw=0)
    ax.set_ylim(2.05, 3.5)
    ax.set_xlabel("training characters seen (millions)")
    ax.set_ylabel("validation bits per byte")
    ax.legend(fontsize=6.3, ncol=2, loc="upper right")
    ax.set_title("(a) validation bpb, mean of 5 seeds (band = 1 std)", loc="left", fontsize=8.5)
    ax = axes[1]
    order = sorted(CHAR_MODELS, key=lambda n: S["char_models"][n]["mean"])
    for i, n in enumerate(order):
        c, t = S["char_models"][n], S["tuned"][n]
        ax.errorbar(c["mean"], i - 0.15, xerr=c["std"], fmt="o", color=COLOR[n], ms=4, capsize=2,
                    label="nanolm learning rate" if i == 0 else None)
        ax.errorbar(t["mean"], i + 0.15, xerr=t["std"], fmt="s", color=COLOR[n], mfc="white", ms=4, capsize=2,
                    label="best learning rate (fresh seeds)" if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[n] for n in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("final validation bits per byte")
    ax.legend(fontsize=6.3, loc="upper right")
    ax.set_title("(b) final bpb, mean ± std over 5 seeds", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "char_models")


def overfitting(S, tv):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    for n in ("lstm", "lstm_512", "lstm_512_drop0.2", "lstm_512_drop0.3"):
        c = S["curves"][n]
        x, m = np.array(c["tokens"]) / 1e6, np.array(c["mean"])
        ax.plot(x, m, color=COLOR[n], label=LABEL[n])
        k = int(np.argmin(m))
        ax.plot(x[k], m[k], "v", color=COLOR[n], ms=5)
    ax.set_ylim(2.05, 2.35)
    ax.set_xlabel("training characters seen (millions)")
    ax.set_ylabel("validation bits per byte")
    ax.legend(fontsize=7)
    ax.set_title("(a) validation curves; triangle = minimum", loc="left", fontsize=8.5)
    ax = axes[1]
    names = ["rnn", "rnn_512", "gru", "lstm", "lstm_512", "lstm_512_drop0.2", "lstm_512_drop0.3", "gpt_textbook", "gpt_modern"]
    for i, n in enumerate(names):
        tr = [tv[f"{n}_s{s}"]["train_bpb"] for s in SEEDS]
        va = [tv[f"{n}_s{s}"]["val_bpb"] for s in SEEDS]
        ax.plot([np.mean(tr), np.mean(va)], [i, i], color=COLOR[n], lw=2)
        ax.plot(np.mean(tr), i, "o", color=COLOR[n], ms=4, mfc="white")
        ax.plot(np.mean(va), i, "o", color=COLOR[n], ms=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABEL[n] for n in names], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("bpb: open = training text, filled = validation")
    ax.set_title("(b) generalisation gap of the final models", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "overfitting")


def context(S):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    Ts = [64, 128, 256, 512]
    for arch, lab in (("lstm", "LSTM 2x256"), ("gpt_modern", "Transformer, modern")):
        m = [S["context"][arch]["by_T"][str(T)]["mean"] for T in Ts]
        s = [S["context"][arch]["by_T"][str(T)]["std"] for T in Ts]
        ax.errorbar(Ts, m, yerr=s, color=COLOR[arch], marker="o", ms=4, capsize=2, label=f"{lab}: nanolm eval")
        c = S["context_eval"]["train_T_at_c64"][arch]
        ax.errorbar(Ts, [c[str(T)]["mean"] for T in Ts], yerr=[c[str(T)]["std"] for T in Ts], color=COLOR[arch],
                    marker="s", ms=3.5, capsize=2, ls="--", lw=1.2, label=f"{lab}: 64 chars of context")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.set_xlabel("training context length T (characters)")
    ax.set_ylabel("validation bits per byte")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("(a) bpb against training length T", loc="left", fontsize=8.5)
    ax = axes[1]
    for n, lab in (("rnn", "RNN 2x256 (T=256)"), ("gru", "GRU 2x256 (T=256)"),
                   ("lstm_T512", "LSTM 2x256 (T=512)"), ("gpt_modern_T512", "Transformer, modern (T=512)")):
        c = S["context_eval"]["curves"][n]
        xs = sorted(int(k) for k in c)
        col = COLOR[n.replace("_T512", "")]
        ax.errorbar(xs, [c[str(x)]["mean"] for x in xs], yerr=[c[str(x)]["std"] for x in xs], color=col,
                    marker="o", ms=3, capsize=1.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_ylim(2.0, 2.95)
    ax.set_xlabel("characters of context c")
    ax.set_ylabel("bits per byte")
    ax.legend(fontsize=6.3)
    ax.set_title("(b) bpb at fixed context c, same 16,384 targets", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "context")


def sensitivity(sens):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for n in ("rnn", "rnn_512", "gru", "lstm", "gpt_modern"):
        g = np.mean([sens[f"{n}_s{s}"] for s in SEEDS], 0)
        ax.plot(np.arange(len(g)), g, color=COLOR[n], label=LABEL[n])
    ax.set_yscale("log")
    ax.set_xlabel("distance k back from the last input character")
    ax.set_ylabel("mean ‖∂loss / ∂input embedding‖")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "sensitivity")


def diversity(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    human = np.mean([np.mean(h["distinct4"]) for h in D["human"]])
    for ax, kind, xl in ((axes[0], "temperature", "temperature τ (top-p off)"), (axes[1], "top_p", "top-p (τ = 1)")):
        rows = [r for r in D["settings"] if r["kind"] == kind]
        xs = sorted({r["value"] for r in rows})
        per = [[np.mean(r["distinct4"]) for r in rows if r["value"] == x] for x in xs]
        ax.errorbar(xs, [np.mean(p) for p in per], yerr=[np.std(p, ddof=1) for p in per], marker="o", ms=4,
                    capsize=2, color="#2a78d6", label="samples, mean over 5 models")
        ax.axhline(human, color="#4d4d4d", ls="--", lw=1, label="validation text")
        ax.set_xlabel(xl)
        ax2 = ax.twinx()
        ax2.plot(xs, [np.mean([r["model_bits_per_char"] for r in rows if r["value"] == x]) for x in xs],
                 color="#e34948", marker="s", ms=3, lw=1, label="model's bits per sampled char")
        ax2.set_ylabel("model's own bits per char", color="#e34948", fontsize=7.5)
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
    axes[0].set_ylabel("distinct 4-gram fraction")
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "diversity")


def ablation(A):
    # head numbering is arbitrary per seed, so show one model: the seed whose maps are drawn
    d = np.array(A["s1337"]["ablation"]["delta_bpb"])
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    im = ax.imshow(d, cmap="Blues")
    for (l, h), v in np.ndenumerate(d):
        ax.text(h, l, f"{v:.3f}", ha="center", va="center", fontsize=6.5, color="white" if v > d.max() / 2 else "black")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(d.shape[1]))
    ax.set_yticks(range(d.shape[0]))
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ val bpb when zeroed")
    fig.tight_layout()
    save(fig, "head_ablation")


def head_types(A):
    """The top head of each type (seed 1337, scored on the validation set) on the dialogue text."""
    r = A["s1337"]
    text = TEXTS["dialogue"]
    labels = [c.replace("\n", "⏎").replace(" ", "␣") for c in text]
    kinds = [("prev", "previous character"), ("word_start", "first letter of the word"), ("line_start", "last newline")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (k, name) in zip(axes, kinds):
        a = np.array(r["validation"][k])
        l, h = np.unravel_index(a.argmax(), a.shape)
        ax.imshow(np.array(r["dialogue"]["maps"][l][h]), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=4.2, family="monospace")
        ax.set_yticklabels(labels, fontsize=4.2, family="monospace")
        ax.tick_params(length=0, pad=1)
        ax.grid(False)
        ax.set_title(f"L{l}H{h}: {name}", fontsize=8)
    axes[0].set_ylabel("query (predicting position)", fontsize=7)
    axes[1].set_xlabel("key (attended position)", fontsize=7)
    fig.tight_layout()
    save(fig, "head_types")


def main():
    S = J("summary")
    learning_curves(S)
    overfitting(S, J("train_vs_val"))
    context(S)
    sensitivity(J("input_sensitivity"))
    diversity(J("diversity"))
    ablation(J("attention_heads"))
    head_types(J("attention_heads"))


if __name__ == "__main__":
    main()
