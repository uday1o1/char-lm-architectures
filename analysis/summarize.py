"""Collect every summary number (means, paired tests, condensed analyses) into results/summary.json (no GPU).

Statistics follow nanolm.analyze: mean and sample std over seeds, and a paired t-test on the
per-seed differences (the same seeds in both arms). The seed noise floor is the median
within-configuration std of val_bpb over the char_models runs.

    python analysis/summarize.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from common import CHAR_MODELS, RESULTS, RUNS, SEEDS, metrics, save

TUNED_SEEDS = [4, 5, 6, 7, 8]
# nanolm's published table (TUTORIAL_nanolm.md, section 1.6.8; 2 seeds, 1337 and 42): mean val bpb
PUBLISHED = {"gpt_modern": 2.105, "lstm": 2.134, "gru": 2.167, "lstm_512": 2.171, "gpt_textbook": 2.189, "rnn": 2.436}

CONTEXT_T = [64, 128, 256, 512]


def ctx_metrics(arch: str, T: int, seed: int) -> dict:
    return metrics("char_models", arch, seed) if T == 256 else metrics("context", f"{arch}_T{T}", seed)


def vals(get, names) -> dict:
    return {n: np.array([get(n, s) for s in SEEDS]) for n in names}


def describe(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "values": x.tolist()}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    t = stats.ttest_rel(b, a)
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)), "p": float(t.pvalue),
            "n": len(d), "all_same_sign": bool((d > 0).all() or (d < 0).all())}


def curve(sweep: str, name: str) -> dict:
    """Mean and std over seeds of the periodic validation bpb, by training tokens."""
    rows = {}
    for s in SEEDS:
        for line in (RUNS / sweep / f"{name}_s{s}" / "eval_log.jsonl").read_text().splitlines():
            e = json.loads(line)
            rows.setdefault(e["train_tokens"], []).append(e["bpb"])
    tok = sorted(rows)
    a = np.array([rows[t] for t in tok])
    return {"tokens": tok, "mean": a.mean(1).tolist(), "std": a.std(1, ddof=1).tolist(), "per_seed": a.T.tolist()}


def context_eval() -> dict:
    """Fixed-context bpb (analysis/context_eval.py): gains from more context, paired over seeds."""
    C = json.loads((RESULTS / "context_eval.json").read_text())
    ce = lambda n, c: np.array([C[f"{n}_s{s}"]["bpb"][str(c)] for s in SEEDS])
    names = CHAR_MODELS + [f"{a}_T{t}" for a in ("lstm", "gpt_modern") for t in (64, 128, 512)]
    res = {"curves": {n: {c: describe(ce(n, int(c))) for c in C[f"{n}_s1337"]["bpb"]} for n in names}}
    cs = [16, 32, 64, 128, 256, 512]
    res["steps_T512"] = {a: {f"{x}->{y}": paired(ce(f"{a}_T512", x), ce(f"{a}_T512", y)) for x, y in zip(cs, cs[1:])}
                         for a in ("lstm", "gpt_modern")}
    gain = {a: ce(f"{a}_T512", 512) - ce(f"{a}_T512", 64) for a in ("lstm", "gpt_modern")}
    res["gain_64_to_512"] = {a: describe(g) for a, g in gain.items()}
    res["gain_gpt_minus_lstm"] = paired(gain["lstm"], gain["gpt_modern"])
    # does training on longer windows help at equal context? the T=64 and T=512 models at c=64
    at64 = lambda a, T: ce(a if T == 256 else f"{a}_T{T}", 64)
    res["train_T_at_c64"] = {a: {T: describe(at64(a, T)) for T in CONTEXT_T} for a in ("lstm", "gpt_modern")}
    res["T64_vs_T512_at_c64"] = {a: paired(at64(a, 64), at64(a, 512)) for a in ("lstm", "gpt_modern")}
    res["gain_beyond_64"] = {n: float(np.mean([C[f"{n}_s{s}"]["bpb"]["64"] - min(v for c, v in C[f"{n}_s{s}"]["bpb"].items()
                                                                               if int(c) >= 64) for s in SEEDS]))
                             for n in names}
    res["gpt_minus_lstm_at_c"] = {c: paired(ce("lstm_T512", c), ce("gpt_modern_T512", c)) for c in (1, 8, 64, 512)}
    return {"context_eval": res}


def analyses() -> dict:
    """Condensed numbers from the other analysis files."""
    J = lambda n: json.loads((RESULTS / f"{n}.json").read_text())
    out = {}
    tv = J("train_vs_val")
    out["train_vs_val"] = {n: {k: float(np.mean([tv[f"{n}_s{s}"][k] for s in SEEDS])) for k in ("train_bpb", "val_bpb", "gap")}
                           for n in CHAR_MODELS + ["lstm_512_drop0.2", "lstm_512_drop0.3"]}
    pp = J("position_profile")
    out["position_profile_check"] = max(abs(v["bpb"] - v["logged_val_bpb"]) for v in pp.values())
    prof = lambda n: np.mean([pp[f"{n}_s{s}"]["bpb_by_position"] for s in SEEDS], 0)
    out["edge_effect"] = {n: {"position_0": float(prof(n)[0]), "positions_0_7": float(prof(n)[:8].mean()),
                              "positions_64_on": float(prof(n)[64:].mean()), "all": float(prof(n).mean())}
                          for n in ("lstm", "gpt_modern")}
    se = J("input_sensitivity")
    out["sensitivity"] = {n: {k: float(np.mean([se[f"{n}_s{s}"][k] for s in SEEDS])) for k in (0, 1, 10, 50, 100, 200, 255)}
                          for n in CHAR_MODELS}
    A = J("attention_heads")
    att = {}
    for feat in ("prev", "word_start", "line_start"):
        tops = []
        for s in SEEDS:
            a = np.array(A[f"s{s}"]["validation"][feat])
            l, h = np.unravel_index(a.argmax(), a.shape)
            tops.append({"seed": s, "layer": int(l), "head": int(h), "score": float(a.max()),
                         "ablation_delta_bpb": float(np.array(A[f"s{s}"]["ablation"]["delta_bpb"])[l, h])})
        prev_top = [np.unravel_index(np.array(A[f"s{s}"]["validation"]["prev"]).argmax(), (4, 4)) for s in SEEDS]
        att[feat] = {"prev_head_score": [float(np.array(A[f"s{s}"]["validation"][feat])[ph]) for s, ph in zip(SEEDS, prev_top)],
                     "tops": tops, "score": describe(np.array([t["score"] for t in tops])),
                     "uniform": A["s1337"]["validation"]["uniform"][feat],
                     "texts": {k: [float(np.max(A[f"s{s}"][k][feat])) if A[f"s{s}"][k][feat] is not None else None
                                   for s in SEEDS] for k in ("dialogue", "prose")}}
    dist = [np.array(A[f"s{s}"]["validation"]["distance"]) for s in SEEDS]
    lines = [np.array(A[f"s{s}"]["validation"]["line_start"]) for s in SEEDS]
    att["longest_lookback_is_line_head"] = [bool(d.argmax() == l.argmax()) for d, l in zip(dist, lines)]
    att["longest_lookback"] = describe(np.array([d.max() for d in dist]))
    att["uniform_distance"] = A["s1337"]["validation"]["uniform"]["distance"]
    abl = [np.array(A[f"s{s}"]["ablation"]["delta_bpb"]) for s in SEEDS]
    att["most_damaging_is_line_head"] = [bool(a.argmax() == l.argmax()) for a, l in zip(abl, lines)]
    att["ablation_max"] = describe(np.array([a.max() for a in abl]))
    att["ablation_min"] = describe(np.array([a.min() for a in abl]))
    ind = [A[f"s{s}"]["induction"] for s in SEEDS]
    att["induction"] = {"max_score": describe(np.array([np.max(i["score"]) for i in ind])), "uniform": ind[0]["uniform"],
                        "bits_first_copy": describe(np.array([i["bits_first_copy"] for i in ind])),
                        "bits_second_copy": describe(np.array([i["bits_second_copy"] for i in ind]))}
    out["attention"] = att
    U, M = J("untrained_init"), J("metrics_by_hand")
    out["untrained"] = []
    for tok in ("char", "byte", "word:2048", "gpt2"):
        u = [r for r in U if r["tokenizer"] == tok]
        m = [r for r in M["untrained"] if r["tokenizer"] == tok]
        out["untrained"].append({"tokenizer": tok, "V": u[0]["V"], "lnV": u[0]["lnV"],
                                 "loss_step0": [r["loss_init"] for r in u], "loss_step1": [r["loss"] for r in m],
                                 "logit_var": [r["logit_var"] for r in u],
                                 "lse_minus_lnV": [r["sample_logsumexp"] - r["lnV"] for r in u],
                                 "z_target_minus_mean": [r["sample_z_target"] - r["logit_mean"] for r in u],
                                 "bpb_step0": [r["bpb_init"] for r in u], "uniform_bpb": m[0]["uniform_bpb"],
                                 "bytes_per_token": m[0]["bytes_per_token"],
                                 "param_change_max": max(r["one_step_max_abs_param_change"] for r in u)})
    out["metrics_by_hand"] = {k: M[k] for k in ("worked_example", "n_runs", "max_relative_ppl_gap", "max_abs_bpb_gap")}
    D = J("diversity")
    div = {"human": float(np.mean([np.mean(h["distinct4"]) for h in D["human"]]))}
    for kind in ("temperature", "top_p"):
        rows = [r for r in D["settings"] if r["kind"] == kind]
        div[kind] = {str(x): {"distinct4": describe(np.array([np.mean(r["distinct4"]) for r in rows if r["value"] == x])),
                              "model_bits": float(np.mean([r["model_bits_per_char"] for r in rows if r["value"] == x]))}
                     for x in sorted({r["value"] for r in rows})}
    out["diversity"] = div
    out["gru_check"] = J("gru_check")
    return out


def main():
    out = {}
    bpb = vals(lambda n, s: metrics("char_models", n, s)["val_bpb"], CHAR_MODELS)
    m0 = {n: metrics("char_models", n, 1337) for n in CHAR_MODELS}
    out["char_models"] = {n: {**describe(bpb[n]), "params": m0[n]["params"],
                              "params_non_embedding": m0[n]["params_non_embedding"],
                              "tokens_seen": m0[n]["tokens_seen"],
                              "flops_6ND": 6 * m0[n]["params_non_embedding"] * m0[n]["tokens_seen"],
                              "train_min": float(np.mean([metrics("char_models", n, s)["train_time_sec"]
                                                          for s in SEEDS]) / 60),
                              "two_seed_mean": float(bpb[n][:2].mean())} for n in CHAR_MODELS}
    out["reproduction"] = {n: {"ours_2_seeds": float(bpb[n][:2].mean()), "published": v,
                               "diff": float(bpb[n][:2].mean()) - v} for n, v in PUBLISHED.items()}
    out["reproduction_max_abs_diff"] = max(abs(r["diff"]) for r in out["reproduction"].values())
    config_means = {}
    for p in RUNS.glob("*/summary.csv"):
        df = pd.read_csv(p)
        for name, g in df.groupby("name"):
            config_means[f"{p.parent.name}/{name}"] = float(g["val_bpb"].mean())
    out["lowest_config_mean"] = min(config_means.items(), key=lambda kv: kv[1])
    out["noise_floor"] = float(np.median([bpb[n].std(ddof=1) for n in CHAR_MODELS]))
    pairs = [("gpt_textbook", n) for n in CHAR_MODELS if n != "gpt_textbook"] + [
        ("rnn", "rnn_512"), ("rnn_512", "lstm"), ("rnn_512", "gru"), ("rnn", "gru"), ("gru", "lstm"),
        ("lstm", "lstm_512"), ("lstm", "gpt_modern")]
    out["paired"] = {f"{a}|{b}": paired(bpb[a], bpb[b]) for a, b in pairs}
    # Holm's step-down correction over the six comparisons with the GPT-2-style Transformer
    fam = [k for k in out["paired"] if k.startswith("gpt_textbook|")]
    order = sorted(fam, key=lambda k: out["paired"][k]["p"])
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * out["paired"][k]["p"]))
        out["paired"][k]["p_holm"] = running
    out["curves"] = {n: curve("char_models", n) for n in CHAR_MODELS}
    for n in ("lstm_512_drop0.2", "lstm_512_drop0.3"):
        out["curves"][n] = curve("lstm512_dropout", n)
    # best point of each mean curve (where validation bpb bottoms out)
    out["curve_minimum"] = {n: {"tokens": c["tokens"][int(np.argmin(c["mean"]))], "bpb": float(min(c["mean"])),
                                "final": c["mean"][-1]} for n, c in out["curves"].items()}
    drop = vals(lambda n, s: metrics("lstm512_dropout", n, s)["val_bpb"], ("lstm_512_drop0.2", "lstm_512_drop0.3"))
    out["dropout"] = {n: {**describe(v), "vs_lstm_512": paired(bpb["lstm_512"], v),
                          "vs_lstm": paired(bpb["lstm"], v)} for n, v in drop.items()}
    ctx = {}
    for arch in ("lstm", "gpt_modern"):
        v = {T: np.array([ctx_metrics(arch, T, s)["val_bpb"] for s in SEEDS]) for T in CONTEXT_T}
        ctx[arch] = {"by_T": {T: describe(v[T]) for T in CONTEXT_T},
                     "steps": {f"{a}->{b}": paired(v[a], v[b]) for a, b in zip(CONTEXT_T, CONTEXT_T[1:])},
                     "tokens_seen": {T: ctx_metrics(arch, T, 1337)["tokens_seen"] for T in CONTEXT_T}}
    ctx["gpt_minus_lstm"] = {T: paired(np.array([ctx_metrics("lstm", T, s)["val_bpb"] for s in SEEDS]),
                                       np.array([ctx_metrics("gpt_modern", T, s)["val_bpb"] for s in SEEDS]))
                             for T in CONTEXT_T}
    out["context"] = ctx
    lr = {}
    for n in CHAR_MODELS:                      # 2-seed means; None = the run diverged (no metrics.json)
        lr[n] = {"1x": float(bpb[n][:2].mean())}
        for tag in ("lr0.25x", "lr0.5x", "lr2x", "lr4x", "lr8x", "lr16x"):
            if (RUNS / "lr_check" / f"{n}_{tag}_s1337").exists():
                ok = all((RUNS / "lr_check" / f"{n}_{tag}_s{s}" / "metrics.json").exists() for s in (1337, 42))
                lr[n][tag[2:]] = float(np.mean([metrics("lr_check", f"{n}_{tag}", s)["val_bpb"]
                                                for s in (1337, 42)])) if ok else None
        lr[n] = dict(sorted(lr[n].items(), key=lambda kv: float(kv[0][:-1])))
        lr[n]["best"] = min((k for k in lr[n] if lr[n][k] is not None), key=lambda k: lr[n][k])
    out["lr_check"] = lr
    tuned = {n: np.array([metrics("char_models_tuned", n, s)["val_bpb"] for s in TUNED_SEEDS]) for n in CHAR_MODELS}
    lr_of = lambda n: yaml.safe_load((RUNS / "char_models_tuned" / f"{n}_s4" / "config.yaml").read_text())["train"]["lr"]
    out["tuned"] = {n: {**describe(tuned[n]), "lr": lr_of(n)} for n in CHAR_MODELS}
    out["tuned_paired"] = {f"{a}|{b}": paired(tuned[a], tuned[b]) for a, b in
                           [("gpt_modern", n) for n in CHAR_MODELS if n != "gpt_modern"] +
                           [("rnn_512", "lstm"), ("rnn_512", "gru"), ("gru", "lstm"), ("lstm", "lstm_512")]}
    out["tuned_noise_floor"] = float(np.median([tuned[n].std(ddof=1) for n in CHAR_MODELS]))
    out.update(context_eval())
    out.update(analyses())
    out["devices"] = sorted({metrics("char_models", n, s)["device"] for n in CHAR_MODELS for s in SEEDS})
    meta = json.loads((RUNS / "char_models" / "gpt_modern_s1337" / "meta.json").read_text())
    out["nanolm_git"] = meta["git"]
    out["torch"] = meta["torch"]
    out["total_train_hours"] = sum(json.loads(p.read_text())["train_time_sec"]
                                   for p in RUNS.rglob("metrics.json")) / 3600
    out["n_runs"] = len(list(RUNS.rglob("metrics.json")))
    save("summary", out)
    print(json.dumps({k: out[k] for k in ("noise_floor", "devices", "n_runs", "total_train_hours")}, indent=1))
    for n in CHAR_MODELS:
        c = out["char_models"][n]
        print(f"{n:14s} {c['mean']:.4f} ± {c['std']:.4f}  2-seed {c['two_seed_mean']:.4f}  params {c['params']:,}")


if __name__ == "__main__":
    main()
