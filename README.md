# Recurrent networks against Transformers on Tiny Shakespeare

A controlled comparison of character-level language models (vanilla RNN, GRU, LSTM, a GPT-2-style Transformer and a modern one with RMSNorm, RoPE and SwiGLU). It asks four questions:

1. What do the evaluation metrics measure?
2. How do the architectures compare at the same training budget, and how robust is the ranking?
3. How much context does each architecture actually use?
4. What do the attention heads do?

All training uses [nanolm](https://github.com/lkk688/DeepDataMiningLearning/tree/main/DeepDataMiningLearning/llm) at commit `8418fc1`, unmodified, and the analysis scripts load its checkpoints and call its functions. Every experiment is a sweep file in `configs/sweeps/`, and every number below is computed from the run records in `runs/` and the analysis outputs in `results/`.

## Results

All results are validation bits per byte (bpb; lower is better), as mean ± std over 5 seeds. Every model is trained on the same 16.4M characters (2000 steps × 8,192 characters: 32 windows of 256, or 8,192/T windows of length T in the `context` sweep).

| model | parameters | bpb, nanolm learning rate | bpb, best learning rate (fresh seeds) |
|---|---:|---:|---:|
| RNN 2×256 | 280,320 | 2.439 ± 0.004 | 2.249 ± 0.004 (4×) |
| RNN 2×512 | 1,084,928 | 2.221 ± 0.007 | 2.163 ± 0.004 (2×) |
| GRU 2×256 | 806,656 | 2.162 ± 0.003 | 2.104 ± 0.006 (4×) |
| LSTM 2×256 | 1,069,824 | 2.137 ± 0.009 | 2.102 ± 0.009 (8×) |
| LSTM 2×512 | 4,236,800 | 2.174 ± 0.006 | 2.121 ± 0.004 (0.5×) |
| Transformer, GPT-2 style | 3,438,336 | 2.195 ± 0.013 | 2.147 ± 0.005 (2×) |
| Transformer, modern | 3,162,624 | 2.106 ± 0.010 | 2.115 ± 0.007 (2×) |

- **Reproduction.** At nanolm's own settings, the first two seeds (1337, 42) reproduce the published table of nanolm's tutorial to within 0.01 bpb for each of its six models; the largest difference is 0.009, for the 2×512 LSTM.
- **Robustness of the ranking.** The modern Transformer's lead depends on the learning rate.
  - At nanolm's learning rates, the modern Transformer is best, ahead of the LSTM (paired p = 0.018) and the GRU (p = 0.0003).
  - At each model's best learning rate from a grid of multiples of nanolm's rate (0.5× to 8× for most models, 0.25× to 2× for the 2×512 LSTM, up to 16× for the 2×256 LSTM; chosen on two seeds, then re-measured on five fresh seeds), the LSTM and GRU are 0.013 and 0.011 bpb lower than the modern Transformer. Neither difference is resolved (paired p = 0.10 and 0.06), so at the best learning rates the order of the top three is open.
- **What holds at either learning rate:**
  - The 2×256 vanilla RNN is last.
  - A gap of between 0.058 and 0.085 bpb to the GRU and the LSTM remains even when the RNN has as many parameters as the LSTM (RNN 2×512; paired p ≤ 0.0002 at both learning-rate settings).
  - The modern Transformer beats the GPT-2-style one.
- **The larger LSTM overfits.** At nanolm's settings, the 2×512 LSTM reaches its best validation bpb (2.130) after 8.2M characters and then gets worse. Its gap between training and validation bpb is 0.77, against 0.35 for the 2×256 LSTM. With dropout 0.3 instead of 0.1 it reaches 2.072, below every model at nanolm's dropout (only this model's dropout was varied, over 0.2 and 0.3, on the same seeds and validation split, so 2.072 is slightly optimistic).
- **Context.** Scored on the same 16,384 characters with a fixed amount of context, every model gets almost all of its gain from context within the first 64 characters.
  - From 64 to 512 characters, the LSTM gains 0.0004 bpb and the modern Transformer 0.006.
  - The usual "bpb against training length T" curve falls mainly because evaluation windows of length T start with no context. At 64 characters of context, the difference between models trained on windows of 64 and 512 characters is not resolved (paired p = 0.85 for the LSTM, 0.32 for the modern Transformer).
- **Attention.** Every one of the five seeds of the modern Transformer has the same three kinds of head:
  - a previous-character head (0.50 to 0.62 of its weight on position t−1, where a uniform head would put 0.02);
  - a word-start head (0.28 to 0.36 on the first letter of the current word, counted only at least two letters into the word, where the previous-character head puts 0.02 to 0.10);
  - a line-start head (0.30 to 0.48 on the last newline, counted at least two characters after it). It is the head that looks back furthest in 4 of the 5 seeds, and the head whose removal costs most in 4 of the 5.

  There are no copying heads: on a random 40-letter string written twice, the best head in each seed puts 0.007 to 0.025 of its weight on the character that followed the current one last time (a uniform head would put 0.017), and the models predict the second copy no better than the first.

## Layout

| path | contents |
|---|---|
| `configs/sweeps/` | the six sweep files: `char_models`, `char_models_tuned`, `lr_check`, `lstm512_dropout`, `context`, `untrained` |
| `runs/sweeps/<sweep>/<run>_s<seed>/` | nanolm's record of every run: `config.yaml`, `meta.json` (git commit, device, command), `train_log.jsonl`, `eval_log.jsonl`, `metrics.json`, `sample.txt`, `stdout.log` (no checkpoints; the two diverged runs, `lr_check/gru_lr8x_s*`, have no `metrics.json` or `sample.txt`) |
| `runs/sweeps/<sweep>/summary.{csv,md}`, `curves.png`, `bars.png`, `analyze_stdout.txt`, `sweep.yaml` | `python -m nanolm.analyze` output per sweep and its printed table, and the sweep file as nanolm.sweep copied it |
| `analysis/` | analysis scripts (see below) and the gate-by-gate GRU (`gru_scratch.py`) |
| `results/` | every analysis output as JSON, and `environment.txt` (machine and package versions); `summary.json` holds the numbers quoted above |
| `figures/` | all figures |
| `tests/` | checks that `GRUScratch` matches `torch.nn.GRU` |

## Reproducing

Requires Python 3.10 or later and one NVIDIA GPU. Training used one RTX 3090 with bf16 autocast, PyTorch 2.6.0, CUDA 12.4 and Python 3.11; `requirements.txt` pins the package versions.

```bash
bash scripts/setup.sh          # clone nanolm at commit 8418fc1 into external/, install requirements
bash scripts/run_sweeps.sh     # train every run (finished runs are skipped), then nanolm.analyze per sweep
bash scripts/run_analysis.sh   # analyses of the trained runs (GPU), the GRU check and tests -> results/*.json, attention maps
source scripts/env.sh
python analysis/summarize.py        # results/summary.json (no GPU)
python analysis/figures.py          # figures/*.png except attention_*.png (no GPU)
```

- **Time:** the sum of per-run training times is 4.7 hours. Runs shared the GPU six at a time, so the per-run times in `metrics.json` include that sharing and are not single-run timings.
- **Without a GPU:** checkpoints are not stored, but `summarize.py` and `figures.py` rebuild every summary number and every figure except the three attention maps (`figures/attention_*.png`, which need the checkpoints) from the committed `runs/` and `results/`.

| script | what it measures |
|---|---|
| `metrics_by_hand.py` | loss, perplexity, bits per token and bpb recomputed from the counts in `metrics.json` |
| `untrained_init.py` | loss at initialisation against ln V for four tokenizers, and what one optimizer step changes |
| `position_profile.py` | bpb by position inside nanolm's evaluation windows |
| `context_eval.py` | bpb on fixed targets with exactly c characters of context |
| `input_sensitivity.py` | gradient of the last prediction with respect to inputs k characters back |
| `train_vs_val.py` | bpb on training text against validation text for each final model |
| `attention_heads.py` | per-head attention scores over the validation set, a copying test on repeated random strings, and the effect of removing each head |
| `figures.py` | every figure except the attention maps; `scripts/attention_maps.sh` draws those with `nanolm.viz` |
| `diversity.py` | distinct 4-gram fraction of samples against temperature and top-p |
| `gru_check.py`, `gru_scratch.py` | a GRU written gate by gate, compared numerically with `torch.nn.GRU` |

## Limitations

- **Data.** Tiny Shakespeare is a single 1.1M-character corpus with one validation split (its last 10%). The learning rates in `char_models_tuned` were chosen on that same split, which may flatter the tuned numbers slightly.
- **Tuning.** Only the learning rate was tuned. Dropout and weight decay keep nanolm's per-architecture values, and dropout 0.3 helped the 2×512 LSTM by more than any learning rate.
- **Scope.** All conclusions are at a budget of 16.4M characters and models of 0.28M to 4.24M parameters.
- **Statistics.** p-values are paired t-tests over 5 seeds. They are not corrected for multiple comparisons, except the Holm-adjusted values (`p_holm` in `results/summary.json`) for the six comparisons with the GPT-2-style Transformer, under which the 2×512 RNN and 2×512 LSTM differences are no longer below 0.05.
