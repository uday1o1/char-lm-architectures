#!/usr/bin/env bash
# Every analysis of the trained runs, plus the GRU check and the tests (GPU). Writes every
# results/*.json except summary.json, and figures/attention_*.png. Run after scripts/run_sweeps.sh,
# then run analysis/summarize.py and analysis/figures.py.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python analysis/metrics_by_hand.py
python analysis/untrained_init.py
python analysis/position_profile.py
python analysis/context_eval.py
python analysis/input_sensitivity.py
python analysis/train_vs_val.py
python analysis/attention_heads.py
python analysis/diversity.py
python analysis/gru_check.py
python -m pytest tests -q
bash scripts/attention_maps.sh
