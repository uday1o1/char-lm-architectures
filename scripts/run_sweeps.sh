#!/usr/bin/env bash
# Train every run (GPU). SLOTS = how many runs share one GPU at a time (default 6).
# Finished runs are skipped, so the script can be re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
GPUS=$(python -c "print(','.join(['0'] * int('${SLOTS:-6}')))")
for s in untrained char_models context lstm512_dropout lr_check char_models_tuned; do
  python -m nanolm.sweep --sweep configs/sweeps/$s.yaml --gpus "$GPUS"
done
for s in untrained char_models context lstm512_dropout lr_check char_models_tuned; do
  python -m nanolm.analyze runs/sweeps/$s --metric val_bpb --baseline gpt_textbook > runs/sweeps/$s/analyze_stdout.txt
done
