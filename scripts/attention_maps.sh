#!/usr/bin/env bash
# nanolm's attention figure for three kinds of text (modern Transformer, seed 1337). The texts are
# the ones analysis/attention_heads.py scores: Shakespeare dialogue, modern prose, and a random
# 40-letter string written twice.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
RUN=runs/sweeps/char_models/gpt_modern_s1337
text() { python -c "import sys; sys.path.insert(0, 'analysis'); import common, attention_heads as a; print(common.TEXTS['$1'] if '$1' in common.TEXTS else a.repeated(1000), end='')"; }
mkdir -p figures
for kind in dialogue prose repeated; do
  python -m nanolm.viz attention --run "$RUN" --text "$(text $kind)"
  mv docs/nanolm_figs/p1_attention_maps.png "figures/attention_$kind.png"
done
rmdir -p docs/nanolm_figs 2>/dev/null || true
