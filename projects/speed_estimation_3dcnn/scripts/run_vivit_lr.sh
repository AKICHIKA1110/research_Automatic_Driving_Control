#!/bin/bash
# ViViT with a Transformer-friendly learning rate (the shared lr=2e-3 collapses it to the mean speed).
# Usage: bash scripts/run_vivit_lr.sh data/v1 results [lr=3e-4]
DATA=${1:?data dir}; OUT=${2:?results dir}; LR=${3:-3e-4}
PY=${PY:-$HOME/venv_train/bin/python}
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  d="$OUT/vivit_T8_lr${LR}_s$s"
  [ -f "$d/summary.json" ] || $PY scripts/train.py "$DATA" --model vivit --T 8 --seed $s --lr $LR --epochs 20 --out "$d" > "$d.log" 2>&1
done
