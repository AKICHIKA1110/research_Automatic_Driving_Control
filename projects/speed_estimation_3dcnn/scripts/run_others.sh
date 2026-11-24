#!/bin/bash
# Other sequence models vs 3D/2D CNN: cnn_lstm / convlstm / r2p1d / tsm / vivit, T=8, seeds 0-2.
# Usage: bash scripts/run_others.sh data/v1 results [epochs=20]
set -e
DATA=${1:?data dir}
OUT=${2:?results dir}
EP=${3:-20}
PY=${PY:-$HOME/venv_train/bin/python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
run() {   # model T seed
  local d="$OUT/$1_T$2_s$3"
  [ -f "$d/summary.json" ] && return
  $PY scripts/train.py "$DATA" --model "$1" --T "$2" --seed "$3" --epochs "$EP" --out "$d" > "$d.log" 2>&1
}
# one model at a time: each run keeps the whole dataset on the GPU (~4 GB), and r2p1d alone needs ~8 GB
for s in 0 1 2; do for m in cnn_lstm tsm vivit convlstm; do run $m 8 $s; done; done
for s in 0 1 2; do run r2p1d 8 $s; done
echo all done
