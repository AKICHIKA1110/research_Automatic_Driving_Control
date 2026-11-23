#!/bin/bash
# All experiments of the 2D vs 3D comparison (two queues share the GPU).
#   main:  T=8, single2d / early2d / early2d_wide / c3d, seeds 0-2
#   sweep: T=4 and 16, early2d / c3d, seed 0
# Usage: bash scripts/run_all.sh data/v1 results [epochs=20]
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

queue1() { for s in 0 1 2; do run c3d 8 $s; run single2d 8 $s; done; run c3d 4 0; run c3d 16 0; }
queue2() { for s in 0 1 2; do run early2d 8 $s; run early2d_wide 8 $s; done; run early2d 4 0; run early2d 16 0; }
queue1 & queue2 & wait
echo all done
