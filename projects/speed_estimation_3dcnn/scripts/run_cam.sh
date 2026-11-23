#!/bin/bash
# BEV vs camera-view input: same runs, same models (early2d / c3d, T=8, seeds 0-2), input = camera-view mask.
# Usage: bash scripts/run_cam.sh data/v1cam results [epochs=20]
set -e
DATA=${1:?data dir}
OUT=${2:?results dir}
EP=${3:-20}
PY=${PY:-$HOME/venv_train/bin/python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
[ -f "$DATA/meta.csv" ] || $PY scripts/make_dataset.py "$DATA" --config config_cam.yaml --workers 18 > "$OUT/make_v1cam.log" 2>&1

run() {   # model T seed
  local d="$OUT/$1_T$2_cam_s$3"
  [ -f "$d/summary.json" ] && return
  $PY scripts/train.py "$DATA" --model "$1" --T "$2" --seed "$3" --input cam --epochs "$EP" --out "$d" > "$d.log" 2>&1
}
queue1() { for s in 0 1 2; do run c3d 8 $s; done; }
queue2() { for s in 0 1 2; do run early2d 8 $s; done; }
queue1 & queue2 & wait
echo all done
