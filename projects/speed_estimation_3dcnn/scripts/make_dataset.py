#!/usr/bin/env python3
"""Generate the 10 Hz sequence dataset.

  python scripts/make_dataset.py data/v1 --runs 1200 --workers 18

Output:
  <out>/runs/run_00000.npz   bev uint8 [N,140,180], v, accel, kappa, s, t, e_y (float32 [N])
  <out>/meta.csv             one row per run (+ split = train/val/test, by run)
  <out>/config.yaml          the project config used (reproducibility)
"""
from __future__ import annotations

import argparse
import csv
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speedest.sequence import generate_run, load_project_config  # noqa: E402

CFG = None
OUT = None


def _init(cfg, out):
    global CFG, OUT
    CFG, OUT = cfg, out
    import cv2
    cv2.setNumThreads(1)   # one process per core; library threads only oversubscribe


def _one(args):
    run_id, seed = args
    frames, meta = generate_run(run_id, CFG, seed)
    np.savez_compressed(OUT / "runs" / f"run_{run_id:05d}.npz", **frames)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_project_config(a.config)
    d = cfg["dataset"]
    n = a.runs or d["runs"]
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "synth_full"}, f, sort_keys=False, allow_unicode=True)

    seeds = np.random.default_rng(d["seed"]).integers(2**31, size=n)
    t0 = time.time()
    metas = []
    with Pool(a.workers, initializer=_init, initargs=(cfg, out)) as pool:
        for i, m in enumerate(pool.imap_unordered(_one, [(i, int(s)) for i, s in enumerate(seeds)])):
            metas.append(m)
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"{i + 1}/{n} runs, {time.time() - t0:.0f} s", flush=True)
    metas.sort(key=lambda m: m["run"])

    # split by run (consecutive frames of one run are near duplicates)
    rng = np.random.default_rng(d["seed"] + 1)
    order = rng.permutation(n)
    n_val, n_test = int(n * d["val_frac"]), int(n * d["test_frac"])
    split = {}
    for j, r in enumerate(order):
        split[int(r)] = "val" if j < n_val else ("test" if j < n_val + n_test else "train")
    for m in metas:
        m["split"] = split[m["run"]]
    with open(out / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()))
        w.writeheader()
        w.writerows(metas)
    print(f"done: {sum(m['n_frames'] for m in metas)} frames -> {out}")


if __name__ == "__main__":
    main()
