#!/usr/bin/env python3
"""Generate the synthetic dataset as .npz shards.

  python3 scripts/generate_dataset.py --out data/synth_v1 --num 100000 --workers 8

Each shard (shard_XXXXX.npz) holds `shard_size` samples:
  bev      uint8 (N, H, W)   BEV line-probability mask (input of the curvature CNN)
  cam      uint8 (N, h, w)   camera-view mask, downscaled (only with --save-camera)
  y_pts    float32 (N, K)    lane centerline y at stations_x [m] (base_link)
  vis      uint8 (N, K)      1 = point exists and is visible
  e_y, e_psi, v, delta  float32 (N,)
  kappa    float32 (N, M)    curvature at preview_s [1/m]
  seq      int32 (N,)        driving sequence id (unique per worker; use for train/val split)
  worker   int32 (N,)
The config used is saved as config.yaml in the output directory.
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lane_synth import Generator, load_config  # noqa: E402

KEYS_F = ["e_y", "e_psi", "v", "delta"]


def work(args):
    cfg, out, worker, shard_ids, shard_size, seed, save_cam, cam_scale = args
    g = Generator(cfg, seed=seed + 1000 * worker, keep_camera=save_cam)
    it = g.sequences()
    for sid in shard_ids:
        buf = {k: [] for k in ["bev", "cam", "y_pts", "vis", "kappa", "seq", *KEYS_F]}
        for _ in range(shard_size):
            s = next(it)
            buf["bev"].append(s["bev"])
            if save_cam:
                buf["cam"].append(cv2.resize(s["cam"], None, fx=cam_scale, fy=cam_scale,
                                             interpolation=cv2.INTER_AREA))
            for k in ["y_pts", "vis", "kappa", "seq", *KEYS_F]:
                buf[k].append(s[k])
        arrs = dict(bev=np.stack(buf["bev"]), y_pts=np.stack(buf["y_pts"]).astype(np.float32),
                    vis=np.stack(buf["vis"]).astype(np.uint8), kappa=np.stack(buf["kappa"]).astype(np.float32),
                    seq=np.asarray(buf["seq"], np.int32), worker=np.full(shard_size, worker, np.int32))
        for k in KEYS_F:
            arrs[k] = np.asarray(buf[k], np.float32)
        if save_cam:
            arrs["cam"] = np.stack(buf["cam"])
        np.savez_compressed(out / f"shard_{sid:05d}.npz", **arrs)
        print(f"[worker {worker}] shard {sid} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=100000)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-camera", action="store_true", help="also store camera-view masks")
    ap.add_argument("--camera-scale", type=float, default=0.5)
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    n_shards = (a.num + a.shard_size - 1) // a.shard_size
    jobs = [(cfg, out, w, list(range(w, n_shards, a.workers)), a.shard_size, a.seed,
             a.save_camera, a.camera_scale) for w in range(min(a.workers, n_shards))]
    t = time.time()
    if len(jobs) == 1:
        work(jobs[0])
    else:
        with mp.Pool(len(jobs)) as pool:
            pool.map(work, jobs)
    print(f"{n_shards * a.shard_size} samples in {time.time() - t:.0f} s → {out}")


if __name__ == "__main__":
    main()
