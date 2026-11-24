#!/usr/bin/env python3
"""Train one speed regressor and evaluate it on the test runs.

  python scripts/train.py data/v1 --model c3d --T 8 --epochs 20 --out results/c3d_T8

Writes <out>/best.pt, <out>/history.csv, <out>/test_pred.csv (one row per test window, with the
condition columns for the analysis) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speedest.data import SplitData  # noqa: E402
from speedest.models import build, count_macs, count_params  # noqa: E402


@torch.no_grad()
def predict(model, data: SplitData, bs=512):
    model.eval()
    out = []
    for i in range(0, len(data), bs):
        idx = torch.arange(i, min(i + bs, len(data)), device=data.ends.device)
        x, _ = data.batch(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x).float())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--input", default="bev", choices=["bev", "cam"], help="bird's-eye or camera-view mask")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=None, help="use only the first N runs per split (debug)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tr = SplitData(a.data, "train", a.T, max_runs=a.max_runs, key=a.input)
    va = SplitData(a.data, "val", a.T, max_runs=a.max_runs, key=a.input)
    te = SplitData(a.data, "test", a.T, max_runs=a.max_runs, key=a.input)
    print(f"windows train {len(tr)} val {len(va)} test {len(te)} (loaded in {time.time() - t0:.0f} s)", flush=True)

    model = build(a.model, a.T)
    n_params, macs = count_params(model), count_macs(model, a.T, *tr.frames.shape[1:])
    cl = getattr(model, "channels_last_ok", a.model != "c3d")   # only for models without 5-D (3D conv) weights
    model = model.cuda().to(memory_format=torch.channels_last) if cl else model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    best, hist = float("inf"), []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), device="cuda")
        tl, n = 0.0, 0
        te0 = time.time()
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            x, y = tr.batch(perm[i:i + a.bs], flip=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)
        pv = predict(model, va)
        mae = (pv - va.v).abs().mean().item()
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_mae=mae, sec=time.time() - te0))
        print(f"ep {ep + 1:2d} loss {tl / n:.4f} val MAE {mae:.4f} m/s ({time.time() - te0:.0f} s)", flush=True)
        if mae < best:
            best = mae
            torch.save(model.state_dict(), out / "best.pt")

    import pandas as pd
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    model.load_state_dict(torch.load(out / "best.pt"))
    pt = predict(model, te).cpu().numpy()
    info = te.info.copy()
    info["pred"] = pt
    info["err"] = info.pred - info.v
    info.to_csv(out / "test_pred.csv", index=False)
    summ = dict(model=a.model, T=a.T, input=a.input, seed=a.seed, epochs=a.epochs, lr=a.lr, params=n_params, macs=macs, best_val_mae=best,
                test_mae=float(info.err.abs().mean()), test_rmse=float(np.sqrt((info.err ** 2).mean())),
                train_windows=len(tr), test_windows=len(te), minutes=(time.time() - t0) / 60)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
