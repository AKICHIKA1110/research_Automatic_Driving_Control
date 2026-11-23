"""Load runs into one GPU tensor and cut T-frame windows on the GPU.

key: "bev" (bird's-eye mask, 140x180) or "cam" (camera-view mask before the BEV warp, if saved).

A sample = frames k-T+1 .. k of one run (10 Hz), label = speed at frame k.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


class SplitData:
    def __init__(self, root, split: str, T: int, device="cuda", max_runs=None, key="bev"):
        root = Path(root)
        meta = pd.read_csv(root / "meta.csv")
        meta = meta[meta.split == split]
        if max_runs:
            meta = meta.head(max_runs)
        bevs, ends, rows = [], [], []
        offset = 0
        for m in meta.itertuples():
            d = np.load(root / "runs" / f"run_{m.run:05d}.npz")
            n = len(d["v"])
            if n < T:
                continue
            bevs.append(d[key])
            for k in range(T - 1, n):
                ends.append(offset + k)
                rows.append((m.run, k, float(d["v"][k]), float(d["accel"][k]), float(d["kappa"][k]),
                             bool(m.dashed), float(m.dash_period), bool(m.seams)))
            offset += n
        self.T = T
        self.frames = torch.from_numpy(np.concatenate(bevs)).to(device)          # uint8 [F,H,W]
        self.ends = torch.tensor(ends, device=device)
        self.info = pd.DataFrame(rows, columns=["run", "frame", "v", "accel", "kappa", "dashed", "dash_period",
                                                "seams"])
        self.v = torch.tensor(self.info.v.values, dtype=torch.float32, device=device)
        self._steps = torch.arange(-T + 1, 1, device=device)

    def __len__(self):
        return len(self.ends)

    def batch(self, idx: torch.Tensor, flip: bool = False):
        """x: float [B,T,H,W] in 0..1 (oldest first), y: speed [B]."""
        fi = self.ends[idx][:, None] + self._steps[None, :]
        x = self.frames[fi].float().div_(255.0)
        if flip:   # mirror left/right: speed unchanged
            m = torch.rand(len(idx), device=x.device) < 0.5
            x[m] = x[m].flip(-1)
        return x, self.v[idx]
