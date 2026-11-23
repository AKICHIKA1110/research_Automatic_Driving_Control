#!/usr/bin/env python3
"""Show consecutive frames and the speed of one run.

  python scripts/preview_run.py data/v1/runs/run_00003.npz --start 40 --n 8 --out preview.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="preview.png")
    a = ap.parse_args()
    d = np.load(a.npz)
    fig = plt.figure(figsize=(2.0 * a.n, 5.6))
    for j in range(a.n):
        k = a.start + j
        ax = fig.add_subplot(2, a.n, j + 1)
        ax.imshow(d["bev"][k], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t={d['t'][k]:.1f}s\nv={d['v'][k]:.2f}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(d["t"], d["v"], label="v [m/s]")
    ax.plot(d["t"], d["kappa"], label="kappa [1/m]", alpha=0.6)
    ax.axvspan(d["t"][a.start], d["t"][a.start + a.n - 1], color="orange", alpha=0.2)
    ax.set_xlabel("t [s]"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)


if __name__ == "__main__":
    main()
