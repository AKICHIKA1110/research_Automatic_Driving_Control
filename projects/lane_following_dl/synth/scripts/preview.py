#!/usr/bin/env python3
"""Generate a few samples and save a preview sheet.

Each row: course top view (vehicle, camera footprint) | camera-view line mask |
BEV mask with the label points (green = visible, red = not visible).

  python3 scripts/preview.py --n 6 --out preview.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lane_synth import Generator, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=25, help="samples to skip between rows")
    ap.add_argument("--out", default="preview.png")
    a = ap.parse_args()

    cfg = load_config(a.config)
    g = Generator(cfg, a.seed)
    it = g.sequences()
    st = np.asarray(cfg["labels"]["stations_x"])
    fig, axes = plt.subplots(a.n, 3, figsize=(15, 4.2 * a.n),
                             gridspec_kw=dict(width_ratios=[1.1, 1.3, 0.9]))
    for r in range(a.n):
        for _ in range(a.skip):
            next(it)
        s = next(it)
        d = s["_debug"]
        course, path, (x, y, yaw) = d["course"], d["path"], d["pose"]
        ax = axes[r, 0]
        for o in course.boundary_offsets():
            p = course.center.offset(o)
            ax.plot(p.xy[:, 0], p.xy[:, 1], "k-", lw=1)
        if course.dashed_center:
            ax.plot(course.center.xy[:, 0], course.center.xy[:, 1], "k--", lw=0.6)
        ax.plot(path.xy[:, 0], path.xy[:, 1], "c-", lw=0.6)
        ax.arrow(x, y, 0.25 * np.cos(yaw), 0.25 * np.sin(yaw), width=0.03, color="r")
        c, sn = np.cos(yaw), np.sin(yaw)
        b = cfg["bev"]
        roi = np.array([[b["x_min"], b["y_min"]], [b["x_max"], b["y_min"]], [b["x_max"], b["y_max"]],
                        [b["x_min"], b["y_max"]], [b["x_min"], b["y_min"]]])
        ax.plot(x + c * roi[:, 0] - sn * roi[:, 1], y + sn * roi[:, 0] + c * roi[:, 1], "m-", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"seq {s['seq']}  v={s['v']:.2f} m/s  lane={course.lane_width:.2f} m")

        ax = axes[r, 1]
        ax.imshow(s["cam"], cmap="gray", vmin=0, vmax=255)
        ax.set_title("camera-view line mask (detector-like)")
        ax.axis("off")

        ax = axes[r, 2]
        ax.imshow(s["bev"], cmap="gray", vmin=0, vmax=255)
        u, v = g.bev.base_to_px(st, s["y_pts"])
        vis = s["vis"].astype(bool)
        ax.plot(u[vis], v[vis], "o", color="lime", ms=5)
        ax.plot(u[~vis], v[~vis], "x", color="red", ms=6)
        ax.set_title(f"BEV  e_y={s['e_y']*100:+.1f} cm  e_psi={np.rad2deg(s['e_psi']):+.1f} deg\n"
                     f"kappa(0)={s['kappa'][0]:+.2f} 1/m")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=70)
    print("saved", a.out)


if __name__ == "__main__":
    main()
