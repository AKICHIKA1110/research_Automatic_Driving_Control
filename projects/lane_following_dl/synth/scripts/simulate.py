#!/usr/bin/env python3
"""Drive the set trajectory on a course and save the images before / after the BEV warp.

  python3 scripts/simulate.py --course config/courses/oval.yaml --out runs/oval
  python3 scripts/simulate.py --course config/courses/oval.yaml --out runs/oval_off5 \
      --offset-y 0.05 --heading-offset 5 --speed 0.3

Output:
  camera/000000.png       camera-view line mask (before the BEV warp), 640x480
  bev/000000.png          BEV mask (after the warp), 180x140
  bev_overlay/000000.png  BEV (enlarged) with the label points and the true centerline
  labels.csv              per-frame pose and labels
  course.png              course, trajectory and camera footprint
  config.yaml, course.yaml  what was used
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lane_synth import Generator, load_config  # noqa: E402
from lane_synth.track import course_from_dict, random_course  # noqa: E402


def overlay_bev(bev_img, g, s, path, scale=4):
    """Enlarged BEV with the true lane centerline (cyan) and the label points."""
    img = cv2.cvtColor(cv2.resize(bev_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                       cv2.COLOR_GRAY2BGR)
    x, y, yaw = s["pose"]
    c, sn = np.cos(yaw), np.sin(yaw)
    P = path.xy - np.array([x, y])
    bx, by = c * P[:, 0] + sn * P[:, 1], -sn * P[:, 0] + c * P[:, 1]
    u, v = g.bev.base_to_px(bx, by)
    ok = (u > -1) & (u < bev_img.shape[1]) & (v > -1) & (v < bev_img.shape[0])
    pts = np.stack([(u[ok] + 0.5) * scale, (v[ok] + 0.5) * scale], 1).astype(np.int32)
    if len(pts) > 1:
        # draw only consecutive runs (the path may enter/leave the view several times)
        brk = np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1
        for run in np.split(pts, brk):
            cv2.polylines(img, [run], False, (255, 255, 0), 1, cv2.LINE_AA)
    st = np.asarray(g.cfg["labels"]["stations_x"])
    u, v = g.bev.base_to_px(st, s["y_pts"])
    for k in range(len(st)):
        p = (int((u[k] + 0.5) * scale), int((v[k] + 0.5) * scale))
        if s["vis"][k]:
            cv2.circle(img, p, 5, (0, 255, 0), -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    txt = [f"e_y {s['e_y']*100:+.1f} cm  e_psi {np.rad2deg(s['e_psi']):+.1f} deg",
           f"kappa(0) {s['kappa'][0]:+.2f} 1/m  (R {abs(1/s['kappa'][0]) if abs(s['kappa'][0]) > 1e-6 else np.inf:.2f} m)"]
    for i, tline in enumerate(txt):
        cv2.putText(img, tline, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return img


def plot_course(course, path, poses, cfg, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    for o in course.boundary_offsets():
        p = course.center.offset(o)
        xy = np.vstack([p.xy, p.xy[:1]]) if p.closed else p.xy
        ax.plot(xy[:, 0], xy[:, 1], "k-", lw=1.5)
    if course.dashed_center:
        ax.plot(course.center.xy[:, 0], course.center.xy[:, 1], "k--", lw=0.8)
    ax.plot(path.xy[:, 0], path.xy[:, 1], "c-", lw=0.8, label="target path (lane center)")
    P = np.array(poses)
    ax.plot(P[:, 0], P[:, 1], "r.", ms=2, label="vehicle (rear axle) per frame")
    b = cfg["bev"]
    roi = np.array([[b["x_min"], b["y_min"]], [b["x_max"], b["y_min"]], [b["x_max"], b["y_max"]],
                    [b["x_min"], b["y_max"]], [b["x_min"], b["y_min"]]])
    for x, y, yaw in P[:: max(1, len(P) // 6)]:
        c, s = np.cos(yaw), np.sin(yaw)
        ax.plot(x + c * roi[:, 0] - s * roi[:, 1], y + s * roi[:, 0] + c * roi[:, 1], "m-", lw=0.8)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("course (black), target path (cyan), BEV area (magenta)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--course", default=None, help="course YAML; omit for a random course")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--offset-y", type=float, default=None, help="lateral offset from the lane center [m]")
    ap.add_argument("--heading-offset", type=float, default=None, help="heading offset [deg]")
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--reverse", action="store_true", help="drive the course in the opposite direction")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    g = Generator(cfg, a.seed)
    if a.course:
        course_d = yaml.safe_load(open(a.course))
        course = course_from_dict(course_d, cfg["track"])
    else:
        course = random_course(g.rng, cfg["track"])
        course_d = None
    over = {k: v for k, v in dict(speed=a.speed, fps=a.fps, offset_y=a.offset_y,
                                  heading_offset_deg=a.heading_offset, start_s=a.start_s,
                                  distance=a.distance).items() if v is not None}

    out = Path(a.out)
    for d in ["camera", "bev"] + ([] if a.no_overlay else ["bev_overlay"]):
        (out / d).mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(dict(cfg, driving=dict(cfg["driving"], **over)), f, allow_unicode=True, sort_keys=False)
    if a.course:
        shutil.copy(a.course, out / "course.yaml")

    path = course.lane_path(0, not a.reverse)
    K = len(cfg["labels"]["stations_x"])
    M = len(cfg["labels"]["preview_s"])
    header = (["frame", "t", "s", "x", "y", "yaw", "v", "e_y", "e_psi"]
              + [f"y_pt{k}" for k in range(K)] + [f"vis{k}" for k in range(K)] + [f"kappa{m}" for m in range(M)])
    poses = []
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, s in enumerate(g.run(course, 0, not a.reverse, over)):
            cv2.imwrite(str(out / "camera" / f"{i:06d}.png"), s["cam"])
            cv2.imwrite(str(out / "bev" / f"{i:06d}.png"), s["bev"])
            if not a.no_overlay:
                cv2.imwrite(str(out / "bev_overlay" / f"{i:06d}.png"), overlay_bev(s["bev"], g, s, path))
            x, y, yaw = s["pose"]
            poses.append((x, y, yaw))
            w.writerow([i, f"{s['t']:.4f}", f"{s['s']:.4f}", f"{x:.5f}", f"{y:.5f}", f"{yaw:.6f}", f"{s['v']:.3f}",
                        f"{s['e_y']:.5f}", f"{s['e_psi']:.6f}"] + [f"{v:.5f}" for v in s["y_pts"]]
                       + [int(v) for v in s["vis"]] + [f"{v:.5f}" for v in s["kappa"]])
    plot_course(course, path, poses, cfg, out / "course.png")
    print(f"{len(poses)} frames → {out}")


if __name__ == "__main__":
    main()
