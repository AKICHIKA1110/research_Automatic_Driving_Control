#!/usr/bin/env python3
"""Intrinsic calibration of the HP60C RGB camera from checkerboard images.

  python3 scripts/calibrate_camera.py IMG_DIR --cols 9 --rows 6 --square 0.025 --out calib/hp60c_rgb.yaml

  --cols / --rows : number of INNER corners (squares - 1) along the long / short side
  --square        : side length of one square [m] (measure the printed board)

Writes a ROS camera_info YAML (readable by the GUI's camera tab) and, next to it,
<out>_report/ with corner detections, undistorted samples and per-image errors.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml


def find_corners(paths, size):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, img_size = [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = g.shape[::-1]
        # dark images: stretch contrast before detection
        g2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
        ok, c = cv2.findChessboardCornersSB(g2, size, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if not ok:
            ok, c = cv2.findChessboardCorners(g2, size, flags)
            if ok:
                c = cv2.cornerSubPix(g2, c, (5, 5), (-1, -1), crit)
        found.append((p, img, c.reshape(-1, 1, 2).astype(np.float32) if ok else None))
    return found, img_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir")
    ap.add_argument("--cols", type=int, required=True, help="inner corners along the long side")
    ap.add_argument("--rows", type=int, required=True, help="inner corners along the short side")
    ap.add_argument("--square", type=float, required=True, help="square size [m]")
    ap.add_argument("--out", default="calib/hp60c_rgb.yaml")
    ap.add_argument("--k3", action="store_true", help="also estimate k3 (default: fixed to 0)")
    ap.add_argument("--max-err", type=float, default=1.0,
                    help="drop images whose RMS reprojection error exceeds this [px] and recalibrate")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.img_dir, "*.png")) + glob.glob(os.path.join(a.img_dir, "*.jpg")))
    if not paths:
        sys.exit(f"no images in {a.img_dir}")
    size = (a.cols, a.rows)
    found, img_size = find_corners(paths, size)
    used = [(p, img, c) for p, img, c in found if c is not None]
    print(f"checkerboard found in {len(used)}/{len(found)} images")
    if len(used) < 8:
        sys.exit("need at least ~8 good images (15-30 recommended)")

    obj = np.zeros((a.cols * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    flags = 0 if a.k3 else cv2.CALIB_FIX_K3

    def calib(sel):
        rms, K, D, rv, tv = cv2.calibrateCamera([obj] * len(sel), [c for _, _, c in sel], img_size, None, None,
                                                flags=flags)
        errs = []
        for (_, _, c), r, t in zip(sel, rv, tv):
            pr, _ = cv2.projectPoints(obj, r, t, K, D)
            errs.append(float(np.sqrt(np.mean(np.sum((pr - c) ** 2, axis=2)))))
        return rms, K, D, rv, tv, errs

    rms, K, D, rv, tv, errs = calib(used)
    bad = [i for i, e in enumerate(errs) if e > a.max_err]
    if bad and len(used) - len(bad) >= 8:
        print(f"dropping {len(bad)} images with error > {a.max_err} px: "
              + ", ".join(os.path.basename(used[i][0]) for i in bad))
        used = [u for i, u in enumerate(used) if i not in bad]
        rms, K, D, rv, tv, errs = calib(used)

    # coverage of the image by detected corners (3x3 grid)
    w, h = img_size
    cov = np.zeros((3, 3), int)
    for _, _, c in used:
        for x, y in c.reshape(-1, 2):
            cov[min(2, int(3 * y / h)), min(2, int(3 * x / w))] += 1
    dist_m = [float(np.linalg.norm(t)) for t in tv]

    D5 = np.zeros(5)
    D5[:min(5, D.size)] = D.ravel()[:5]
    P = np.hstack([K, np.zeros((3, 1))])
    out = dict(
        image_width=w, image_height=h, camera_name="hp60c_rgb",
        camera_matrix=dict(rows=3, cols=3, data=[float(v) for v in K.ravel()]),
        distortion_model="plumb_bob",
        distortion_coefficients=dict(rows=1, cols=5, data=[float(v) for v in D5]),
        rectification_matrix=dict(rows=3, cols=3, data=[float(v) for v in np.eye(3).ravel()]),
        projection_matrix=dict(rows=3, cols=4, data=[float(v) for v in P.ravel()]),
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# HP60C RGB intrinsics: {len(used)} images, board {a.cols}x{a.rows} inner corners, "
                f"square {a.square} m, RMS reprojection error {rms:.3f} px\n")
        yaml.safe_dump(out, f, sort_keys=False)

    # report images
    rep = os.path.splitext(a.out)[0] + "_report"
    os.makedirs(rep, exist_ok=True)
    for (p, img, c), e in zip(used, errs):
        v = img.copy()
        cv2.drawChessboardCorners(v, size, c, True)
        cv2.putText(v, f"{e:.2f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(rep, "corners_" + os.path.basename(p)), v)
    p0, img0, _ = used[0]
    und = cv2.undistort(img0, K, D)
    cv2.imwrite(os.path.join(rep, "undistort_compare.png"), np.hstack([img0, und]))
    with open(os.path.join(rep, "per_image_error.csv"), "w") as f:
        f.write("file,rms_px,distance_m\n")
        for (p, _, _), e, d in zip(used, errs, dist_m):
            f.write(f"{os.path.basename(p)},{e:.4f},{d:.3f}\n")

    np.set_printoptions(precision=4, suppress=True)
    print(f"\nRMS reprojection error: {rms:.3f} px  (images used: {len(used)})")
    print(f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}")
    print(f"dist (k1 k2 p1 p2 k3) = {D5}")
    print(f"per-image error: min {min(errs):.3f}  median {np.median(errs):.3f}  max {max(errs):.3f} px")
    print(f"board distance: {min(dist_m):.2f} - {max(dist_m):.2f} m")
    print("corner coverage (3x3, rows = top..bottom):")
    print(cov)
    print(f"\nwrote {a.out}\nreport: {rep}/")


if __name__ == "__main__":
    main()
