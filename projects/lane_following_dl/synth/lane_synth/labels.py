"""Ground-truth labels from the true vehicle pose and the target lane centerline."""
from __future__ import annotations

import numpy as np

from .camera import Camera
from .track import Path, wrap_angle


def nearest_index(path: Path, xy, hint: int | None = None, window: int = 400) -> int:
    if hint is None:
        return int(np.argmin(np.sum((path.xy - xy) ** 2, 1)))
    n = len(path.xy)
    idx = (hint + np.arange(-window, window)) % n if path.closed else \
        np.arange(max(0, hint - window), min(n, hint + window))
    return int(idx[np.argmin(np.sum((path.xy[idx] - xy) ** 2, 1))])


def _ahead(path: Path, i0: int, length: float):
    """Indices from i0 forward along the path covering `length` metres."""
    n = int(length / 0.005) + 2
    idx = i0 + np.arange(n)
    if path.closed:
        return idx % len(path.xy)
    return idx[idx < len(path.xy)]


def compute_labels(path: Path, pose, idx_hint, cfg_labels, bev_cfg, cam_true: Camera, R_wb, t_wb,
                   half_width: float = 0.2):
    """Labels for one frame.

    pose: true (x, y, yaw) of base_link in world.
    Returns dict with
      y_pts[k]  lateral position [m] of the lane centerline at x = stations_x[k] (base_link)
      vis[k]    1 if that point exists ahead, lies in the BEV ROI and is in the camera image
      e_y       lateral offset of base_link from the centerline (vehicle left of path → +)
      e_psi     heading error (vehicle yaw − path heading)
      kappa[m]  path curvature at arc length preview_s[m] from the nearest point
      path_xy[j]   point of the centerline at arc length path_s[j] ahead (base_link x, y) [m]
      path_valid[j] 1 if the point is in the BEV area and the point or one of the two
                    boundary lines next to it (±half_width) is seen by the camera,
                    i.e. the point can be inferred from the image
    """
    x, y, yaw = pose
    i0 = nearest_index(path, np.array([x, y]), idx_hint)
    # project onto the polyline segment next to the nearest sample (sub-sample accuracy)
    n = len(path.xy)
    best = None
    for j in (i0 - 1, i0):
        if not path.closed and (j < 0 or j + 1 >= n):
            continue
        a_, b_ = path.xy[j % n], path.xy[(j + 1) % n]
        seg = b_ - a_
        u = float(np.clip(np.dot(np.array([x, y]) - a_, seg) / np.dot(seg, seg), 0.0, 1.0))
        q = a_ + u * seg
        dist = float(np.sum((np.array([x, y]) - q) ** 2))
        if best is None or dist < best[0]:
            h0, h1 = path.heading[j % n], path.heading[(j + 1) % n]
            best = (dist, q, wrap_angle(h0 + u * wrap_angle(h1 - h0)))
    _, q, h = best
    d = np.array([x, y]) - q
    e_y = float(-np.sin(h) * d[0] + np.cos(h) * d[1])
    e_psi = float(wrap_angle(yaw - h))

    ds = 0.005
    kap = []
    for s in cfg_labels["preview_s"]:
        j = i0 + int(round(s / ds))
        j = j % len(path.xy) if path.closed else min(j, len(path.xy) - 1)
        kap.append(path.kappa[j])

    st = np.asarray(cfg_labels["stations_x"], float)
    idx = _ahead(path, i0, st.max() * 3.0)
    P = path.xy[idx] - np.array([x, y])
    c, s_ = np.cos(yaw), np.sin(yaw)
    bx = c * P[:, 0] + s_ * P[:, 1]
    by = -s_ * P[:, 0] + c * P[:, 1]
    y_pts = np.zeros(len(st), np.float32)
    exists = np.zeros(len(st), bool)
    for k, xs in enumerate(st):
        # first forward crossing of x = xs
        cross = np.flatnonzero((bx[:-1] <= xs) & (bx[1:] > xs))
        if len(cross):
            j = cross[0]
            a = (xs - bx[j]) / (bx[j + 1] - bx[j])
            y_pts[k] = by[j] + a * (by[j + 1] - by[j])
            exists[k] = True
    in_roi = exists & (y_pts >= bev_cfg["y_min"]) & (y_pts <= bev_cfg["y_max"])
    # is the point actually seen by the (true) camera? (point is on the floor, z = 0)
    # base_link under body roll/pitch: transform floor point to the tilted base frame
    Pworld = np.stack([x + c * st - s_ * y_pts, y + s_ * st + c * y_pts, np.zeros_like(st)], 1)
    Pb = (Pworld - t_wb) @ R_wb
    uv, front = cam_true.project_base(Pb)
    vis = in_roi & cam_true.in_image(uv, front)
    # points along the centerline by arc length (exist even in tight curves)
    ps = np.asarray(cfg_labels.get("path_s", []), float)
    path_xy = np.zeros((len(ps), 2), np.float32)
    path_valid = np.zeros(len(ps), np.uint8)
    for k, sk in enumerate(ps):
        j = i0 + int(round(sk / ds))
        if path.closed:
            j %= len(path.xy)
        elif j >= len(path.xy):
            continue
        hj = path.heading[j]
        nrm = np.array([-np.sin(hj), np.cos(hj)])
        cand = np.stack([path.xy[j], path.xy[j] + half_width * nrm, path.xy[j] - half_width * nrm])
        rel = cand - np.array([x, y])
        cb = np.stack([c * rel[:, 0] + s_ * rel[:, 1], -s_ * rel[:, 0] + c * rel[:, 1]], 1)
        path_xy[k] = cb[0]
        in_bev = ((cb[:, 0] >= bev_cfg["x_min"]) & (cb[:, 0] <= bev_cfg["x_max"])
                  & (cb[:, 1] >= bev_cfg["y_min"]) & (cb[:, 1] <= bev_cfg["y_max"]))
        Pw = np.concatenate([cand, np.zeros((3, 1))], 1)
        uv_c, fr_c = cam_true.project_base((Pw - t_wb) @ R_wb)
        seen = cam_true.in_image(uv_c, fr_c) & in_bev
        path_valid[k] = bool(in_bev[0] and seen.any())
    return dict(y_pts=y_pts, vis=vis.astype(np.uint8), e_y=e_y, e_psi=e_psi,
                kappa=np.asarray(kap, np.float32), path_xy=path_xy, path_valid=path_valid, idx=i0)
