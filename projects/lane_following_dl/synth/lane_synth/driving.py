"""Virtual driving: kinematic bicycle + pure pursuit with disturbances."""
from __future__ import annotations

import numpy as np

from .labels import nearest_index
from .track import Path, wrap_angle


def drive(path: Path, rng, cfg: dict, n_frames: int):
    """Yield frame states while driving along `path` (target lane centerline).

    Each state: dict(x, y, yaw, v, delta, accel, lat_acc, idx).
    The driver follows the centerline with a slowly drifting lateral offset and
    steering noise, so the poses cover realistic tracking errors.
    """
    L = cfg["wheelbase"]
    dmax = np.deg2rad(cfg["max_steer_deg"])
    tau = cfg["steer_tau"]
    dt = 0.01
    frame_dt = 1.0 / cfg["fps"]
    v_target = rng.uniform(*cfg["pp_speed"])
    Ld = rng.uniform(*cfg["lookahead"])

    # start pose: somewhere on the path with an initial error
    if path.closed:
        i = int(rng.integers(len(path.xy)))
    else:
        i = int(rng.integers(0, max(1, len(path.xy) // 3)))
    h = path.heading[i]
    ey0 = rng.uniform(-1, 1) * cfg["init_ey"]
    x, y = path.xy[i] + ey0 * np.array([-np.sin(h), np.cos(h)])
    yaw = h + np.deg2rad(rng.uniform(-1, 1) * cfg["init_epsi_deg"])
    v, delta, offset, noise = v_target * rng.uniform(0.5, 1.0), 0.0, 0.0, 0.0
    idx = i
    t_next, frames, t = 0.0, 0, 0.0
    a_prev_v = v
    while frames < n_frames:
        idx = nearest_index(path, np.array([x, y]), idx, window=60)
        if not path.closed and idx > len(path.xy) - int(1.2 / 0.005):
            return  # end of an open road
        # slowly varying target offset and steering noise (Ornstein-Uhlenbeck)
        offset += -offset * dt / 1.5 + cfg["offset_sigma"] * np.sqrt(2 * dt / 1.5) * rng.normal()
        noise += -noise * dt / 0.3 + np.deg2rad(cfg["steer_noise_deg"]) * np.sqrt(2 * dt / 0.3) * rng.normal()
        # pure pursuit to a point Ld ahead on the (offset) centerline
        j = idx + int(Ld / 0.005)
        j = j % len(path.xy) if path.closed else min(j, len(path.xy) - 1)
        hj = path.heading[j]
        tx, ty = path.xy[j] + offset * np.array([-np.sin(hj), np.cos(hj)])
        alpha = wrap_angle(np.arctan2(ty - y, tx - x) - yaw)
        d_cmd = np.clip(np.arctan2(2 * L * np.sin(alpha), Ld) + noise, -dmax, dmax)
        delta += (d_cmd - delta) * dt / tau
        # speed: slow down in curves a little, with noise
        v_des = v_target / (1 + 0.3 * abs(path.kappa[idx]))
        v += np.clip(v_des - v, -1.5 * dt, 1.5 * dt)
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw = wrap_angle(yaw + v / L * np.tan(delta) * dt)
        t += dt
        if t >= t_next:
            t_next += frame_dt
            accel = (v - a_prev_v) / frame_dt
            a_prev_v = v
            if rng.random() < cfg["keep_prob"]:
                frames += 1
                yield dict(x=x, y=y, yaw=yaw, v=v, delta=delta, accel=accel,
                           lat_acc=v * v / L * np.tan(delta), idx=idx)


def follow_exact(path: Path, cfg: dict):
    """Drive exactly along the target path (no controller, no noise).

    The vehicle (base_link = rear axle center) moves along the lane centerline
    shifted by a constant lateral offset `offset_y` [m] with a constant heading
    offset `heading_offset_deg`, at constant speed. Frames every 1/fps s.
    Yields dict(x, y, yaw, v, delta, accel, lat_acc, idx, s, t).
    """
    v = float(cfg["speed"])
    fps = float(cfg["fps"])
    off = float(cfg.get("offset_y", 0.0))
    dpsi = np.deg2rad(float(cfg.get("heading_offset_deg", 0.0)))
    L = float(cfg.get("wheelbase", 0.145))
    s0 = float(cfg.get("start_s", 0.0))
    dist = cfg.get("distance")
    total = path.length if dist is None else float(dist)
    if not path.closed:
        total = min(total, path.length - s0 - 1.2)   # keep the camera view on the road
    n = int(total / (v / fps)) + 1
    for k in range(n):
        s = s0 + k * v / fps
        p, h, k_c, i = path.pose_at_s(s)
        x, y = p + off * np.array([-np.sin(h), np.cos(h)])
        kap = k_c / (1.0 - k_c * off)   # curvature of the driven (offset) curve
        yield dict(x=float(x), y=float(y), yaw=float(wrap_angle(h + dpsi)), v=v,
                   delta=float(np.arctan(L * kap)), accel=0.0, lat_acc=v * v * kap,
                   idx=i, s=float(s), t=k / fps)
