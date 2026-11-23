"""Variable-speed runs on lane-synth courses, sampled at a fixed frame rate.

One run = one random course, one lane and direction, a random speed profile
(accelerations, decelerations, near stops). Every frame is kept (no random
dropping), so consecutive frames are exactly 1/fps apart.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

SYNTH = Path(__file__).resolve().parents[2] / "lane_following_dl" / "synth"
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

from lane_synth.generator import Generator, load_config  # noqa: E402
from lane_synth.render import CameraRenderer, WorldRaster  # noqa: E402
from lane_synth.track import random_course, wrap_angle  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_project_config(path=None) -> dict:
    with open(path or PROJECT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["synth_full"] = deep_update(load_config(), cfg.get("synth", {}))
    return cfg


def speed_profile(rng, n_steps: int, dt: float, c: dict) -> tuple[np.ndarray, float]:
    """Speed [m/s] every dt: piecewise targets reached with a per-run acceleration limit."""
    a_max = rng.uniform(*c["accel_max"])
    v = np.empty(n_steps)
    cur = rng.uniform(0.0, c["speed_max"])
    target, t_left = cur, 0.0
    for i in range(n_steps):
        if t_left <= 0:
            if rng.random() < c["stop_prob"]:
                target = rng.uniform(*c["stop_speed"])
            else:
                target = rng.uniform(0.1, c["speed_max"])
            t_left = rng.uniform(*c["hold"])
        cur += np.clip(target - cur, -a_max * dt, a_max * dt)
        v[i] = cur
        t_left -= dt
    return v, a_max


def ou(rng, n: int, dt: float, sigma: float, tau: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = x[i - 1] - x[i - 1] * dt / tau + sigma * np.sqrt(2 * dt / tau) * rng.normal()
    return x


def generate_run(run_id: int, cfg: dict, seed: int):
    """Render one run. Returns (frames dict of arrays, meta dict)."""
    sc, c = cfg["synth_full"], cfg["sequence"]
    rng = np.random.default_rng(seed)
    cam_size = c.get("camera_size")          # [W, H]: also save the camera-view mask (before the BEV warp)
    gen = Generator(sc, seed=int(rng.integers(2**31)), keep_camera=bool(cam_size))

    course = random_course(rng, sc["track"])
    seams = bool(rng.random() < c["seams_prob"])
    track_cfg = dict(sc["track"], floor_seams_prob=1.0 if seams else 0.0)
    world = WorldRaster(course, sc["render"]["world_res"], rng, track_cfg)
    lane = int(rng.integers(len(course.lane_center_offsets())))
    forward = bool(rng.random() < 0.5)
    path = course.lane_path(lane, forward)
    cam_true = gen._true_camera()
    renderer = CameraRenderer(cam_true, sc["render"]["supersample"], sc["render"]["max_range"])

    fps = float(c["fps"])
    dt = 0.01
    sub = int(round(1.0 / fps / dt))
    duration = rng.uniform(*c["duration"])
    if not path.closed:   # an open road ends: keep the camera on the road
        duration = min(duration, max(2.0, (path.length - 1.5) / max(0.3, c["speed_max"] * 0.6)))
    n_frames = int(duration * fps)
    n_steps = n_frames * sub
    v, a_max = speed_profile(rng, n_steps, dt, c)
    s = np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])
    s0 = rng.uniform(0, path.length) if path.closed else 0.0
    if not path.closed and s[-1] > path.length - 1.2:   # too fast for this road: stop at its end
        keep = s <= path.length - 1.2
        n_frames = int(np.count_nonzero(keep[::sub]))
    off = ou(rng, n_steps, dt, c["offset_sigma"], 2.0)
    dpsi = ou(rng, n_steps, dt, np.deg2rad(c["heading_sigma_deg"]), 1.0)

    L = sc["driving"]["wheelbase"]
    keys = ("v", "accel", "kappa", "s", "t", "e_y")
    out = {k: [] for k in keys}
    bevs, cams = [], []
    for k in range(n_frames):
        i = k * sub
        p, h, kap_c, idx = path.pose_at_s(s0 + s[i])
        x, y = p + off[i] * np.array([-np.sin(h), np.cos(h)])
        accel = (v[i] - v[i - sub]) / (sub * dt) if i >= sub else 0.0
        kap = kap_c / (1.0 - kap_c * off[i])
        st = dict(x=float(x), y=float(y), yaw=float(wrap_angle(h + dpsi[i])), v=float(v[i]),
                  delta=float(np.arctan(L * kap)), accel=float(accel), lat_acc=float(v[i] ** 2 * kap),
                  idx=idx, s=float(s[i]), t=k / fps)
        smp = gen._sample(course, world, path, cam_true, renderer, st, run_id)
        bevs.append(smp["bev"])
        if cam_size:   # keep_camera only adds outputs: the random draws (and so the run) are unchanged
            cams.append(cv2.resize(smp["cam"], tuple(cam_size), interpolation=cv2.INTER_AREA))
        out["v"].append(st["v"]); out["accel"].append(st["accel"]); out["kappa"].append(float(kap_c))
        out["s"].append(st["s"]); out["t"].append(st["t"]); out["e_y"].append(float(smp["e_y"]))
    frames = {k: np.asarray(vv, np.float32) for k, vv in out.items()}
    frames["bev"] = np.stack(bevs).astype(np.uint8)
    if cam_size:
        frames["cam"] = np.stack(cams).astype(np.uint8)
    meta = dict(run=run_id, seed=seed, n_frames=n_frames, closed=bool(path.closed), dashed=bool(course.dashed_center),
                dash_period=float(course.dash_length + course.dash_gap) if course.dashed_center else float("nan"),
                seams=seams, lane_width=float(course.lane_width), accel_max=float(a_max),
                path_length=float(path.length))
    return frames, meta
