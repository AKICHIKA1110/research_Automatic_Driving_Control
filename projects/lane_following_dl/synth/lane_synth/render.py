"""Rendering: course raster → camera-view line mask → BEV mask."""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera, rpy_to_R
from .track import Course, Path


class WorldRaster:
    """Top-down raster of the white lines (value = detection probability 0..255)."""

    def __init__(self, course: Course, res: float, rng=None, track_cfg=None, margin=1.0):
        self.res = res
        offs = course.boundary_offsets()
        xy = np.concatenate([course.center.offset(o).xy for o in offs])
        self.x0, self.y0 = xy.min(0) - margin
        x1, y1 = xy.max(0) + margin
        self.w = int(np.ceil((x1 - self.x0) / res))
        self.h = int(np.ceil((y1 - self.y0) / res))
        self.img = np.zeros((self.h, self.w), np.uint8)
        if rng is not None and track_cfg is not None and rng.random() < track_cfg["floor_seams_prob"]:
            self._draw_seams(rng, track_cfg)
        thick = max(1, int(round(course.line_width / res)))
        for o in offs:
            self._polyline(course.center.offset(o), thick)
        if course.dashed_center:
            self._dashed(course.center, thick, course.dash_length, course.dash_gap,
                         phase=0.0 if rng is None else float(rng.uniform(0, course.dash_length)))

    def to_px(self, xy):
        return np.stack([(xy[:, 0] - self.x0) / self.res - 0.5, (xy[:, 1] - self.y0) / self.res - 0.5], 1)

    def _draw(self, xy, thick, value=255, closed=False):
        pts = np.round(self.to_px(xy) * 16).astype(np.int32)
        cv2.polylines(self.img, [pts], closed, int(value), thick, cv2.LINE_AA, shift=4)

    def _polyline(self, p: Path, thick):
        self._draw(p.xy, thick, closed=p.closed)

    def _dashed(self, p: Path, thick, dash, gap, phase):
        period = dash + gap
        on = ((p.s + phase) % period) < dash
        idx = np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1
        for seg in np.split(np.arange(len(p.s)), idx):
            if len(seg) > 1 and on[seg[0]]:
                self._draw(p.xy[seg], thick)

    def _draw_seams(self, rng, cfg):
        """Floor board seams: faint parallel lines (a typical false-positive source)."""
        spacing = rng.uniform(*cfg["floor_seam_spacing"])
        val = 255 * rng.uniform(*cfg["floor_seam_strength"])
        ang = rng.uniform(0, np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        n = np.array([-d[1], d[0]])
        c = np.array([self.x0 + self.w * self.res / 2, self.y0 + self.h * self.res / 2])
        half = max(self.w, self.h) * self.res
        for k in np.arange(-half, half, spacing):
            p0 = c + n * k - d * half
            p1 = c + n * k + d * half
            self._draw(np.stack([p0, p1]), 1, val)


def base_pose_matrix(x, y, yaw, roll=0.0, pitch=0.0):
    """World ← base_link (rotation, translation). Body roll/pitch about base origin."""
    return rpy_to_R(roll, pitch, yaw), np.array([x, y, 0.0])


class CameraRenderer:
    """Ray-casts each pixel onto the floor and samples the world raster."""

    def __init__(self, cam: Camera, ss: int, max_range: float, rays=None):
        self.cam = cam
        self.ss = ss
        self.max_range = max_range
        self.rays = cam.pixel_rays(ss) if rays is None else rays  # (H*ss, W*ss, 3) optical frame

    def render(self, world: WorldRaster, R_wb, t_wb):
        cam = self.cam
        R_wo = R_wb @ cam.R_base_opt
        C = R_wb @ cam.xyz + t_wb
        d = self.rays @ R_wo.T.astype(np.float32)
        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dz < -1e-6, -C[2] / dz, -1.0)
        gx = C[0] + t * d[..., 0]
        gy = C[1] + t * d[..., 1]
        rng = np.hypot(gx - C[0], gy - C[1])
        valid = (t > 0) & (rng < self.max_range)
        mx = np.where(valid, (gx - world.x0) / world.res - 0.5, -10).astype(np.float32)
        my = np.where(valid, (gy - world.y0) / world.res - 0.5, -10).astype(np.float32)
        img = cv2.remap(world.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        size = (cam.width, cam.height)
        prob = cv2.resize(img.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_AREA)
        rng_img = cv2.resize(np.where(valid, rng, 99.0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        return prob, rng_img


class BevWarper:
    """Camera image → bird's-eye view using the *nominal* calibration (as on the robot)."""

    def __init__(self, cam: Camera, bcfg: dict):
        res = bcfg["res"]
        self.x = np.arange(bcfg["x_max"] - res / 2, bcfg["x_min"], -res)   # rows: far → near
        self.y = np.arange(bcfg["y_max"] - res / 2, bcfg["y_min"], -res)   # cols: left → right
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        uv, front = cam.project_base(P)
        ok = cam.in_image(uv, front)
        self.mapx = np.where(ok, uv[:, 0], -10).reshape(X.shape).astype(np.float32)
        self.mapy = np.where(ok, uv[:, 1], -10).reshape(X.shape).astype(np.float32)
        self.valid = ok.reshape(X.shape)
        self.res = res
        self.bcfg = bcfg

    @property
    def shape(self):
        return self.mapx.shape

    def warp(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def base_to_px(self, x, y):
        """base_link (x, y) → BEV pixel (col, row)."""
        return (self.bcfg["y_max"] - y) / self.res - 0.5, (self.bcfg["x_max"] - x) / self.res - 0.5


def _ellipse_mask(shape, rng, size_range):
    h, w = shape
    m = np.zeros(shape, np.float32)
    c = (int(rng.uniform(0, w)), int(rng.uniform(h * 0.3, h)))
    ax = (int(rng.uniform(*size_range)), int(rng.uniform(*size_range) * rng.uniform(0.3, 1.0)))
    cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), max(ax) * 0.25 + 1)


def detection_noise(prob, rng_img, rng, cfg):
    """Make an ideal line mask look like a line-detection CNN output (probability map).

    With `enable: false` the ideal mask is returned unchanged (as uint8).
    """
    if not cfg.get("enable", True):
        return (np.clip(prob, 0, 1) * 255 + 0.5).astype(np.uint8)
    p = prob.copy()
    h, w = p.shape
    if rng.random() < cfg["morph_prob"]:
        k = np.ones((int(rng.integers(2, 4)),) * 2, np.uint8)
        p = cv2.dilate(p, k) if rng.random() < 0.5 else cv2.erode(p, k)
    # weaker detection far away
    r0, r1 = cfg["range_fade"]
    fade = np.clip(1.0 - (rng_img - r0) / (r1 - r0) * rng.uniform(0.3, 1.0), 0.0, 1.0)
    p *= fade
    # missing parts (shadows, occlusion, worn tape)
    for _ in range(int(rng.integers(cfg["dropout_blobs"][0], cfg["dropout_blobs"][1] + 1))):
        p *= 1.0 - _ellipse_mask((h, w), rng, cfg["dropout_size"])
    # false positives: specular reflections of the ceiling lights
    for _ in range(int(rng.integers(cfg["specular_blobs"][0], cfg["specular_blobs"][1] + 1))):
        p = np.maximum(p, _ellipse_mask((h, w), rng, cfg["specular_size"]) * rng.uniform(*cfg["specular_strength"]))
    n = int(rng.integers(cfg["speckle_count"][0], cfg["speckle_count"][1] + 1))
    if n:
        ys, xs = rng.integers(h // 4, h, n), rng.integers(0, w, n)
        sp = np.zeros_like(p)
        sp[ys, xs] = rng.uniform(0.3, 1.0, n)
        p = np.maximum(p, cv2.dilate(sp, np.ones((3, 3), np.uint8)))
    s = rng.uniform(*cfg["blur_sigma"])
    if s > 0.2:
        p = cv2.GaussianBlur(p, (0, 0), s)
    p = p * rng.uniform(*cfg["gain"]) + rng.normal(0, cfg["gauss_noise"], p.shape)
    p = np.clip(p, 0, 1)
    if rng.random() < cfg["binarize_prob"]:
        p = (p > rng.uniform(0.3, 0.6)).astype(np.float32)
    return (p * 255 + 0.5).astype(np.uint8)
