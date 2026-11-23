"""Course (track) generation.

A course is described by the centerline of the road as a dense polyline
(`Path`). Lane boundaries, the center dashed line and the lane centerlines
are obtained by offsetting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

DS = 0.005  # sampling step of all paths [m]


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Path:
    xy: np.ndarray        # (N, 2)
    heading: np.ndarray   # (N,)
    kappa: np.ndarray     # (N,) signed curvature, left turn positive
    closed: bool
    s: np.ndarray = field(init=False)

    def __post_init__(self):
        d = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])

    @property
    def length(self) -> float:
        if self.closed:
            return float(self.s[-1] + np.linalg.norm(self.xy[0] - self.xy[-1]))
        return float(self.s[-1])

    def offset(self, d: float) -> "Path":
        """Parallel path at lateral offset d (left positive)."""
        n = np.stack([-np.sin(self.heading), np.cos(self.heading)], axis=1)
        denom = 1.0 - self.kappa * d
        return Path(self.xy + d * n, self.heading.copy(), self.kappa / denom, self.closed)

    def reversed(self) -> "Path":
        return Path(self.xy[::-1].copy(), wrap_angle(self.heading[::-1] + np.pi),
                    -self.kappa[::-1].copy(), self.closed)

    def pose_at_s(self, s: float):
        """Interpolated (xy, heading, kappa, nearest index) at arc length s."""
        if self.closed:
            s = s % self.length
            ss = np.append(self.s, self.length)
            xy = np.vstack([self.xy, self.xy[:1]])
            hh = np.unwrap(np.append(self.heading, self.heading[0]))
        else:
            s = float(np.clip(s, 0.0, self.s[-1]))
            ss, xy, hh = self.s, self.xy, np.unwrap(self.heading)
        j = int(np.clip(np.searchsorted(ss, s) - 1, 0, len(ss) - 2))
        a = (s - ss[j]) / (ss[j + 1] - ss[j])
        p = xy[j] + a * (xy[j + 1] - xy[j])
        h = wrap_angle(hh[j] + a * (hh[j + 1] - hh[j]))
        i = (j + int(a >= 0.5)) % len(self.xy)
        return p, float(h), float(self.kappa[i]), i

    def index_at_s(self, s: float) -> int:
        if self.closed:
            s = s % self.length
        return int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))


@dataclass
class Course:
    center: Path            # road centerline
    lane_width: float
    line_width: float
    dashed_center: bool     # True: 2 lanes with a center dashed line
    dash_length: float
    dash_gap: float

    def boundary_offsets(self):
        """Lateral offsets of the solid boundary lines."""
        if self.dashed_center:
            return [self.lane_width, -self.lane_width]
        return [self.lane_width / 2, -self.lane_width / 2]

    def lane_center_offsets(self):
        if self.dashed_center:
            return [self.lane_width / 2, -self.lane_width / 2]
        return [0.0]

    def lane_path(self, lane_idx: int, forward: bool) -> Path:
        """Centerline of a lane, oriented in the driving direction.

        With right-hand traffic the vehicle drives in the right lane of its
        direction, but both lanes are used so that the data covers both.
        """
        p = self.center.offset(self.lane_center_offsets()[lane_idx])
        return p if forward else p.reversed()


# ---------------------------------------------------------------------------
# path construction helpers


def _arc_points(p0, h0, kappa, length):
    n = max(int(np.ceil(length / DS)), 1)
    s = np.arange(1, n + 1) * (length / n)
    if abs(kappa) < 1e-9:
        xy = p0 + np.outer(s, [np.cos(h0), np.sin(h0)])
        h = np.full(n, h0)
    else:
        h = h0 + kappa * s
        xy = p0 + np.stack([(np.sin(h) - np.sin(h0)) / kappa,
                            -(np.cos(h) - np.cos(h0)) / kappa], axis=1)
    return xy, h, np.full(n, kappa)


def integrate_segments(segments, p0=(0.0, 0.0), h0=0.0) -> Path:
    """Build an open path from segments (length, kappa_start, kappa_end).

    Curvature varies linearly in a segment (clothoid); constant = arc.
    """
    xy = [np.asarray(p0, float)]
    hs = [h0]
    ks = [segments[0][1]]
    p, h = np.asarray(p0, float), h0
    for length, k0, k1 in segments:
        n = max(int(np.ceil(length / DS)), 1)
        ds = length / n
        for i in range(n):
            km = k0 + (k1 - k0) * (i + 0.5) / n
            hm = h + 0.5 * km * ds
            p = p + ds * np.array([np.cos(hm), np.sin(hm)])
            h = h + km * ds
            xy.append(p)
            hs.append(h)
            ks.append(k0 + (k1 - k0) * (i + 1) / n)
    return Path(np.array(xy), wrap_angle(np.array(hs)), np.array(ks), closed=False)


def path_from_segments(segments, closed: bool, p0=(0.0, 0.0), h0_deg=0.0) -> Path:
    """Path from a list of segment dicts (as written in a course YAML).

      {type: straight, length: L}
      {type: arc, radius: R, angle_deg: A}          A > 0: left turn, A < 0: right turn
      {type: clothoid, length: L, k0: K0, k1: K1}   curvature changes linearly K0 → K1
    """
    p = np.asarray(p0, float)
    h = np.deg2rad(h0_deg)
    xy, hs, ks = [p[None]], [[h]], [[0.0]]
    for sg in segments:
        kind = sg["type"]
        if kind == "straight":
            pts, hh, kk = _arc_points(p, h, 0.0, float(sg["length"]))
        elif kind == "arc":
            R, A = float(sg["radius"]), np.deg2rad(float(sg["angle_deg"]))
            pts, hh, kk = _arc_points(p, h, np.sign(A) / R, R * abs(A))
        elif kind == "clothoid":
            q = integrate_segments([(float(sg["length"]), float(sg["k0"]), float(sg["k1"]))], p0=p, h0=h)
            pts, hh, kk = q.xy[1:], q.heading[1:], q.kappa[1:]
        else:
            raise ValueError(f"unknown segment type: {kind}")
        xy.append(pts); hs.append(hh); ks.append(kk)
        p, h = pts[-1], hh[-1]
    xy, hs, ks = np.concatenate(xy), np.concatenate(hs), np.concatenate(ks)
    ks[0] = ks[1]
    if closed:
        gap = np.linalg.norm(xy[-1] - xy[0])
        dh = abs(wrap_angle(hs[-1] - hs[0]))
        if gap > 0.01 or dh > np.deg2rad(1.0):
            raise ValueError(f"course does not close: end gap {gap*100:.1f} cm, heading {np.rad2deg(dh):.1f} deg")
        xy, hs, ks = xy[:-1], hs[:-1], ks[:-1]   # last point == first point
    return Path(xy, wrap_angle(hs), ks, closed)


def course_from_dict(d: dict, track_cfg: dict) -> "Course":
    """Course from a course YAML (see config/courses/)."""
    center = path_from_segments(d["segments"], bool(d.get("closed", True)),
                                d.get("start_xy", (0.0, 0.0)), d.get("start_heading_deg", 0.0))
    lane_w = float(d.get("lane_width", np.mean(track_cfg["lane_width"])))
    line_w = float(d.get("line_width", np.mean(track_cfg["line_width"])))
    dashed = d.get("center_line", track_cfg.get("center_line", "none")) == "dashed"
    dash = float(d.get("dash_length", np.mean(track_cfg["dash_length"])))
    return Course(center, lane_w, line_w, dashed, dash, dash * float(d.get("dash_gap_ratio", 1.0)))


def fillet_polygon(vertices, radii, min_straight=0.05) -> Path | None:
    """Closed path made of straights and arcs by rounding polygon corners."""
    v = np.asarray(vertices, float)
    n = len(v)
    a = v - np.roll(v, 1, axis=0)                  # incoming edge of each vertex
    edge_len = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)  # edge i -> i+1
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.roll(a, -1, axis=0)                     # outgoing edge
    theta = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
    t = radii * np.tan(np.abs(theta) / 2)
    if np.any(t + np.roll(t, -1) + min_straight > edge_len):
        return None
    if abs(abs(theta.sum()) - 2 * np.pi) > 1e-6:
        return None  # not a simple polygon
    xy, hs, ks = [], [], []
    for i in range(n):
        p_in = v[i] - a[i] * t[i]
        h_in = np.arctan2(a[i, 1], a[i, 0])
        k = np.sign(theta[i]) / radii[i]
        axy, ah, ak = _arc_points(p_in, h_in, k, radii[i] * abs(theta[i]))
        xy.append(p_in[None]); hs.append([h_in]); ks.append([k])
        xy.append(axy); hs.append(ah); ks.append(ak)
        p_out = v[i] + b[i] * t[i]
        nxt = (i + 1) % n
        p_next_in = v[nxt] - a[nxt] * t[nxt]
        sxy, sh, sk = _arc_points(p_out, np.arctan2(b[i, 1], b[i, 0]), 0.0,
                                  np.linalg.norm(p_next_in - p_out))
        xy.append(sxy[:-1]); hs.append(sh[:-1]); ks.append(sk[:-1])
    return Path(np.concatenate(xy), wrap_angle(np.concatenate(hs)),
                np.concatenate(ks), closed=True)


def self_clearance_ok(path: Path, min_dist: float) -> bool:
    """Check that parts of the path far apart along s are also far apart in space."""
    xy = path.xy[::4]
    s = path.s[::4]
    tree = cKDTree(xy)
    pairs = tree.query_pairs(min_dist, output_type="ndarray")
    if len(pairs) == 0:
        return True
    ds = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
    if path.closed:
        ds = np.minimum(ds, path.length - ds)
    return not np.any(ds > 1.6 * min_dist)


# ---------------------------------------------------------------------------
# random courses


def _uniform(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _fillet_random(rng, v, c) -> Path | None:
    """Round the corners of polygon v with random radii that fit the edges."""
    a = v - np.roll(v, 1, axis=0)
    e_in = np.linalg.norm(a, axis=1)
    e_out = np.roll(e_in, -1)
    a /= e_in[:, None]
    b = np.roll(a, -1, axis=0)
    theta = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))
    # tangent length must be ≤ half of the shorter adjacent edge
    r_allow = (np.minimum(e_in, e_out) - c["min_straight"]) / 2 / np.tan(np.maximum(theta, 1e-3) / 2)
    r_hi = np.minimum(c["radius"][1], r_allow)
    if np.any(r_hi < c["radius"][0]):
        return None
    return fillet_polygon(v, rng.uniform(c["radius"][0], r_hi), c["min_straight"])


def _polygon_vertices(rng, c, kind):
    bx, by = _uniform(rng, c["box_x"]), _uniform(rng, c["box_y"])
    if kind == "rounded_rect":       # oval / stadium / rounded rectangle (like the real course)
        return np.array([[0, 0], [bx, 0], [bx, by], [0, by]], float)
    if kind == "notched":            # rectangle with a notch → S-bends (concave corners)
        x1 = rng.uniform(0.2, 0.5) * bx
        x2 = x1 + rng.uniform(0.25, 0.45) * bx
        d = rng.uniform(0.3, 0.55) * by
        return np.array([[0, 0], [bx, 0], [bx, by], [x2, by], [x2, by - d], [x1, by - d], [x1, by], [0, by]], float)
    if kind == "l_shape":            # one corner cut away
        w, d = rng.uniform(0.35, 0.6) * bx, rng.uniform(0.35, 0.6) * by
        return np.array([[0, 0], [bx, 0], [bx, by - d], [bx - w, by - d], [bx - w, by], [0, by]], float)
    # random star-shaped polygon (may be non-convex)
    n = int(rng.integers(c["n_vertices"][0], c["n_vertices"][1] + 1))
    ang = np.sort((np.arange(n) + rng.uniform(-0.3, 0.3, n)) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi))
    rad = rng.uniform(0.45, 1.0, n)
    return np.stack([np.cos(ang) * rad * bx / 2, np.sin(ang) * rad * by / 2], axis=1)


def random_closed_center(rng, cfg, road_half: float) -> Path | None:
    c = cfg["closed"]
    kinds = list(c["shape_probs"].keys())
    kind = rng.choice(kinds, p=np.asarray(list(c["shape_probs"].values())) / sum(c["shape_probs"].values()))
    path = _fillet_random(rng, _polygon_vertices(rng, c, kind), c)
    if path is None:
        return None
    if not self_clearance_ok(path, 2 * road_half + cfg["clearance"]):
        return None
    return path


def random_open_center(rng, cfg, road_half: float) -> Path | None:
    o = cfg["open"]
    total = _uniform(rng, o["length"])
    kmax = 1.0 / o["min_radius"]
    segs, length, k = [], 0.0, 0.0
    while length < total:
        L = _uniform(rng, o["seg_length"])
        kind = rng.choice(["straight", "arc", "clothoid"], p=[0.3, 0.4, 0.3])
        if kind == "straight":
            k1 = 0.0
            if abs(k) > 1e-9:  # ease out of the previous curve
                segs.append((min(L, 0.3), k, 0.0))
                length += min(L, 0.3)
            segs.append((L, 0.0, 0.0))
        elif kind == "arc":
            k1 = float(rng.uniform(-kmax, kmax))
            segs.append((0.2, k, k1))  # short transition into the arc
            segs.append((L, k1, k1))
            length += 0.2
        else:
            k1 = float(rng.uniform(-kmax, kmax))
            segs.append((L, k, k1))
        k = k1
        length += L
    path = integrate_segments(segs, h0=float(rng.uniform(-np.pi, np.pi)))
    if not self_clearance_ok(path, 2 * road_half + cfg["clearance"]):
        return None
    return path


def random_course(rng, cfg) -> Course:
    closed = rng.random() < cfg["closed_course_prob"]  # decide first so retries keep the ratio
    for _ in range(2000):
        lane_w = _uniform(rng, cfg["lane_width"])
        dashed = cfg.get("center_line", "none") == "dashed" and rng.random() > cfg.get("single_lane_prob", 0.0)
        road_half = (lane_w if dashed else lane_w / 2) + 0.03
        if closed:
            center = random_closed_center(rng, cfg, road_half)
        else:
            center = random_open_center(rng, cfg, road_half)
        if center is None:
            continue
        # inner offsets must not invert the curvature
        if np.any(np.abs(center.kappa) * road_half > 0.95):
            continue
        dash = _uniform(rng, cfg["dash_length"])
        return Course(center, lane_w, _uniform(rng, cfg["line_width"]), dashed,
                      dash, dash * _uniform(rng, cfg["dash_gap_ratio"]))
    raise RuntimeError("failed to generate a course; check track config")
