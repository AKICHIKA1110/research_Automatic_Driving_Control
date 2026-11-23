#!/usr/bin/env python3
"""Build the fixed course set in config/courses/ (re-run to regenerate the YAMLs).

  python3 scripts/build_courses.py            # write config/courses/*.yaml
  python3 scripts/build_courses.py --check    # only validate and print a summary

Closed courses are made so that they close exactly:
  * polygon courses: corners of a polygon rounded with arcs (radius per corner;
    a clockwise corner becomes a right turn, so S-bends appear)
  * symmetric courses with clothoid transitions (each corner turns exactly 90 / 180 deg)
Open roads are free sequences of straight / arc / clothoid segments.
All courses: lane width 0.40 m (two white lines, no center line), line width 0.02 m.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lane_synth.track import course_from_dict  # noqa: E402

LANE_W, LINE_W = 0.40, 0.02
OUT = ROOT / "config" / "courses"


def r6(x):
    return float(round(float(x), 6))


# ---------------------------------------------------------------------------
# builders

def polygon(vertices, radii):
    """Segments of a closed path: polygon corners rounded with arcs.

    vertices: corner points (counter-clockwise overall), radii: arc radius per corner.
    Returns (start_xy, start_heading_deg, segments). Raises if an edge is too short.
    """
    v = np.asarray(vertices, float)
    r = np.broadcast_to(np.asarray(radii, float), (len(v),))
    n = len(v)
    a = v - np.roll(v, 1, axis=0)
    a /= np.linalg.norm(a, axis=1, keepdims=True)          # incoming direction at corner i
    b = np.roll(a, -1, axis=0)                              # outgoing direction
    theta = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
    if abs(theta.sum() - 2 * np.pi) > 1e-6:
        raise ValueError("polygon must be simple and counter-clockwise")
    t = r * np.tan(np.abs(theta) / 2)                       # tangent length
    edge = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)   # edge i -> i+1
    straight = edge - t - np.roll(t, -1)
    if np.any(straight < 0.05):
        raise ValueError(f"edge too short for the radii: {np.round(straight, 3)}")
    segs = []
    for i in range(n):
        segs.append(dict(type="arc", radius=r6(r[i]), angle_deg=r6(np.rad2deg(theta[i]))))
        segs.append(dict(type="straight", length=r6(straight[i])))
    p_in = v[0] - a[0] * t[0]
    return [r6(p_in[0]), r6(p_in[1])], r6(np.rad2deg(np.arctan2(a[0, 1], a[0, 0]))), segs


def clothoid_corner(R, L, turn_deg):
    """clothoid (0 -> 1/R, length L) + arc + clothoid (1/R -> 0); total turn = turn_deg."""
    k = np.sign(turn_deg) / R
    arc = abs(turn_deg) - np.rad2deg(abs(k) * L)            # each clothoid turns k*L/2
    if arc <= 0:
        raise ValueError("clothoids too long for this corner")
    return [dict(type="clothoid", length=r6(L), k0=0.0, k1=r6(k)),
            dict(type="arc", radius=r6(R), angle_deg=r6(np.sign(turn_deg) * arc)),
            dict(type="clothoid", length=r6(L), k0=r6(k), k1=0.0)]


# ---------------------------------------------------------------------------
# the course set

def courses():
    C = {}

    def add(name, note, closed, segments, start_xy=(0.0, 0.0), start_heading=0.0):
        C[name] = dict(note=note, closed=closed, segments=segments,
                       start_xy=list(start_xy), start_heading_deg=start_heading)

    # --- closed: simple shapes --------------------------------------------
    add("oval_large", "大きいオーバル：直線 4.0 m + 半径 1.5 m の半円 × 2（ゆるいカーブ・長い直線）", True,
        [dict(type="straight", length=4.0), dict(type="arc", radius=1.5, angle_deg=180),
         dict(type="straight", length=4.0), dict(type="arc", radius=1.5, angle_deg=180)])

    add("circle_r1.2", "半径 1.2 m の円（曲率一定 κ = 0.83 を連続して見る）", True,
        [dict(type="arc", radius=1.2, angle_deg=180), dict(type="arc", radius=1.2, angle_deg=180)])

    add("stadium_clothoid", "クロソイド付きオーバル：直線 3.0 m、半径 0.9 m、緩和曲線 0.6 m（曲率が連続に変化）", True,
        [dict(type="straight", length=3.0)] + clothoid_corner(0.9, 0.6, 180) +
        [dict(type="straight", length=3.0)] + clothoid_corner(0.9, 0.6, 180))

    add("rect_clothoid", "クロソイド付き角丸長方形：5.0 m × 3.0 m の直線、角は半径 0.8 m・緩和曲線 0.4 m", True,
        sum(([dict(type="straight", length=L)] + clothoid_corner(0.8, 0.4, 90) for L in (5.0, 3.0, 5.0, 3.0)), []))

    add("hairpin_paperclip", "ヘアピン：直線 5.0 m + 半径 0.55 m の 180° ターン × 2（急カーブ）", True,
        [dict(type="straight", length=5.0), dict(type="arc", radius=0.55, angle_deg=180),
         dict(type="straight", length=5.0), dict(type="arc", radius=0.55, angle_deg=180)])

    # --- closed: polygon circuits (right turns / S-bends) -------------------
    xy, h, s = polygon([(0, 0), (6, 0), (6, 4), (3.5, 4), (3.5, 2.2), (2.5, 2.2), (2.5, 4), (0, 4)],
                       [1.0, 1.0, 0.8, 0.55, 0.45, 0.45, 0.55, 1.0])
    add("notched_rect", "切り欠き付き長方形 6 m × 4 m：上辺の凹みで右カーブと S 字（半径 0.45〜1.0 m）", True, s, xy, h)

    xy, h, s = polygon([(0, 0), (5, 0), (5, 2.5), (2.6, 2.5), (2.6, 5), (0, 5)],
                       [1.2, 0.9, 0.7, 0.7, 0.9, 1.2])
    add("l_shape", "L 字コース 5 m × 5 m：内角は右カーブ（半径 0.7〜1.2 m）", True, s, xy, h)

    xy, h, s = polygon([(0, 0), (4, -0.6), (7.5, 0.4), (8.5, 3.0), (6.5, 4.5), (5.0, 3.2), (3.2, 4.6), (0.6, 3.6)],
                       [1.4, 2.0, 1.0, 0.8, 0.6, 0.6, 1.0, 1.1])
    add("technical_circuit", "テクニカルサーキット 9 m × 5 m：大小のカーブ・角度の違うコーナー・右カーブ（半径 0.6〜2.0 m）", True, s, xy, h)

    xy, h, s = polygon([(0, 0), (12, 0), (12, 3), (9, 4.5), (12, 7), (12, 10), (6, 10), (4, 6.5), (0, 8.5)],
                       [2.0, 1.5, 1.2, 1.0, 1.2, 1.6, 1.0, 1.0, 2.0])
    add("grand_circuit", "大型サーキット 13 m × 10 m：長い直線・高速コーナー・切り返し（半径 1.0〜2.0 m）", True, s, xy, h)

    xy, h, s = polygon([(0, 0), (3, 0.8), (6, 0), (7, 2.2), (6, 4.5), (3, 3.7), (0, 4.5), (-1, 2.2)],
                       [0.9, 1.4, 0.9, 1.0, 0.9, 1.4, 0.9, 1.0])
    add("bean", "豆形 8 m × 4.5 m：長辺がくびれて、ゆるい右カーブが入る（半径 0.9〜1.4 m）", True, s, xy, h)

    # --- open roads -----------------------------------------------------
    k = lambda R: 1.0 / R  # noqa: E731
    add("open_winding", "ワインディング道路（約 19 m）：半径 0.6〜2.5 m の左右カーブをクロソイドでつなぐ", False,
        [dict(type="straight", length=1.5),
         dict(type="clothoid", length=0.5, k0=0.0, k1=k(2.5)), dict(type="arc", radius=2.5, angle_deg=40),
         dict(type="clothoid", length=1.0, k0=k(2.5), k1=-k(1.0)), dict(type="arc", radius=1.0, angle_deg=-70),
         dict(type="clothoid", length=0.6, k0=-k(1.0), k1=0.0), dict(type="straight", length=1.2),
         dict(type="clothoid", length=0.5, k0=0.0, k1=k(0.6)), dict(type="arc", radius=0.6, angle_deg=80),
         dict(type="clothoid", length=0.8, k0=k(0.6), k1=-k(1.5)), dict(type="arc", radius=1.5, angle_deg=-90),
         dict(type="clothoid", length=0.5, k0=-k(1.5), k1=0.0), dict(type="straight", length=1.0),
         dict(type="clothoid", length=0.6, k0=0.0, k1=-k(0.8)), dict(type="arc", radius=0.8, angle_deg=-60),
         dict(type="clothoid", length=0.6, k0=-k(0.8), k1=0.0), dict(type="straight", length=2.5)])

    add("open_chicane", "シケイン（約 12 m）：短い直線をはさんだ左右左の切り返し（半径 0.7 m）", False,
        [dict(type="straight", length=2.0),
         dict(type="arc", radius=0.7, angle_deg=45), dict(type="straight", length=0.3),
         dict(type="arc", radius=0.7, angle_deg=-90), dict(type="straight", length=0.3),
         dict(type="arc", radius=0.7, angle_deg=90), dict(type="straight", length=0.3),
         dict(type="arc", radius=0.7, angle_deg=-45), dict(type="straight", length=1.5),
         dict(type="arc", radius=0.7, angle_deg=-60), dict(type="arc", radius=0.7, angle_deg=60),
         dict(type="straight", length=3.0)])

    add("open_spiral", "スパイラル（約 8 m）：曲率 0 → 1/0.6 → 0 をゆっくり変化（連続した κ の値を網羅）", False,
        [dict(type="straight", length=1.5),
         dict(type="clothoid", length=1.8, k0=0.0, k1=k(0.6)),
         dict(type="clothoid", length=1.8, k0=k(0.6), k1=0.0),
         dict(type="straight", length=2.5)])
    return C


# ---------------------------------------------------------------------------

def min_gap(path, sep):
    """Smallest distance between parts of the path more than `sep` apart along s."""
    from scipy.spatial import cKDTree
    xy, s = path.xy[::2], path.s[::2]
    pairs = cKDTree(xy).query_pairs(3.0, output_type="ndarray")
    ds = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
    if path.closed:
        ds = np.minimum(ds, path.length - ds)
    far = pairs[ds > sep]
    if len(far) == 0:
        return np.inf
    return float(np.min(np.linalg.norm(xy[far[:, 0]] - xy[far[:, 1]], axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    track_cfg = yaml.safe_load(open(ROOT / "config" / "default.yaml"))["track"]
    ok = True
    print(f"{'course':20s} {'type':6s} {'length':>7s} {'Rmin':>6s} {'gap':>6s} size")
    for name, c in courses().items():
        d = dict(name=name, closed=c["closed"], center_line="none", lane_width=LANE_W, line_width=LINE_W,
                 start_xy=c["start_xy"], start_heading_deg=c["start_heading_deg"], segments=c["segments"])
        course = course_from_dict(d, track_cfg)          # raises if a closed course does not close
        p = course.center
        rmin = 1.0 / max(np.abs(p.kappa).max(), 1e-9)
        gap = min_gap(p, 1.5)                             # road centerline to another part of the road
        need = LANE_W + LINE_W + 0.10
        size = p.xy.max(0) - p.xy.min(0)
        flag = "" if gap >= need else f"  <-- too close (need {need:.2f} m)"
        ok &= gap >= need
        print(f"{name:20s} {'closed' if c['closed'] else 'open':6s} {p.length:6.2f}m {rmin:5.2f}m "
              f"{min(gap, 99):5.2f}m {size[0]:.1f}x{size[1]:.1f}m{flag}")
        if not a.check:
            with open(OUT / f"{name}.yaml", "w") as f:
                f.write(f"# {c['note']}\n# scripts/build_courses.py で生成（手で編集する場合はスクリプト側を直す）\n")
                yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False, default_flow_style=None, width=120)
    if not ok:
        sys.exit("some courses fail the clearance check")


if __name__ == "__main__":
    main()
