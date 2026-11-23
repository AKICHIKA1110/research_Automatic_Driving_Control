"""Geometry checks: run with `python3 -m pytest tests` in the synth directory."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lane_synth import load_config  # noqa: E402
from lane_synth.camera import Camera  # noqa: E402
from lane_synth.labels import compute_labels  # noqa: E402
from lane_synth.render import BevWarper, CameraRenderer, WorldRaster, base_pose_matrix  # noqa: E402
from lane_synth.track import Course, fillet_polygon, integrate_segments  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def straight_course(dashed=True, lane_width=0.38):
    center = integrate_segments([(6.0, 0.0, 0.0)], p0=(-1.0, 0.0), h0=0.0)
    return Course(center, lane_width, 0.02, dashed, 0.1, 0.1)


def test_path_curvature_and_closure():
    sq = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], float)
    p = fillet_polygon(sq, np.full(4, 0.7))
    assert p is not None and p.closed
    # perimeter of a rounded rectangle
    expected = 2 * (3 + 2) - 8 * 0.7 + 2 * np.pi * 0.7
    assert abs(p.length - expected) < 0.02
    assert np.allclose(np.abs(p.kappa).max(), 1 / 0.7)


def test_offset_curvature():
    p = integrate_segments([(np.pi * 0.5, 2.0, 2.0)])  # left arc R = 0.5
    q = p.offset(0.1)                                  # inner side
    assert np.allclose(q.kappa, 1 / 0.4, rtol=1e-6)


def test_projection_roundtrip(cfg):
    cam = Camera.from_config(cfg["camera"])
    P = np.array([[0.5, 0.0, 0.0], [0.8, 0.2, 0.0], [0.6, -0.15, 0.0]])
    uv, front = cam.project_base(P)
    assert front.all() and cam.in_image(uv, front).all()
    # the point straight ahead lies on the image's vertical center line
    assert abs(uv[0, 0] - cfg["camera"]["cx"]) < 1e-6


def test_labels_straight(cfg):
    course = straight_course()
    path = course.lane_path(1, True)  # right lane, y = -0.19
    cam = Camera.from_config(cfg["camera"])
    R, t = base_pose_matrix(0.0, -0.19, 0.0)
    lab = compute_labels(path, (0.0, -0.19, 0.0), None, cfg["labels"], cfg["bev"], cam, R, t)
    assert np.allclose(lab["y_pts"], 0.0, atol=1e-6)
    assert abs(lab["e_y"]) < 1e-6 and abs(lab["e_psi"]) < 1e-9
    assert lab["vis"].all()
    assert np.allclose(lab["kappa"], 0.0)


def test_labels_circle(cfg):
    R0 = 1.0
    center = integrate_segments([(2 * np.pi * R0 * 0.75, 1 / R0, 1 / R0)], p0=(0, 0), h0=0.0)
    course = Course(center, 0.38, 0.02, False, 0.1, 0.1)
    path = course.lane_path(0, True)  # single lane: centerline = road center, left turn
    cam = Camera.from_config(cfg["camera"])
    Rm, t = base_pose_matrix(0.0, 0.0, 0.0)
    lab = compute_labels(path, (0.0, 0.0, 0.0), None, cfg["labels"], cfg["bev"], cam, Rm, t)
    xs = np.asarray(cfg["labels"]["stations_x"])
    expected = R0 - np.sqrt(R0 ** 2 - xs ** 2)
    ok = lab["vis"].astype(bool)
    assert ok.sum() >= 3
    assert np.allclose(lab["y_pts"][ok], expected[ok], atol=2e-3)
    assert np.allclose(lab["kappa"], 1 / R0, atol=1e-6)


def test_render_bev_line_positions(cfg):
    """Without noise, the BEV image must show the boundary lines at the right place."""
    course = straight_course(dashed=False, lane_width=0.38)
    world = WorldRaster(course, cfg["render"]["world_res"])
    cam = Camera.from_config(cfg["camera"])
    rend = CameraRenderer(cam, 2, 4.0)
    R, t = base_pose_matrix(0.0, 0.0, 0.0)
    prob, _ = rend.render(world, R, t)
    bev = BevWarper(cam, cfg["bev"])
    img = bev.warp((prob * 255).astype(np.uint8)).astype(float)
    row = img[img.shape[0] // 2]          # x ≈ 0.65 m
    cols = np.flatnonzero(row > 100)
    left = cols[cols < img.shape[1] / 2].mean()
    right = cols[cols > img.shape[1] / 2].mean()
    y_left = cfg["bev"]["y_max"] - (left + 0.5) * cfg["bev"]["res"]
    y_right = cfg["bev"]["y_max"] - (right + 0.5) * cfg["bev"]["res"]
    assert abs(y_left - 0.19) < 0.01 and abs(y_right + 0.19) < 0.01


def test_course_yaml_and_exact_drive(cfg):
    import yaml
    from lane_synth.driving import follow_exact
    from lane_synth.track import course_from_dict
    d = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "courses" / "oval.yaml"))
    course = course_from_dict(d, cfg["track"])
    assert course.center.closed and not course.dashed_center
    assert abs(course.center.length - (2 * 1.8 + 2 * np.pi * 0.7)) < 0.01
    path = course.lane_path(0, True)
    frames = list(follow_exact(path, dict(cfg["driving"], offset_y=0.03, heading_offset_deg=2.0)))
    cam = Camera.from_config(cfg["camera"])
    for st in frames[::20]:
        R, t = base_pose_matrix(st["x"], st["y"], st["yaw"])
        lab = compute_labels(path, (st["x"], st["y"], st["yaw"]), st["idx"], cfg["labels"], cfg["bev"], cam, R, t)
        assert abs(lab["e_y"] - 0.03) < 1e-3
        assert abs(np.rad2deg(lab["e_psi"]) - 2.0) < 0.05
