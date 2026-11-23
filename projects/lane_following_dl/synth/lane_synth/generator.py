"""Sample generation: course → virtual drive → camera mask → BEV + labels."""
from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np
import yaml

from .camera import Camera
from .driving import drive, follow_exact
from .labels import compute_labels
from .render import BevWarper, CameraRenderer, WorldRaster, base_pose_matrix, detection_noise
from .track import Course, random_course

DEFAULT_CONFIG = FsPath(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


class Generator:
    """Renders frames of a vehicle driving on a course.

    The extra effects are switched in the config and are all off by default:
      perturb.enable          calibration error per sequence, body pitch/roll per frame
      detection_noise.enable  degrade the ideal mask like a line-detection CNN output
      driving.mode            "exact" (follow the set trajectory) | "pure_pursuit"
    """

    def __init__(self, cfg: dict, seed: int = 0, keep_camera: bool = True):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.cam_nominal = Camera.from_config(cfg["camera"])
        self.bev = BevWarper(self.cam_nominal, cfg["bev"])
        self.keep_camera = keep_camera

    # -- one run on a given course (simulation tool) -------------------------
    def run(self, course: Course, lane: int = 0, forward: bool = True, drive_cfg: dict | None = None):
        """Frames of one drive on `course` with the exact-trajectory mode."""
        cfg = self.cfg
        world = WorldRaster(course, cfg["render"]["world_res"], self.rng if self._noise_on() else None,
                            cfg["track"] if self._noise_on() else None)
        path = course.lane_path(lane, forward)
        cam_true = self._true_camera()
        renderer = CameraRenderer(cam_true, cfg["render"]["supersample"], cfg["render"]["max_range"])
        dcfg = dict(cfg["driving"], **(drive_cfg or {}))
        for st in follow_exact(path, dcfg):
            yield self._sample(course, world, path, cam_true, renderer, st, 0)

    # -- endless random stream (dataset generation) ---------------------------
    def sequences(self):
        cfg, rng = self.cfg, self.rng
        seq_id = 0
        while True:
            course = random_course(rng, cfg["track"])
            world = WorldRaster(course, cfg["render"]["world_res"], rng if self._noise_on() else None,
                                cfg["track"] if self._noise_on() else None)
            lane = int(rng.integers(len(course.lane_center_offsets())))
            forward = bool(rng.random() < 0.5)
            path = course.lane_path(lane, forward)
            cam_true = self._true_camera()
            renderer = CameraRenderer(cam_true, cfg["render"]["supersample"], cfg["render"]["max_range"])
            d = cfg["driving"]
            if d.get("mode", "exact") == "exact":
                n = int(rng.integers(*d["frames_per_seq"]))
                dcfg = dict(d, speed=float(rng.uniform(*d["speed_range"])),
                            start_s=float(rng.uniform(0, path.length)) if path.closed else 0.0,
                            distance=n * d["speed_range"][1] / d["fps"])
                frames = follow_exact(path, dcfg)
            else:
                frames = drive(path, rng, d, int(rng.integers(*d["frames_per_seq"])))
            for st in frames:
                yield self._sample(course, world, path, cam_true, renderer, st, seq_id)
            seq_id += 1

    # -------------------------------------------------------------------------
    def _noise_on(self):
        return bool(self.cfg["detection_noise"].get("enable", False))

    def _true_camera(self) -> Camera:
        p = self.cfg["perturb"]
        return self.cam_nominal.perturbed(self.rng, p) if p.get("enable", False) else self.cam_nominal

    def _sample(self, course, world, path, cam_true, renderer, st, seq_id):
        cfg, rng = self.cfg, self.rng
        p = cfg["perturb"]
        pitch = roll = 0.0
        if p.get("enable", False):
            pitch = np.deg2rad(rng.normal(0, p["body_pitch_deg"]) - p["pitch_per_accel_deg"] * st["accel"])
            roll = np.deg2rad(rng.normal(0, p["body_roll_deg"]) + p["roll_per_latacc_deg"] * st["lat_acc"])
        R_wb, t_wb = base_pose_matrix(st["x"], st["y"], st["yaw"], roll, pitch)
        prob, rng_img = renderer.render(world, R_wb, t_wb)
        cam_mask = detection_noise(prob, rng_img, rng, cfg["detection_noise"])
        bev = self.bev.warp(cam_mask)
        lab = compute_labels(path, (st["x"], st["y"], st["yaw"]), st["idx"], cfg["labels"],
                             cfg["bev"], cam_true, R_wb, t_wb, half_width=course.lane_width / 2)
        out = dict(bev=bev, y_pts=lab["y_pts"], vis=lab["vis"], e_y=lab["e_y"], e_psi=lab["e_psi"],
                   kappa=lab["kappa"], path_xy=lab["path_xy"], path_valid=lab["path_valid"], v=st["v"], delta=st["delta"], seq=seq_id,
                   lane_width=course.lane_width, dashed=course.dashed_center,
                   pose=(st["x"], st["y"], st["yaw"]), s=st.get("s", np.nan), t=st.get("t", np.nan))
        if self.keep_camera:
            out["cam"] = cam_mask
            out["_debug"] = dict(course=course, path=path, pose=out["pose"], cam_true=cam_true, clean=prob)
        return out
