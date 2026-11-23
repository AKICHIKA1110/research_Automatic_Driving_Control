"""Camera model: intrinsics, distortion and mounting pose on base_link."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

# camera body (x fwd, y left, z up) -> optical frame (x right, y down, z fwd)
R_BODY_OPT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])


def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    K: np.ndarray       # 3x3
    dist: np.ndarray    # (5,)
    xyz: np.ndarray     # camera position in base_link [m]
    rpy: np.ndarray     # camera orientation in base_link [rad]

    @classmethod
    def from_config(cls, c: dict) -> "Camera":
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
        return cls(int(c["width"]), int(c["height"]), K, np.asarray(c["dist"], float),
                   np.asarray(c["xyz"], float), np.deg2rad(np.asarray(c["rpy_deg"], float)))

    @property
    def R_base_opt(self) -> np.ndarray:
        return rpy_to_R(*self.rpy) @ R_BODY_OPT

    def perturbed(self, rng, pcfg: dict) -> "Camera":
        """Copy with random calibration error (used as the 'true' camera)."""
        drpy = np.deg2rad(rng.uniform(-1, 1, 3) * np.asarray(pcfg["calib_rpy_deg"]))
        dxyz = rng.uniform(-1, 1, 3) * np.asarray(pcfg["calib_xyz"])
        K = self.K.copy()
        K[0, 0] *= 1 + rng.uniform(-1, 1) * pcfg["intrinsic_rel"]
        K[1, 1] *= 1 + rng.uniform(-1, 1) * pcfg["intrinsic_rel"]
        return replace(self, K=K, xyz=self.xyz + dxyz, rpy=self.rpy + drpy)

    def project_base(self, P: np.ndarray):
        """Project points in base_link (N,3) → pixel uv (N,2), in-front flag (N,)."""
        Pc = (P - self.xyz) @ self.R_base_opt          # base -> optical
        front = Pc[:, 2] > 1e-3
        uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2), front

    def in_image(self, uv, front):
        return front & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)

    def pixel_rays(self, ss: int = 1) -> np.ndarray:
        """Unit ray directions in the optical frame for each (super-sampled) pixel."""
        h, w = self.height * ss, self.width * ss
        u, v = np.meshgrid((np.arange(w) + 0.5) / ss - 0.5, (np.arange(h) + 0.5) / ss - 0.5)
        pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-6)
        if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
            n = cv2.undistortPointsIter(pts, self.K, self.dist, None, None, crit)
        else:  # OpenCV 5.x
            n = cv2.undistortPoints(pts, self.K, self.dist, None, None, None, crit)
        n = n.reshape(h, w, 2)
        r = np.concatenate([n, np.ones((h, w, 1))], -1)
        return (r / np.linalg.norm(r, axis=-1, keepdims=True)).astype(np.float32)
