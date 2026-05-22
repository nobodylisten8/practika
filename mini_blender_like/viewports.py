from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class OrbitCamera:
    yaw: float = 35.0    # degrees
    pitch: float = -20.0 # degrees
    dist: float = 6.0
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def rotation_matrix(self) -> np.ndarray:
        yaw = np.deg2rad(self.yaw)
        pitch = np.deg2rad(self.pitch)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        # yaw around Y, pitch around X
        Ry = np.array([[cy, 0, sy],
                       [0,  1, 0 ],
                       [-sy,0, cy]],
                      dtype=np.float64)
        Rx = np.array([[1, 0,  0 ],
                       [0, cp, -sp],
                       [0, sp, cp]], dtype=np.float64)
        return Ry @ Rx

    def position(self) -> np.ndarray:
        R = self.rotation_matrix()
        # camera in local space at (0,0,dist) looking at origin
        local = np.array([0.0, 0.0, self.dist], dtype=np.float64)
        pos = np.array(self.target, dtype=np.float64) + (R @ local)
        return pos

    def view_matrix(self) -> np.ndarray:
        # simple look-at to target
        eye = self.position()
        target = np.array(self.target, dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        f = target - eye
        f = f / (np.linalg.norm(f) + 1e-12)
        r = np.cross(f, up)
        r = r / (np.linalg.norm(r) + 1e-12)
        u = np.cross(r, f)

        # view matrix (3x4)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = r
        M[1, :3] = u
        M[2, :3] = -f
        M[:3, 3] = -M[:3, :3] @ eye
        return M


def project_points_perspective(P: np.ndarray, cam: OrbitCamera,
                               fov_deg: float, aspect: float) -> np.ndarray:
    """
    P: (N,3) world points -> returns (N,2) normalized screen coords in [-1,1]
    """
    V = cam.view_matrix()
    N = P.shape[0]
    Ph = np.concatenate([P, np.ones((N, 1), dtype=np.float64)], axis=1)
    Pc = (V @ Ph.T).T[:, :3]  # camera space

    # perspective
    fov = np.deg2rad(fov_deg)
    f = 1.0 / np.tan(fov / 2.0)
    x = (Pc[:, 0] * f / aspect) / (-Pc[:, 2] + 1e-12)
    y = (Pc[:, 1] * f) / (-Pc[:, 2] + 1e-12)

    return np.stack([x, y], axis=1)


def project_points_ortho(P: np.ndarray, mode: str = "XY") -> np.ndarray:
    """
    mode: XY, XZ, YZ
    returns (N,2)
    """
    if mode == "XY":
        return P[:, [0, 1]]
    if mode == "XZ":
        return P[:, [0, 2]]
    if mode == "YZ":
        return P[:, [1, 2]]
    raise ValueError("Unknown ortho mode")