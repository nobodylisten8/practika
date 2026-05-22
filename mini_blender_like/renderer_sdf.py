from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .scene import Scene, Object3D


@dataclass
class RenderConfig:
    width: int = 640
    height: int = 360
    fov_deg: float = 55.0
    max_steps: int = 160
    max_dist: float = 25.0
    eps: float = 1e-3

    # Camera
    camera_pos: Tuple[float, float, float] = (0.0, 0.0, 4.0)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Lighting / background
    bg: float = 0.08
    light_dir: Tuple[float, float, float] = (0.4, 0.7, -0.6)


# ----------------------------
# SDF primitives
# ----------------------------
def sdf_sphere(p: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
    return np.linalg.norm(p - c, axis=-1) - r


def sdf_plane(p: np.ndarray, n: np.ndarray, h: float) -> np.ndarray:
    # plane: dot(p, n) + h = 0
    n = n / (np.linalg.norm(n) + 1e-12)
    return np.sum(p * n, axis=-1) + h


def sdf_box(p: np.ndarray, c: np.ndarray, b: np.ndarray) -> np.ndarray:
    # axis-aligned box with half-sizes b, centered at c
    q = np.abs(p - c) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0)
    return outside + inside


def sdf_cylinder_y(p: np.ndarray, c: np.ndarray, r: float, half_h: float) -> np.ndarray:
    """
    Finite cylinder centered at c, axis along Y, with radius r and half-height half_h.
    """
    qx = p[..., 0] - c[0]
    qz = p[..., 2] - c[2]
    d_xz = np.sqrt(qx*qx + qz*qz) - r
    d_y = np.abs(p[..., 1] - c[1]) - half_h
    outside = np.sqrt(np.maximum(d_xz, 0.0)**2 + np.maximum(d_y, 0.0)**2)
    inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
    return outside + inside


def sdf_torus_y(p: np.ndarray, c: np.ndarray, R: float, r: float) -> np.ndarray:
    """
    Torus centered at c, around Y axis.
      R - major radius, r - minor radius
    """
    x = p[..., 0] - c[0]
    y = p[..., 1] - c[1]
    z = p[..., 2] - c[2]
    q = np.sqrt(x*x + z*z) - R
    return np.sqrt(q*q + y*y) - r


# ----------------------------
# Scene SDF composition
# ----------------------------
def scene_sdf_and_color(scene: Scene):
    objs = [o for o in scene.objects if o.kind in ("sphere", "box", "plane", "cylinder", "torus")]

    def sdf_only(p: np.ndarray) -> np.ndarray:
        dmin = np.full(p.shape[:-1], np.inf, dtype=np.float64)
        for o in objs:
            d = _sdf_obj(o, p)
            dmin = np.minimum(dmin, d)
        return dmin

    def sdf_and_id(p: np.ndarray):
        dmin = np.full(p.shape[:-1], np.inf, dtype=np.float64)
        oid = np.full(p.shape[:-1], -1, dtype=np.int32)
        for i, o in enumerate(objs):
            d = _sdf_obj(o, p)
            mask = d < dmin
            dmin = np.where(mask, d, dmin)
            oid = np.where(mask, i, oid)
        return dmin, oid

    def color_of_id(oid: np.ndarray) -> np.ndarray:
        H, W = oid.shape
        col = np.zeros((H, W, 3), dtype=np.float64)
        for i, o in enumerate(objs):
            rgb = np.array(o.material.base_color, dtype=np.float64)
            col[oid == i] = rgb
        return col

    return sdf_only, sdf_and_id, color_of_id, len(objs)


def _sdf_obj(o: Object3D, p: np.ndarray) -> np.ndarray:
    if o.kind == "sphere":
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        r = float(o.sdf_params["radius"])
        return sdf_sphere(p, c, r)

    if o.kind == "box":
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        b = np.array(o.sdf_params["half_size"], dtype=np.float64)
        return sdf_box(p, c, b)

    if o.kind == "plane":
        n = np.array(o.sdf_params["normal"], dtype=np.float64)
        h = float(o.sdf_params["h"])
        return sdf_plane(p, n, h)

    if o.kind == "cylinder":
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        r = float(o.sdf_params["radius"])
        half_h = float(o.sdf_params["half_height"])
        return sdf_cylinder_y(p, c, r, half_h)

    if o.kind == "torus":
        c = np.array(o.sdf_params["center"], dtype=np.float64)
        R = float(o.sdf_params["major_radius"])
        r = float(o.sdf_params["minor_radius"])
        return sdf_torus_y(p, c, R, r)

    return np.full(p.shape[:-1], np.inf, dtype=np.float64)


def estimate_normal_scene(p: np.ndarray, sdf_only, eps: float) -> np.ndarray:
    ex = np.array([eps, 0.0, 0.0])
    ey = np.array([0.0, eps, 0.0])
    ez = np.array([0.0, 0.0, eps])
    dx = sdf_only(p + ex) - sdf_only(p - ex)
    dy = sdf_only(p + ey) - sdf_only(p - ey)
    dz = sdf_only(p + ez) - sdf_only(p - ez)
    n = np.stack([dx, dy, dz], axis=-1)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n / n_norm


# ----------------------------
# Background helpers (no PIL)
# ----------------------------
def load_background_via_matplotlib(path: str) -> np.ndarray:
    import matplotlib.pyplot as plt
    img = plt.imread(path)
    if img.dtype.kind in ("u", "i"):
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.clip(img, 0.0, 1.0)


def resize_nearest(bg: np.ndarray, H: int, W: int) -> np.ndarray:
    y_idx = (np.linspace(0, bg.shape[0] - 1, H)).astype(int)
    x_idx = (np.linspace(0, bg.shape[1] - 1, W)).astype(int)
    return bg[y_idx][:, x_idx]


# ----------------------------
# Camera helpers (look-at)
# ----------------------------
def _camera_basis(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (right, up2, forward) unit vectors for a look-at camera.
    forward points from eye to target.
    """
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)

    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-12)

    u = np.cross(r, f)
    u = u / (np.linalg.norm(u) + 1e-12)

    return r, u, f


# ----------------------------
# Render
# ----------------------------
def render(scene: Scene, cfg: RenderConfig,
           out_dir: Optional[str] = None,
           save_prefix: str = "render") -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Improvements vs old version:
      - Camera is now a real look-at camera (pos + target + up), not hardcoded -Z.
        This avoids "weird angle" when caller changes camera_pos.
      - Caller can auto-fit camera_pos/max_dist, but renderer now respects direction.
    """
    W, H = cfg.width, cfg.height
    aspect = W / H
    fov = np.deg2rad(cfg.fov_deg)

    cam = np.array(cfg.camera_pos, dtype=np.float64)
    target = np.array(cfg.camera_target, dtype=np.float64)
    up = np.array(cfg.camera_up, dtype=np.float64)

    right, up2, forward = _camera_basis(cam, target, up)

    light = np.array(cfg.light_dir, dtype=np.float64)
    light = light / (np.linalg.norm(light) + 1e-12)

    sdf_only, sdf_and_id, color_of_id, sdf_object_count = scene_sdf_and_color(scene)

    bg_img = None
    if scene.background_path:
        try:
            bg_img = load_background_via_matplotlib(scene.background_path)
        except Exception:
            bg_img = None

    # Rays in camera space: build directions for each pixel in world space
    ys, xs = np.mgrid[0:H, 0:W]
    x = (2 * (xs + 0.5) / W - 1) * np.tan(fov / 2) * aspect
    y = (1 - 2 * (ys + 0.5) / H) * np.tan(fov / 2)

    # dir = normalize(forward + x*right + y*up2)
    dirs = (forward[None, None, :] + x[..., None] * right[None, None, :] + y[..., None] * up2[None, None, :])
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12)

    t = np.zeros((H, W), dtype=np.float64)
    hit = np.zeros((H, W), dtype=bool)
    oid = np.full((H, W), -1, dtype=np.int32)

    start = time.perf_counter()

    for _ in range(cfg.max_steps):
        p = cam + dirs * t[..., None]
        dist, cur_oid = sdf_and_id(p)

        newly_hit = dist < cfg.eps
        oid = np.where(newly_hit & (oid == -1), cur_oid, oid)
        hit = hit | newly_hit

        t = np.where(hit, t, t + dist)

        too_far = t > cfg.max_dist
        # NOTE: we keep too_far pixels as background; mark as not-hit
        hit = np.where(too_far, False, hit)
        oid = np.where(too_far, -1, oid)

    # Background base
    if bg_img is not None:
        img = resize_nearest(bg_img, H, W).copy()
    else:
        img = np.full((H, W, 3), cfg.bg, dtype=np.float64)

    # Shade hits
    if np.any(hit) and sdf_object_count > 0:
        p_hit = cam + dirs[hit] * t[hit, None]
        n = estimate_normal_scene(p_hit, sdf_only, cfg.eps)

        lambert = np.clip(np.sum(n * (-light), axis=-1), 0.0, 1.0)
        ambient = 0.22
        shade = ambient + (1.0 - ambient) * lambert

        col = color_of_id(oid)
        img[hit] = col[hit] * shade[:, None]

    end = time.perf_counter()

    metrics = {
        "width": W,
        "height": H,
        "pixels": int(W * H),
        "scene_objects_total": int(len(scene.objects)),
        "sdf_objects_used": int(sdf_object_count),
        "max_steps": int(cfg.max_steps),
        "max_dist": float(cfg.max_dist),
        "eps": float(cfg.eps),
        "render_time_sec": float(end - start),
        "hit_ratio": float(hit.mean()),
        "mean_intensity": float(img.mean()),
        "camera_pos": tuple(map(float, cfg.camera_pos)),
        "camera_target": tuple(map(float, cfg.camera_target)),
        "camera_up": tuple(map(float, cfg.camera_up)),
    }

    if out_dir:
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"{save_prefix}.png")
        plt.imsave(png_path, np.clip(img, 0.0, 1.0))

        pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"{save_prefix}_metrics.csv"), index=False)

        summary = {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "render_config": asdict(cfg),
            "metrics": metrics,
            "background_path": scene.background_path,
            "scene_objects": [object_to_dict(o) for o in scene.objects],
        }
        pd.DataFrame([summary]).to_json(
            os.path.join(out_dir, f"{save_prefix}_summary.json"),
            orient="records",
            force_ascii=False,
            indent=2
        )

    return np.clip(img, 0.0, 1.0), metrics


def object_to_dict(o: Object3D) -> Dict[str, Any]:
    d = {
        "name": o.name,
        "kind": o.kind,
        "material": {
            "name": o.material.name,
            "base_color": o.material.base_color,
            "texture_path": o.material.texture_path,
        },
        "sdf_params": o.sdf_params,
    }
    if o.vertices is not None:
        d["mesh_vertices_count"] = int(o.vertices.shape[0])
    if o.faces is not None:
        d["mesh_faces_count"] = int(o.faces.shape[0])
    return d