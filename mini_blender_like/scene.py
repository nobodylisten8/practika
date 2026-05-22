from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class Material:
    name: str = "mat"
    base_color: Tuple[float, float, float] = (0.2, 0.6, 0.95)
    texture_path: Optional[str] = None  # for "variant 1" we keep it as metadata


@dataclass
class Object3D:
    name: str
    kind: str  # sphere, box, plane, mesh_obj
    transform: Dict[str, Any] = field(default_factory=dict)
    material: Material = field(default_factory=Material)

    # SDF primitive parameters (for ray marching)
    # sphere: center, radius
    # box: center, half_size
    # plane: normal, h (dot(p,n)+h=0)
    sdf_params: Dict[str, Any] = field(default_factory=dict)

    # Mesh data (for viewport display; for obj import)
    vertices: Optional[np.ndarray] = None  # (N,3)
    faces: Optional[np.ndarray] = None     # (M,3) indices


@dataclass
class Scene:
    objects: List[Object3D] = field(default_factory=list)
    background_path: Optional[str] = None

    def add(self, obj: Object3D) -> None:
        self.objects.append(obj)

    def remove_by_index(self, idx: int) -> None:
        if 0 <= idx < len(self.objects):
            self.objects.pop(idx)

    def clear(self) -> None:
        self.objects.clear()
        self.background_path = None

    def summary_rows(self):
        rows = []
        for i, o in enumerate(self.objects):
            rows.append({
                "index": i,
                "name": o.name,
                "kind": o.kind,
                "color": o.material.base_color,
                "texture_path": o.material.texture_path,
            })
        return rows