from __future__ import annotations
from typing import Tuple, List
import numpy as np


def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very small OBJ loader:
      - supports 'v x y z'
      - supports 'f i j k' or 'f i/.. j/.. k/..' (triangles)
      - ignores quads and other features (can be triangulated externally)
    Returns (vertices Nx3, faces Mx3 int32 0-based)
    """
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    # keep it simple: only triangles
                    continue
                idxs = []
                for p in parts:
                    # handle i / i/t / i/t/n
                    token = p.split("/")[0]
                    if token:
                        idxs.append(int(token) - 1)
                if len(idxs) == 3:
                    faces.append(idxs)

    if not verts:
        raise ValueError("OBJ: no vertices found.")
    if not faces:
        raise ValueError("OBJ: no triangle faces found (only triangles supported).")

    v = np.array(verts, dtype=np.float64)
    f_idx = np.array(faces, dtype=np.int32)
    return v, f_idx