from __future__ import annotations
from typing import Tuple, List
import numpy as np


def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Компактный загрузчик OBJ:
      - поддерживает строки 'v x y z'
      - поддерживает 'f i j k' или 'f i/.. j/.. k/..' (только треугольники)
      - игнорирует четырёхугольники и прочие элементы (их можно триангулировать внешними средствами)
    Возвращает: (вершины — массив Nx3, грани — массив Mx3 типа int32, индексация с нуля)
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
                    continue
                idxs = []
                for p in parts:
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
