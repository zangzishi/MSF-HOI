from __future__ import annotations

from pathlib import Path

import numpy as np


def load_object_keypoints(source_hdf5_path: str, object_name: str) -> np.ndarray:
    dataset_root = Path(source_hdf5_path).parent
    keypoint_path = dataset_root / "object_keypoints_meshsample1500" / f"{object_name}.npz"
    if not keypoint_path.exists():
        raise FileNotFoundError(f"object keypoint file not found: {keypoint_path}")
    loaded = np.load(keypoint_path)
    if "cartesian" not in loaded:
        raise KeyError(f"object keypoint file missing cartesian: {keypoint_path}")
    points = np.asarray(loaded["cartesian"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"object keypoints must have shape [N,3]: {keypoint_path}")
    return points
