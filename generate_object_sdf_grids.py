from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

import h5py
import numpy as np
import trimesh
from tqdm import tqdm


def collect_objects_from_hdf5(hdf5_paths: Sequence[Path]) -> List[str]:
    objects: Set[str] = set()
    for hdf5_path in hdf5_paths:
        if not hdf5_path.exists():
            continue
        with h5py.File(hdf5_path, "r") as h5_file:
            for subject_key in h5_file.keys():
                for sequence_key in h5_file[subject_key].keys():
                    objects.add(str(sequence_key).split("_")[0])
    return sorted(objects)


def iter_grid_points(bounds_min: np.ndarray, bounds_max: np.ndarray, resolution: int) -> np.ndarray:
    xs = np.linspace(bounds_min[0], bounds_max[0], resolution, dtype=np.float32)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution, dtype=np.float32)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution, dtype=np.float32)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def signed_distance_chunks(mesh: trimesh.Trimesh, points: np.ndarray, chunk_size: int) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, int(points.shape[0]), chunk_size):
        end = min(start + chunk_size, int(points.shape[0]))
        signed_distance = trimesh.proximity.signed_distance(mesh, points[start:end])
        outputs.append((-signed_distance).astype(np.float32))
    return np.concatenate(outputs, axis=0)


def generate_sdf_for_mesh(
    mesh_path: Path,
    output_path: Path,
    resolution: int,
    padding_ratio: float,
    min_padding: float,
    chunk_size: int,
    overwrite: bool,
) -> Dict[str, object]:
    if output_path.exists() and not overwrite:
        with np.load(output_path) as loaded:
            shape = tuple(int(value) for value in loaded["sdf"].shape)
        return {"object": output_path.stem, "status": "skipped_existing", "sdf_shape": shape}
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Failed to load mesh: {mesh_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no vertices or faces: {mesh_path}")
    mesh.remove_unreferenced_vertices()
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    extent = bounds[1] - bounds[0]
    padding = max(float(np.max(extent)) * float(padding_ratio), float(min_padding))
    bounds_min = (bounds[0] - padding).astype(np.float32)
    bounds_max = (bounds[1] + padding).astype(np.float32)
    points = iter_grid_points(bounds_min, bounds_max, int(resolution))
    sdf = signed_distance_chunks(mesh, points, int(chunk_size)).reshape(
        int(resolution), int(resolution), int(resolution)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sdf=sdf.astype(np.float32),
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        resolution=np.asarray([int(resolution)], dtype=np.int32),
        mesh_watertight=np.asarray([bool(mesh.is_watertight)], dtype=bool),
    )
    return {"object": output_path.stem, "status": "generated", "sdf_shape": tuple(int(value) for value in sdf.shape)}


def dataset_specs(data_root: Path) -> List[Dict[str, object]]:
    return [
        {
            "name": "behave",
            "root": data_root / "behave_smplh_ground",
            "hdf5_files": [
                data_root / "behave_smplh_ground" / "dataset_train_10fps.hdf5",
                data_root / "behave_smplh_ground" / "dataset_test_1fps.hdf5",
            ],
        },
        {
            "name": "grab",
            "root": data_root / "grab_smplh_ground",
            "hdf5_files": [
                data_root / "grab_smplh_ground" / "dataset_train_10fps.hdf5",
                data_root / "grab_smplh_ground" / "dataset_test_1fps.hdf5",
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object SDF grids")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    parser.add_argument("--min-padding", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["behave", "grab"], choices=["behave", "grab"])
    args = parser.parse_args()
    if args.resolution < 8:
        raise ValueError("--resolution must be at least 8")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    selected_datasets = set(args.datasets)
    total_reports: List[Dict[str, object]] = []
    for spec in dataset_specs(args.data_root):
        dataset_name = str(spec["name"])
        if dataset_name not in selected_datasets:
            continue
        dataset_root = Path(spec["root"])
        mesh_dir = dataset_root / "object_meshes"
        output_dir = dataset_root / "object_sdf"
        hdf5_files = [Path(path) for path in spec["hdf5_files"]]
        objects = collect_objects_from_hdf5(hdf5_files)
        if not objects:
            raise RuntimeError(f"No objects found for {dataset_name}")
        reports: List[Dict[str, object]] = []
        for object_name in tqdm(objects, desc=f"{dataset_name} sdf"):
            mesh_path = mesh_dir / f"{object_name}.ply"
            if not mesh_path.exists():
                raise FileNotFoundError(f"Missing object mesh: {mesh_path}")
            reports.append(
                generate_sdf_for_mesh(
                    mesh_path,
                    output_dir / f"{object_name}.npz",
                    int(args.resolution),
                    float(args.padding_ratio),
                    float(args.min_padding),
                    int(args.chunk_size),
                    bool(args.overwrite),
                )
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "sdf_generation_report.json").open("w", encoding="utf-8") as file_pointer:
            json.dump({"dataset": dataset_name, "objects": objects, "reports": reports}, file_pointer, indent=2)
        total_reports.extend(reports)
        print(f"[done] {dataset_name}: wrote {len(reports)} SDF files")
    print(f"[done] total objects processed: {len(total_reports)}")


if __name__ == "__main__":
    main()
