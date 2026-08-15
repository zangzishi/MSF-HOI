import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

MODE_TO_TARGET: Dict[Tuple[int, int, int], str] = {
    (1, 0, 0): "sbj",
    (0, 1, 0): "obj",
    (1, 1, 0): "sbj_obj",
    (1, 0, 1): "sbj_contact",
    (0, 1, 1): "obj_contact",
    (1, 1, 1): "sbj_obj_contact",
}
METRIC_NAMES: Tuple[str, ...] = (
    "1-NNA",
    "COV",
    "MMD",
    "Div",
    "MMod",
    "MPJPE",
    "MPJPE_PA",
    "SBJ_CONTACT_MESHES",
    "SBJ_CONTACT_DIFFUSED",
    "OBJ_V2V",
    "OBJ_CENTER",
    "OBJ_CONTACT_MESHES",
    "OBJ_CONTACT_DIFFUSED",
    "P_sc",
    "PD",
    "Min_D",
    "SBJ_GP_%",
    "SBJ_MAD_gnd",
    "OBJ_GP_%",
    "OBJ_MAD_gnd",
    "d_bal",
)
CONTACT_METRICS = {
    "SBJ_CONTACT_MESHES",
    "SBJ_CONTACT_DIFFUSED",
    "OBJ_CONTACT_MESHES",
    "OBJ_CONTACT_DIFFUSED",
}
BEST_OF_MAX_METRICS = CONTACT_METRICS | {"SBJ_GP_%", "OBJ_GP_%"}
MEAN_METRICS = {"SBJ_MAD_gnd", "OBJ_MAD_gnd", "d_bal"}
PERCENTAGE_METRICS = {"SBJ_GP_%", "OBJ_GP_%"}
DIVERSITY_SAMPLE_SIZE = 200
GROUND_SUCCESS_EPS = 0.02
HUMAN_MASS_GRAMS = 70000.0
OBJECT_MASS_GRAMS: Dict[str, float] = {
    "backpack": 3000.0,
    "banana": 150.0,
    "basketball": 600.0,
    "binoculars": 800.0,
    "boxlarge": 8000.0,
    "boxlong": 4000.0,
    "boxmedium": 2500.0,
    "boxsmall": 800.0,
    "boxtiny": 100.0,
    "camera": 1000.0,
    "chairblack": 12000.0,
    "chairwood": 6000.0,
    "coffeemug": 350.0,
    "cup": 200.0,
    "doorknob": 300.0,
    "eyeglasses": 30.0,
    "flashlight": 250.0,
    "flute": 400.0,
    "fryingpan": 1200.0,
    "gamecontroller": 250.0,
    "hammer": 600.0,
    "headphones": 250.0,
    "keyboard": 800.0,
    "knife": 200.0,
    "lightbulb": 30.0,
    "monitor": 4500.0,
    "mouse": 100.0,
    "mug": 350.0,
    "phone": 200.0,
    "plasticcontainer": 150.0,
    "stool": 2500.0,
    "suitcase": 8000.0,
    "tablesmall": 8000.0,
    "tablesquare": 20000.0,
    "teapot": 1200.0,
    "toolbox": 5000.0,
    "toothbrush": 20.0,
    "trashbin": 1000.0,
    "wineglass": 150.0,
    "yogaball": 1000.0,
    "yogamat": 1200.0,
}
PROJECT_ROOT = Path(__file__).resolve().parent
OBJECT_CANONICAL_COM_PATH = PROJECT_ROOT / "assets/object_canonical_mass_properties.json"
_OBJECT_CANONICAL_COM: Optional[Dict[str, np.ndarray]] = None


def normalize_object_name(value: str) -> str:
    name = str(value).strip().lower()
    return "coffeemug" if name == "coffee_mug" else name


def object_mass_grams(value: str) -> float:
    name = normalize_object_name(value)
    if name not in OBJECT_MASS_GRAMS:
        raise KeyError(name)
    return float(OBJECT_MASS_GRAMS[name])


def load_object_canonical_com() -> Dict[str, np.ndarray]:
    global _OBJECT_CANONICAL_COM
    if _OBJECT_CANONICAL_COM is None:
        payload = json.loads(OBJECT_CANONICAL_COM_PATH.read_text(encoding="utf-8"))
        objects = payload["objects"]
        _OBJECT_CANONICAL_COM = {
            normalize_object_name(name): np.asarray(
                entry["canonical_center_of_mass"], dtype=np.float32
            ).reshape(3)
            for name, entry in objects.items()
        }
    return _OBJECT_CANONICAL_COM


def resolve_object_centers_of_mass(
    object_name: str,
    object_centers: np.ndarray,
    object_rotations: np.ndarray,
) -> np.ndarray:
    canonical = load_object_canonical_com()[normalize_object_name(object_name)]
    rotated = canonical.reshape(1, 1, 3) @ np.transpose(object_rotations, (0, 2, 1))
    return (rotated.reshape(-1, 3) + object_centers).astype(np.float32)


def sequence_name_to_triplet(sequence_name: str) -> Tuple[str, str]:
    chunks = sequence_name.split("_")
    return chunks[0], "_".join(chunks[1:])


def iter_hdf5_sequences(
    hdf5_file: h5py.File,
    max_sequences: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    result: List[Tuple[str, str, str]] = []
    for subject in sorted(hdf5_file.keys()):
        for sequence_name in sorted(hdf5_file[subject].keys()):
            obj, act = sequence_name_to_triplet(sequence_name)
            result.append((subject, obj, act))
            if max_sequences is not None and len(result) >= max_sequences:
                return result
    return result


def feature_dim(feature_name: str) -> int:
    if feature_name == "human_joints":
        return 71 * 3
    if feature_name == "object_pose":
        return 12
    if feature_name == "human_joints_object_pose":
        return 71 * 3 + 12
    raise ValueError(feature_name)


def extract_sequence_features(
    sequence_group: h5py.Group,
    feature_name: str,
) -> np.ndarray:
    frame_count = int(sequence_group.attrs.get("T", sequence_group["sbj_j"].shape[0]))
    if feature_name == "human_joints":
        joints = np.asarray(sequence_group["sbj_j"][:frame_count], dtype=np.float32)
        return (joints - joints[:, [0]])[:, 1:].reshape(frame_count, -1)
    if feature_name == "object_pose":
        centers = np.asarray(sequence_group["obj_c"][:frame_count], dtype=np.float32)
        rotations = np.asarray(sequence_group["obj_R"][:frame_count], dtype=np.float32).reshape(frame_count, 9)
        return np.concatenate([rotations, centers], axis=1)
    if feature_name == "human_joints_object_pose":
        joints = np.asarray(sequence_group["sbj_j"][:frame_count], dtype=np.float32)
        human = (joints - joints[:, [0]])[:, 1:].reshape(frame_count, -1)
        centers = np.asarray(sequence_group["obj_c"][:frame_count], dtype=np.float32)
        rotations = np.asarray(sequence_group["obj_R"][:frame_count], dtype=np.float32).reshape(frame_count, 9)
        return np.concatenate([human, rotations, centers], axis=1)
    raise ValueError(feature_name)


def load_features_from_hdf5(
    hdf5_path: Path,
    feature_name: str,
    max_sequences: Optional[int] = None,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for subject, obj, act in iter_hdf5_sequences(handle, max_sequences):
            values = extract_sequence_features(handle[subject][f"{obj}_{act}"], feature_name)
            if values.shape[0]:
                chunks.append(values)
    if not chunks:
        return np.zeros((0, feature_dim(feature_name)), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def load_features_with_object_labels_from_hdf5(
    hdf5_path: Path,
    feature_name: str,
    max_sequences: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    chunks: List[np.ndarray] = []
    labels: List[str] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for subject, obj, act in iter_hdf5_sequences(handle, max_sequences):
            values = extract_sequence_features(handle[subject][f"{obj}_{act}"], feature_name)
            if values.shape[0]:
                chunks.append(values)
                labels.extend([obj] * int(values.shape[0]))
    if not chunks:
        return np.zeros((0, feature_dim(feature_name)), dtype=np.float32), np.asarray([], dtype=object)
    return np.concatenate(chunks, axis=0).astype(np.float32), np.asarray(labels, dtype=object)


def diversity_score(
    features: np.ndarray,
    sample_size: int = DIVERSITY_SAMPLE_SIZE,
    seed: int = 0,
) -> float:
    if features.shape[0] < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    replace = features.shape[0] < sample_size
    size = sample_size if replace else min(sample_size, features.shape[0])
    first = rng.choice(features.shape[0], size=size, replace=replace)
    second = rng.choice(features.shape[0], size=size, replace=replace)
    return float(np.linalg.norm(features[first] - features[second], axis=1).mean())


def multimodality_score(
    features: np.ndarray,
    object_labels: np.ndarray,
    sample_size: int = DIVERSITY_SAMPLE_SIZE,
    seed: int = 0,
) -> float:
    if features.shape[0] < 2 or object_labels.shape[0] != features.shape[0]:
        return float("nan")
    rng = np.random.default_rng(seed)
    normalized = np.asarray([normalize_object_name(str(value)) for value in object_labels], dtype=object)
    scores: List[float] = []
    for object_name in sorted(set(normalized.tolist())):
        values = features[normalized == object_name]
        if values.shape[0] < 2:
            continue
        replace = values.shape[0] < sample_size
        size = sample_size if replace else min(sample_size, values.shape[0])
        first = rng.choice(values.shape[0], size=size, replace=replace)
        second = rng.choice(values.shape[0], size=size, replace=replace)
        scores.append(float(np.linalg.norm(values[first] - values[second], axis=1).mean()))
    return float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else float("nan")


def load_ground_z_array(dataset: h5py.Dataset, frame_count: int) -> np.ndarray:
    values = dataset[()]
    if np.isscalar(values):
        return np.full((frame_count,), float(values), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] == frame_count:
        return values
    if values.shape[0] == 1:
        return np.full((frame_count,), float(values[0]), dtype=np.float32)
    limit = min(frame_count, values.shape[0])
    result = np.empty((frame_count,), dtype=np.float32)
    result[:limit] = values[:limit]
    result[limit:] = float(values[limit - 1])
    return result


def resolve_sample_frame_indices(
    sampled_sequence: h5py.Group,
    reference_frame_count: int,
    sampled_frame_count: int,
) -> np.ndarray:
    if "frame_indices" in sampled_sequence:
        indices = np.asarray(sampled_sequence["frame_indices"][:], dtype=np.int64).reshape(-1)
    else:
        indices = np.arange(sampled_frame_count, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < reference_frame_count)]
    return indices[: min(sampled_frame_count, indices.shape[0])].astype(np.int64)


def load_object_sdf_entry(
    data_root: Path,
    dataset_name: str,
    object_name: str,
    cache: Dict[Tuple[str, str], Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    key = (dataset_name, normalize_object_name(object_name))
    if key not in cache:
        sdf_path = data_root / f"{dataset_name}_smplh_ground" / "object_sdf" / f"{key[1]}.npz"
        loaded = np.load(sdf_path)
        cache[key] = {
            "sdf": np.asarray(loaded["sdf"], dtype=np.float32),
            "bounds_min": np.asarray(loaded["bounds_min"], dtype=np.float32).reshape(3),
            "bounds_max": np.asarray(loaded["bounds_max"], dtype=np.float32).reshape(3),
        }
    return cache[key]


def sample_sdf_values(
    subject_vertices: np.ndarray,
    object_centers: np.ndarray,
    object_rotations: np.ndarray,
    sdf_entry: Mapping[str, np.ndarray],
    device: torch.device,
    frame_batch_size: int,
    vertex_chunk_size: int,
) -> np.ndarray:
    output = np.zeros((subject_vertices.shape[0], subject_vertices.shape[1]), dtype=np.float32)
    sdf_grid_base = torch.from_numpy(sdf_entry["sdf"][None, None, ...])
    bounds_min = torch.from_numpy(sdf_entry["bounds_min"]).view(1, 1, 3)
    bounds_max = torch.from_numpy(sdf_entry["bounds_max"]).view(1, 1, 3)
    for frame_start in range(0, subject_vertices.shape[0], frame_batch_size):
        frame_end = min(subject_vertices.shape[0], frame_start + frame_batch_size)
        vertices = torch.from_numpy(subject_vertices[frame_start:frame_end]).to(device=device, dtype=torch.float32)
        centers = torch.from_numpy(object_centers[frame_start:frame_end]).to(device=device, dtype=torch.float32)
        rotations = torch.from_numpy(object_rotations[frame_start:frame_end]).to(device=device, dtype=torch.float32)
        batch = int(vertices.shape[0])
        grid = sdf_grid_base.to(device=device, dtype=torch.float32).expand(batch, -1, -1, -1, -1)
        extent = torch.clamp(
            (bounds_max - bounds_min).to(device=device, dtype=torch.float32),
            min=1e-6,
        )
        chunks: List[np.ndarray] = []
        for start in range(0, vertices.shape[1], vertex_chunk_size):
            end = min(vertices.shape[1], start + vertex_chunk_size)
            local = torch.bmm(
                vertices[:, start:end] - centers.unsqueeze(1),
                rotations,
            )
            normalized = 2.0 * (
                (local - bounds_min.to(device=device, dtype=torch.float32)) / extent
            ) - 1.0
            sample_grid = normalized.view(batch, end - start, 1, 1, 3)
            sampled = F.grid_sample(
                grid,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(batch, end - start)
            chunks.append(sampled.detach().cpu().numpy())
        output[frame_start:frame_end] = np.concatenate(chunks, axis=1).astype(np.float32)
    return output


def compute_balance_gt_centroid_xy_distance(
    sbj_vertices: np.ndarray,
    gt_sbj_vertices: np.ndarray,
    obj_centers: np.ndarray,
    obj_rotations: np.ndarray,
    gt_obj_centers: np.ndarray,
    gt_obj_rotations: np.ndarray,
    object_name: str,
) -> np.ndarray:
    mass = object_mass_grams(object_name)
    sampled_object_com = resolve_object_centers_of_mass(object_name, obj_centers, obj_rotations)
    gt_object_com = resolve_object_centers_of_mass(object_name, gt_obj_centers, gt_obj_rotations)
    sampled_human_com = np.asarray(sbj_vertices, dtype=np.float32).mean(axis=1)
    gt_human_com = np.asarray(gt_sbj_vertices, dtype=np.float32).mean(axis=1)
    sampled_system_com = (
        HUMAN_MASS_GRAMS * sampled_human_com + mass * sampled_object_com
    ) / (HUMAN_MASS_GRAMS + mass)
    gt_system_com = (
        HUMAN_MASS_GRAMS * gt_human_com + mass * gt_object_com
    ) / (HUMAN_MASS_GRAMS + mass)
    return np.linalg.norm(sampled_system_com[:, :2] - gt_system_com[:, :2], axis=1).astype(np.float32)


def get_spatial_physics_metrics(
    dataset_name: str,
    data_root: Path,
    reference_hdf5: Path,
    samples_file: Path,
    sample_target: str,
    device: torch.device,
    contact_threshold: float,
    contact_indexes: np.ndarray,
    sdf_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    max_sequences: Optional[int],
    sdf_frame_batch_size: int,
    sdf_vertex_chunk_size: int,
) -> Dict[str, np.ndarray]:
    metrics: Dict[str, List[np.ndarray]] = defaultdict(list)
    with h5py.File(str(reference_hdf5), "r") as reference_file, h5py.File(str(samples_file), "r") as samples_file_handle:
        sequences = iter_hdf5_sequences(reference_file, max_sequences)
        for subject, object_name, action in sequences:
            sequence_name = f"{object_name}_{action}"
            reference_sequence = reference_file[subject][sequence_name]
            sampled_sequence = samples_file_handle[subject][sequence_name]
            reference_count = min(
                int(reference_sequence.attrs.get("T", reference_sequence["obj_v"].shape[0])),
                int(reference_sequence["obj_v"].shape[0]),
                int(reference_sequence["sbj_v"].shape[0]),
            )
            sampled_count = min(
                int(sampled_sequence["obj_v"].shape[0]),
                int(sampled_sequence["sbj_v"].shape[0]),
                int(sampled_sequence["obj_c"].shape[0]),
                int(sampled_sequence["obj_R"].shape[0]),
            )
            frame_indices = resolve_sample_frame_indices(sampled_sequence, reference_count, sampled_count)
            total_frames = int(frame_indices.shape[0])
            if total_frames == 0:
                continue
            sampled_sbj_v = np.asarray(sampled_sequence["sbj_v"][:total_frames], dtype=np.float32)
            sampled_obj_v = np.asarray(sampled_sequence["obj_v"][:total_frames], dtype=np.float32)
            sampled_obj_c = np.asarray(sampled_sequence["obj_c"][:total_frames], dtype=np.float32)
            sampled_obj_r = np.asarray(sampled_sequence["obj_R"][:total_frames], dtype=np.float32).reshape(total_frames, 3, 3)
            reference_sbj_v = np.asarray(reference_sequence["sbj_v"][frame_indices], dtype=np.float32)
            reference_obj_c = np.asarray(reference_sequence["obj_c"][frame_indices], dtype=np.float32)
            reference_obj_r = np.asarray(reference_sequence["obj_R"][frame_indices], dtype=np.float32).reshape(total_frames, 3, 3)
            ground_z = None
            if "ground_z" in reference_sequence:
                ground_z = load_ground_z_array(reference_sequence["ground_z"], reference_count)[frame_indices]
            if "sbj" in sample_target or "obj" in sample_target:
                metrics["d_bal"].append(
                    compute_balance_gt_centroid_xy_distance(
                        sampled_sbj_v,
                        reference_sbj_v,
                        sampled_obj_c,
                        sampled_obj_r,
                        reference_obj_c,
                        reference_obj_r,
                        object_name,
                    )
                )
            if ground_z is not None and "sbj" in sample_target:
                sbj_delta = sampled_sbj_v[..., 2].min(axis=1) - ground_z
                metrics["SBJ_GP_%"].append(
                    (np.abs(sbj_delta) <= GROUND_SUCCESS_EPS).astype(np.float32)
                )
                metrics["SBJ_MAD_gnd"].append(
                    np.maximum(-sbj_delta, 0.0).astype(np.float32)
                )
            if ground_z is not None and "obj" in sample_target:
                obj_delta = sampled_obj_v[..., 2].min(axis=1) - ground_z
                metrics["OBJ_GP_%"].append(
                    (np.abs(obj_delta) <= GROUND_SUCCESS_EPS).astype(np.float32)
                )
                metrics["OBJ_MAD_gnd"].append(
                    np.maximum(-obj_delta, 0.0).astype(np.float32)
                )
            if "sbj" in sample_target or "obj" in sample_target:
                sdf_entry = load_object_sdf_entry(data_root, dataset_name, object_name, sdf_cache)
                sdf_values = sample_sdf_values(
                    sampled_sbj_v[:, contact_indexes],
                    sampled_obj_c,
                    sampled_obj_r,
                    sdf_entry,
                    device,
                    sdf_frame_batch_size,
                    sdf_vertex_chunk_size,
                )
                penetration = np.maximum(-sdf_values, 0.0)
                metrics["Min_D"].append(np.min(np.abs(sdf_values), axis=1).astype(np.float32))
                metrics["P_sc"].append(np.mean(penetration, axis=1).astype(np.float32))
                metrics["PD"].append(np.max(penetration, axis=1).astype(np.float32))
    return {
        name: np.concatenate(values, axis=0).astype(np.float32)
        for name, values in metrics.items()
        if values
    }


class NearestNeighborIndex:
    def __init__(self, train_features: np.ndarray) -> None:
        self.train_features = train_features.astype(np.float32)
        self.model = NearestNeighbors(algorithm="auto", metric="euclidean", n_jobs=-1)
        self.model.fit(self.train_features)

    def query(self, query_features: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if query_features.shape[0] == 0:
            return np.zeros((0, k), dtype=np.float32), np.zeros((0, k), dtype=np.int64)
        effective = max(1, min(k, self.train_features.shape[0]))
        distances, indices = self.model.kneighbors(
            query_features.astype(np.float32),
            n_neighbors=effective,
            return_distance=True,
        )
        distances = np.square(distances).astype(np.float32)
        if effective < k:
            count = k - effective
            distances = np.pad(distances, ((0, 0), (0, count)), mode="edge")
            indices = np.pad(indices, ((0, 0), (0, count)), mode="edge")
        return distances, indices


def nearest_neighbor_accuracy(reference_features: np.ndarray, sample_features: np.ndarray) -> float:
    if reference_features.shape[0] == 0 or sample_features.shape[0] == 0:
        return float("nan")
    features = np.concatenate([reference_features, sample_features], axis=0)
    labels = np.concatenate(
        [
            np.zeros(reference_features.shape[0], dtype=np.int64),
            np.ones(sample_features.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    index = NearestNeighborIndex(features)
    _, ref_indices = index.query(reference_features, 2)
    _, sample_indices = index.query(sample_features, 2)
    reference_hits = np.sum(labels[ref_indices[:, 1]] == 0)
    sample_hits = np.sum(labels[sample_indices[:, 1]] == 1)
    return float(reference_hits + sample_hits) / float(reference_features.shape[0] + sample_features.shape[0])


def coverage(reference_features: np.ndarray, sample_features: np.ndarray) -> float:
    if reference_features.shape[0] == 0 or sample_features.shape[0] == 0:
        return float("nan")
    index = NearestNeighborIndex(reference_features)
    _, indices = index.query(sample_features, 1)
    return float(np.unique(indices).shape[0]) / float(reference_features.shape[0])


def minimum_matching_distance(reference_features: np.ndarray, sample_features: np.ndarray) -> float:
    if reference_features.shape[0] == 0 or sample_features.shape[0] == 0:
        return float("nan")
    index = NearestNeighborIndex(sample_features)
    distances, _ = index.query(reference_features, 1)
    return float(np.sum(distances)) / float(reference_features.shape[0])


def compute_similarity_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    transposed = False
    if first.shape[0] not in (2, 3):
        first = first.T
        second = second.T
        transposed = True
    mean_first = first.mean(axis=1, keepdims=True)
    mean_second = second.mean(axis=1, keepdims=True)
    centered_first = first - mean_first
    centered_second = second - mean_second
    variance = np.sum(centered_first ** 2)
    cross = centered_first.dot(centered_second.T)
    left, _, right_transposed = np.linalg.svd(cross)
    right = right_transposed.T
    diagonal = np.eye(left.shape[0])
    diagonal[-1, -1] *= np.sign(np.linalg.det(left.dot(right.T)))
    rotation = right.dot(diagonal.dot(left.T))
    scale = np.trace(rotation.dot(cross)) / variance
    translation = mean_second - scale * rotation.dot(mean_first)
    aligned = scale * rotation.dot(first) + translation
    return aligned.T if transposed else aligned


def align_to_root(joints: np.ndarray, root_index: int = 0) -> np.ndarray:
    return joints - joints[..., root_index : root_index + 1, :]


def get_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = align_to_root(predicted)
    target = align_to_root(target)
    return float(np.sqrt(np.sum((target - predicted) ** 2, axis=-1)).mean(-1))


def get_mpjpe_pa(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = align_to_root(predicted)
    target = align_to_root(target)
    aligned = compute_similarity_transform(predicted, target)
    return float(np.sqrt(np.sum((target - aligned) ** 2, axis=-1)).mean(-1))


@torch.no_grad()
def contacts_worker_chunked(
    subject_vertices: torch.Tensor,
    object_vertices: torch.Tensor,
    contact_threshold: float,
    object_chunk_size: int,
) -> np.ndarray:
    minimum = torch.full(
        (subject_vertices.shape[0], subject_vertices.shape[1]),
        float("inf"),
        device=subject_vertices.device,
        dtype=subject_vertices.dtype,
    )
    for start in range(0, object_vertices.shape[1], object_chunk_size):
        end = min(start + object_chunk_size, object_vertices.shape[1])
        distances = torch.cdist(subject_vertices, object_vertices[:, start:end, :])
        minimum = torch.minimum(minimum, distances.min(dim=-1).values)
    return (minimum <= contact_threshold).cpu().numpy()


@torch.no_grad()
def compute_contacts_mask(
    subject_vertices: np.ndarray,
    object_vertices: np.ndarray,
    contact_threshold: float,
    device: torch.device,
    batch_size: int,
    object_chunk_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, subject_vertices.shape[0], batch_size):
        end = min(start + batch_size, subject_vertices.shape[0])
        subject_batch = torch.from_numpy(subject_vertices[start:end]).to(device=device, dtype=torch.float32)
        object_batch = torch.from_numpy(object_vertices[start:end]).to(device=device, dtype=torch.float32)
        outputs.append(
            contacts_worker_chunked(
                subject_batch,
                object_batch,
                contact_threshold,
                object_chunk_size,
            )
        )
    return np.concatenate(outputs, axis=0)


def compute_contact_accuracy(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    target = np.asarray(ground_truth, dtype=bool)
    estimate = np.asarray(prediction, dtype=bool)
    if target.shape != estimate.shape:
        raise ValueError(f"contact mask shape mismatch: {target.shape} {estimate.shape}")
    true_positive = np.logical_and(target, estimate).sum(axis=1).astype(np.float32)
    true_negative = np.logical_and(~target, ~estimate).sum(axis=1).astype(np.float32)
    return ((true_positive + true_negative) / float(max(target.shape[1], 1))).astype(np.float32)


def sampled_subject_contact_accuracy(
    sampled_subject: np.ndarray,
    reference_object: np.ndarray,
    reference_subject: np.ndarray,
    contact_threshold: float,
    device: torch.device,
    contact_batch_size: int,
    object_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ground_truth = compute_contacts_mask(
        reference_subject,
        reference_object,
        contact_threshold,
        device,
        contact_batch_size,
        object_chunk_size,
    )
    prediction = compute_contacts_mask(
        sampled_subject,
        reference_object,
        contact_threshold,
        device,
        contact_batch_size,
        object_chunk_size,
    )
    return ground_truth, compute_contact_accuracy(ground_truth, prediction)


def sampled_object_contact_accuracy(
    sampled_object: np.ndarray,
    reference_object: np.ndarray,
    reference_subject: np.ndarray,
    contact_threshold: float,
    device: torch.device,
    contact_batch_size: int,
    object_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ground_truth = compute_contacts_mask(
        reference_subject,
        reference_object,
        contact_threshold,
        device,
        contact_batch_size,
        object_chunk_size,
    )
    prediction = compute_contacts_mask(
        reference_subject,
        sampled_object,
        contact_threshold,
        device,
        contact_batch_size,
        object_chunk_size,
    )
    return ground_truth, compute_contact_accuracy(ground_truth, prediction)


def get_sbj_metrics(
    reference_hdf5: Path,
    samples_file: Path,
    contact_indexes: np.ndarray,
    contact_threshold: float,
    device: torch.device,
    contact_batch_size: int,
    object_chunk_size: int,
    max_sequences: Optional[int],
) -> Dict[str, np.ndarray]:
    values: Dict[str, List[np.ndarray]] = defaultdict(list)
    with h5py.File(str(reference_hdf5), "r") as reference_file, h5py.File(str(samples_file), "r") as samples_handle:
        for subject, object_name, action in iter_hdf5_sequences(reference_file, max_sequences):
            sequence_name = f"{object_name}_{action}"
            reference_sequence = reference_file[subject][sequence_name]
            sampled_sequence = samples_handle[subject][sequence_name]
            reference_count = min(
                int(reference_sequence.attrs.get("T", reference_sequence["sbj_j"].shape[0])),
                int(reference_sequence["sbj_j"].shape[0]),
                int(reference_sequence["obj_v"].shape[0]),
                int(reference_sequence["sbj_v"].shape[0]),
            )
            sampled_count = min(
                int(sampled_sequence["sbj_j"].shape[0]),
                int(sampled_sequence["sbj_v"].shape[0]),
            )
            frame_indices = resolve_sample_frame_indices(sampled_sequence, reference_count, sampled_count)
            total_frames = int(frame_indices.shape[0])
            if total_frames == 0:
                continue
            target_joints = np.asarray(reference_sequence["sbj_j"][frame_indices], dtype=np.float32)
            sampled_joints = np.asarray(sampled_sequence["sbj_j"][:total_frames], dtype=np.float32)
            values["MPJPE"].extend(
                get_mpjpe(sampled_joints[index], target_joints[index])
                for index in range(total_frames)
            )
            values["MPJPE_PA"].extend(
                get_mpjpe_pa(sampled_joints[index], target_joints[index])
                for index in range(total_frames)
            )
            reference_object = np.asarray(reference_sequence["obj_v"][frame_indices], dtype=np.float32)
            reference_subject = np.asarray(reference_sequence["sbj_v"][frame_indices], dtype=np.float32)
            sampled_subject = np.asarray(sampled_sequence["sbj_v"][:total_frames], dtype=np.float32)
            ground_truth, mesh_accuracy = sampled_subject_contact_accuracy(
                sampled_subject[:, contact_indexes],
                reference_object,
                reference_subject[:, contact_indexes],
                contact_threshold,
                device,
                contact_batch_size,
                object_chunk_size,
            )
            values["SBJ_CONTACT_MESHES"].append(mesh_accuracy)
            if "sbj_contact" in sampled_sequence:
                diffused = np.asarray(sampled_sequence["sbj_contact"][:total_frames], dtype=np.float32)
                values["SBJ_CONTACT_DIFFUSED"].append(
                    compute_contact_accuracy(
                        ground_truth,
                        diffused[:, contact_indexes] > 0.5,
                    )
                )
    result: Dict[str, np.ndarray] = {}
    for name in ("MPJPE", "MPJPE_PA", "SBJ_CONTACT_MESHES", "SBJ_CONTACT_DIFFUSED"):
        chunks = values.get(name, [])
        result[name] = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.asarray([0.0], dtype=np.float32)
    return result


def get_obj_metrics(
    reference_hdf5: Path,
    samples_file: Path,
    contact_indexes: np.ndarray,
    contact_threshold: float,
    device: torch.device,
    contact_batch_size: int,
    object_chunk_size: int,
    max_sequences: Optional[int],
) -> Dict[str, np.ndarray]:
    values: Dict[str, List[np.ndarray]] = defaultdict(list)
    with h5py.File(str(reference_hdf5), "r") as reference_file, h5py.File(str(samples_file), "r") as samples_handle:
        for subject, object_name, action in iter_hdf5_sequences(reference_file, max_sequences):
            sequence_name = f"{object_name}_{action}"
            reference_sequence = reference_file[subject][sequence_name]
            sampled_sequence = samples_handle[subject][sequence_name]
            reference_count = min(
                int(reference_sequence.attrs.get("T", reference_sequence["obj_v"].shape[0])),
                int(reference_sequence["obj_v"].shape[0]),
                int(reference_sequence["sbj_v"].shape[0]),
            )
            sampled_count = int(sampled_sequence["obj_v"].shape[0])
            frame_indices = resolve_sample_frame_indices(sampled_sequence, reference_count, sampled_count)
            total_frames = int(frame_indices.shape[0])
            if total_frames == 0:
                continue
            reference_object = np.asarray(reference_sequence["obj_v"][frame_indices], dtype=np.float32)
            reference_subject = np.asarray(reference_sequence["sbj_v"][frame_indices], dtype=np.float32)
            sampled_object = np.asarray(sampled_sequence["obj_v"][:total_frames], dtype=np.float32)
            values["OBJ_V2V"].append(np.linalg.norm(reference_object - sampled_object, axis=-1).mean(-1))
            values["OBJ_CENTER"].append(
                np.linalg.norm(reference_object.mean(axis=1) - sampled_object.mean(axis=1), axis=1)
            )
            ground_truth, mesh_accuracy = sampled_object_contact_accuracy(
                sampled_object,
                reference_object,
                reference_subject[:, contact_indexes],
                contact_threshold,
                device,
                contact_batch_size,
                object_chunk_size,
            )
            values["OBJ_CONTACT_MESHES"].append(mesh_accuracy)
            if "sbj_contact" in sampled_sequence:
                diffused = np.asarray(sampled_sequence["sbj_contact"][:total_frames], dtype=np.float32)
                values["OBJ_CONTACT_DIFFUSED"].append(
                    compute_contact_accuracy(
                        ground_truth,
                        diffused[:, contact_indexes] > 0.5,
                    )
                )
    result: Dict[str, np.ndarray] = {}
    for name in ("OBJ_V2V", "OBJ_CENTER", "OBJ_CONTACT_MESHES", "OBJ_CONTACT_DIFFUSED"):
        chunks = values.get(name, [])
        result[name] = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.asarray([0.0], dtype=np.float32)
    return result


def aggregate_reconstruction_metric(values: List[np.ndarray], metric_name: str) -> float:
    stacked = np.stack([np.asarray(value).reshape(-1) for value in values], axis=1)
    if metric_name in BEST_OF_MAX_METRICS:
        reduced = np.nanmax(stacked, axis=1)
    elif metric_name in MEAN_METRICS:
        reduced = np.nanmean(stacked, axis=1)
    else:
        reduced = np.nanmin(stacked, axis=1)
    if metric_name in PERCENTAGE_METRICS:
        return float(np.nanmean(reduced) * 100.0)
    return float(np.nanmean(reduced))


def feature_name_for_target(target: str) -> str:
    if "sbj_obj" in target:
        return "human_joints_object_pose"
    if "sbj" in target:
        return "human_joints"
    if "obj" in target:
        return "object_pose"
    raise ValueError(target)


def discover_sample_files(sample_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    discovered: Dict[str, Dict[str, List[Path]]] = {}
    for dataset_dir in sorted(sample_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        target_map: Dict[str, List[Path]] = {}
        for target_dir in sorted(dataset_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            files = sorted(target_dir.glob("samples_rep_*.hdf5"))
            if files:
                target_map[target_dir.name] = files
        if target_map:
            discovered[dataset_dir.name] = target_map
    return discovered


def resolve_reference_hdf5(
    data_root: Path,
    dataset_name: str,
    split: str,
    fps: int,
) -> Path:
    return data_root / f"{dataset_name}_smplh_ground" / f"dataset_{split}_{fps}fps.hdf5"


def finite_float(value: float) -> Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MSFHOI samples")
    parser.add_argument("--sample-root", type=str, default="experiments")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--sampling-targets", nargs="*", default=None)
    parser.add_argument("--train-fps", type=int, default=10)
    parser.add_argument("--eval-fps", type=int, default=1)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--contact-batch-size", type=int, default=50)
    parser.add_argument("--obj-chunk-size", type=int, default=4096)
    parser.add_argument("--sdf-frame-batch-size", type=int, default=16)
    parser.add_argument("--sdf-vertex-chunk-size", type=int, default=2048)
    parser.add_argument("--contact-index-file", type=str, default="data/smpl_template_decimated_idxs.npy")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    sample_root = Path(args.sample_root)
    data_root = Path(args.data_root)
    discovered = discover_sample_files(sample_root)
    if args.datasets is not None:
        allowed = set(args.datasets)
        discovered = {
            dataset: targets
            for dataset, targets in discovered.items()
            if dataset in allowed
        }
    contact_indexes = np.asarray(np.load(Path(args.contact_index_file)), dtype=np.int64).reshape(-1)
    device = choose_device(args.device)
    output_path = Path(args.output_json) if args.output_json else sample_root / "evaluation_metrics.json"
    feature_cache: Dict[Tuple[str, str], np.ndarray] = {}
    labeled_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    sdf_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    results: Dict[str, object] = {
        "sample_root": str(sample_root),
        "data_root": str(data_root),
        "metrics": list(METRIC_NAMES),
        "datasets": {},
    }

    for dataset_name, targets in sorted(discovered.items()):
        selected_targets = sorted(targets.keys())
        if args.sampling_targets is not None:
            selected_targets = [name for name in selected_targets if name in set(args.sampling_targets)]
        if not selected_targets:
            continue
        reference_test = resolve_reference_hdf5(data_root, dataset_name, "test", args.eval_fps)
        reference_train = resolve_reference_hdf5(data_root, dataset_name, "train", args.train_fps)
        dataset_result: Dict[str, object] = {}
        for target in selected_targets:
            files = targets[target]
            target_result: Dict[str, object] = {}
            if target != "contact":
                feature_name = feature_name_for_target(target)
                test_key = (str(reference_test), feature_name)
                train_key = (str(reference_train), feature_name)
                if test_key not in feature_cache:
                    feature_cache[test_key] = load_features_from_hdf5(
                        reference_test,
                        feature_name,
                        args.max_sequences,
                    )
                if train_key not in feature_cache:
                    feature_cache[train_key] = load_features_from_hdf5(
                        reference_train,
                        feature_name,
                        args.max_sequences,
                    )
                generation: Dict[str, List[float]] = {
                    "1-NNA": [],
                    "COV": [],
                    "MMD": [],
                    "Div": [],
                    "MMod": [],
                }
                for sample_file in files:
                    key = (str(sample_file), feature_name)
                    if key not in labeled_cache:
                        labeled_cache[key] = load_features_with_object_labels_from_hdf5(
                            sample_file,
                            feature_name,
                            args.max_sequences,
                        )
                    sample_features, labels = labeled_cache[key]
                    generation["1-NNA"].append(
                        nearest_neighbor_accuracy(feature_cache[test_key], sample_features)
                    )
                    generation["COV"].append(coverage(feature_cache[test_key], sample_features))
                    generation["MMD"].append(
                        minimum_matching_distance(feature_cache[test_key], sample_features)
                    )
                    generation["Div"].append(diversity_score(sample_features))
                    generation["MMod"].append(multimodality_score(sample_features, labels))
                target_result["generation"] = {
                    name: finite_float(float(np.nanmean(values))) if values else None
                    for name, values in generation.items()
                }

            reconstruction: Dict[str, List[np.ndarray]] = {name: [] for name in METRIC_NAMES[5:]}
            for sample_file in files:
                if "sbj" in target:
                    for name, values in get_sbj_metrics(
                        reference_test,
                        sample_file,
                        contact_indexes,
                        args.contact_threshold,
                        device,
                        args.contact_batch_size,
                        args.obj_chunk_size,
                        args.max_sequences,
                    ).items():
                        reconstruction[name].append(values)
                if "obj" in target:
                    for name, values in get_obj_metrics(
                        reference_test,
                        sample_file,
                        contact_indexes,
                        args.contact_threshold,
                        device,
                        args.contact_batch_size,
                        args.obj_chunk_size,
                        args.max_sequences,
                    ).items():
                        reconstruction[name].append(values)
                if "sbj" in target or "obj" in target:
                    for name, values in get_spatial_physics_metrics(
                        dataset_name,
                        data_root,
                        reference_test,
                        sample_file,
                        target,
                        device,
                        args.contact_threshold,
                        contact_indexes,
                        sdf_cache,
                        args.max_sequences,
                        args.sdf_frame_batch_size,
                        args.sdf_vertex_chunk_size,
                    ).items():
                        reconstruction[name].append(values)
            target_result["reconstruction"] = {
                name: finite_float(aggregate_reconstruction_metric(values, name))
                for name, values in reconstruction.items()
                if values
            }
            dataset_result[target] = target_result
        results["datasets"][dataset_name] = dataset_result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()



