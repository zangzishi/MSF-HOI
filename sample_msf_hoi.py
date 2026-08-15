from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from core.pipeline import MSFHOIPipeline
from core.object_keypoints import load_object_keypoints
from models.contact_weights import load_contact_weights
from models import resolve_MSFHOI_model_kwargs_from_config
from models.transformer import MSFHOIFlowModel
from solver import MSFHOIODESolver, MSFHOITimestepScheduler, normalize_MSFHOI_mode

try:
    import smplx
except Exception:
    smplx = None

OBJECT_CLASS_NAMES: List[str] = [
    "backpack", "banana", "basketball", "binoculars", "boxlarge", "boxlong", "boxmedium", "boxsmall", "boxtiny", "camera",
    "chairblack", "chairwood", "cup", "doorknob", "eyeglasses", "flashlight", "flute", "fryingpan", "gamecontroller", "hammer",
    "headphones", "keyboard", "knife", "lightbulb", "monitor", "mouse", "mug", "phone", "plasticcontainer", "stool", "suitcase",
    "tablesmall", "tablesquare", "teapot", "toolbox", "toothbrush", "trashbin", "wineglass", "yogaball", "yogamat",
]
OBJECT_CLASS_TO_INDEX: Dict[str, int] = {name: index for index, name in enumerate(OBJECT_CLASS_NAMES)}
PROJECT_ROOT = Path(__file__).resolve().parent
ROTATION_6D_ENCODING = "first_two_rows_v1"
def matrix9_to_rot6d(matrix9: np.ndarray) -> np.ndarray:
    matrix = matrix9.reshape(*matrix9.shape[:-1], 3, 3)
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))


def matrix9_to_axis_angle(matrix9: np.ndarray) -> np.ndarray:
    matrix = matrix9.reshape(-1, 3, 3)
    axis_angle = Rotation.from_matrix(matrix).as_rotvec().astype(np.float32)
    return axis_angle.reshape(*matrix9.shape[:-1], 3)


def rotation6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    projection = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - projection * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def build_smplh_model_cache(
    smpl_model_folder: str,
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    if smplx is None:
        raise ImportError("Package `smplx` is required to export sampled sbj_v/sbj_j.")

    cache: Dict[str, torch.nn.Module] = {}
    for gender in ("male", "female"):
        model = smplx.create(
            str(smpl_model_folder),
            model_type="smplh",
            gender=gender,
            use_pca=False,
            num_betas=10,
            flat_hand_mean=True,
            ext="pkl",
        )
        model = model.to(device=device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        cache[gender] = model
    return cache


def parse_gender_value(raw_gender: object) -> str:
    if isinstance(raw_gender, (bytes, bytearray)):
        normalized = raw_gender.decode("utf-8", errors="ignore").strip().lower()
    else:
        normalized = str(raw_gender).strip().lower()
        if normalized.startswith("b'") and normalized.endswith("'") and len(normalized) > 3:
            normalized = normalized[2:-1]

    if normalized.startswith("f"):
        return "female"
    return "male"


def mode_to_output_folder_name(mode: Tuple[int, int, int]) -> str:
    mapping = {
        (0, 1, 1): "obj_contact",
        (1, 0, 1): "sbj_contact",
        (1, 1, 0): "sbj_obj",
        (1, 1, 1): "sbj_obj_contact",
        (1, 0, 0): "sbj",
        (0, 1, 0): "obj",
        (0, 0, 1): "contact",
    }
    return mapping.get(mode, "sample")


def mode_to_code(mode: Tuple[int, int, int]) -> str:
    return "".join(str(int(value)) for value in mode)


def resolve_sampling_output_base(output_root: Path, mode: Tuple[int, int, int]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    max_index = 0
    index_width = 3
    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue

        matched = re.match(r"^(\d+)_([01]{3})$", entry.name)
        if matched is None:
            continue

        prefix = matched.group(1)
        max_index = max(max_index, int(prefix))
        index_width = max(index_width, len(prefix))

    next_index = max_index + 1
    index_width = max(index_width, len(str(next_index)))
    folder_name = f"{next_index:0{index_width}d}_{mode_to_code(mode)}"
    return output_root / folder_name


def infer_dataset_name(hdf5_path: str) -> str:
    lower_path = hdf5_path.lower()
    if "behave" in lower_path:
        return "behave"
    if "grab" in lower_path:
        return "grab"
    return Path(hdf5_path).stem


def load_pointnext_features(pointnext_files: Sequence[str]) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}
    for value in pointnext_files:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"PointNeXt file not found: {path}")
        with path.open("rb") as file_pointer:
            loaded = pickle.load(file_pointer)
        if not isinstance(loaded, dict):
            raise TypeError(f"PointNeXt file must contain a mapping: {path}")
        for object_name, feature in loaded.items():
            array = np.asarray(feature, dtype=np.float32).reshape(-1)
            if array.shape != (1024,):
                raise ValueError(f"PointNeXt feature must have 1024 values: {path}")
            features[str(object_name).lower()] = array
    return features

def load_object_sdf_map(source_hdf5_path: str, object_names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for object_name in sorted(set(str(value).lower() for value in object_names)):
        sdf_path = Path(source_hdf5_path).parent / "object_sdf" / f"{object_name}.npz"
        if not sdf_path.exists():
            raise FileNotFoundError(f"object SDF file not found: {sdf_path}")
        loaded = np.load(sdf_path)
        if set(loaded.files) != {"sdf", "bounds_min", "bounds_max"}:
            raise KeyError(f"object SDF requires exactly sdf,bounds_min,bounds_max: {sdf_path}")
        sdf = np.asarray(loaded["sdf"], dtype=np.float32)
        bounds_min = np.asarray(loaded["bounds_min"], dtype=np.float32).reshape(-1)
        bounds_max = np.asarray(loaded["bounds_max"], dtype=np.float32).reshape(-1)
        if sdf.ndim != 3 or bounds_min.shape != (3,) or bounds_max.shape != (3,):
            raise ValueError(f"invalid object SDF shapes: {sdf_path}")
        result[object_name] = {"sdf": sdf, "bounds_min": bounds_min, "bounds_max": bounds_max}
    return result

def load_object_keypoints_map(source_hdf5_path: str, object_names: Sequence[str]) -> Dict[str, np.ndarray]:
    return {name: load_object_keypoints(source_hdf5_path, name) for name in sorted(set(str(value).lower() for value in object_names))}

def load_human_foot_vertex_indices(segmentation_file: str) -> np.ndarray:
    segmentation_path = PROJECT_ROOT / segmentation_file
    if not segmentation_path.exists():
        raise FileNotFoundError(f"segmentation_file not found: {segmentation_path}")
    with segmentation_path.open("rb") as file_pointer:
        loaded = pickle.load(file_pointer, encoding="latin1")
    if not isinstance(loaded, dict):
        raise TypeError(f"segmentation_file must contain a mapping: {segmentation_path}")
    tokens = ("leftFoot", "rightFoot", "leftToes", "rightToes", "left_foot", "right_foot", "left_toe", "right_toe")
    values = [np.asarray(value, dtype=np.int64).reshape(-1) for key, value in loaded.items() if any(token.lower() in str(key).lower() for token in tokens)]
    if not values:
        raise KeyError(f"foot segmentation fields missing: {segmentation_path}")
    result = np.unique(np.concatenate(values)).astype(np.int64)
    result = result[(result >= 0) & (result < 6890)]
    if result.size == 0:
        raise ValueError(f"foot segmentation fields are empty: {segmentation_path}")
    return result

def load_contact_vertex_indices(contact_index_file: str) -> np.ndarray:
    contact_path = PROJECT_ROOT / contact_index_file
    if not contact_path.exists():
        raise FileNotFoundError(f"contact index file not found: {contact_path}")
    values = np.asarray(np.load(contact_path), dtype=np.int64).reshape(-1)
    if values.shape != (690,) or np.unique(values).shape != (690,) or int(values.min()) < 0 or int(values.max()) >= 6890:
        raise ValueError(f"contact index file must contain 690 unique SMPL-H indices: {contact_path}")
    return values

def build_contact_map_sequence(sequence_group: h5py.Group, contact_vertex_indices: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    if "sbj_contact" not in sequence_group or "sbj_contact_z" not in sequence_group:
        raise KeyError("required contact fields sbj_contact and sbj_contact_z are missing")
    sbj_contact = np.asarray(sequence_group["sbj_contact"][:], dtype=np.float32)
    sbj_contact_z = np.asarray(sequence_group["sbj_contact_z"][:], dtype=np.float32)
    if sbj_contact.ndim != 2 or sbj_contact.shape[1] != 6890 or sbj_contact_z.ndim != 2 or sbj_contact_z.shape[1] != 128:
        raise ValueError("required contact fields have invalid shapes")
    return sbj_contact[np.asarray(frame_indices, dtype=np.int64)][:, contact_vertex_indices]

def build_human_feature_sequence(sequence_group: h5py.Group) -> np.ndarray:
    betas = np.asarray(sequence_group["sbj_smpl_betas"][:], dtype=np.float32)

    sbj_global = np.asarray(sequence_group["sbj_smpl_global"][:], dtype=np.float32)
    if sbj_global.ndim == 3 and sbj_global.shape[1] == 1:
        sbj_global = sbj_global[:, 0]
    global_6d = matrix9_to_rot6d(sbj_global)

    body_6d = matrix9_to_rot6d(np.asarray(sequence_group["sbj_smpl_body"][:], dtype=np.float32))
    left_6d = matrix9_to_rot6d(np.asarray(sequence_group["sbj_smpl_lh"][:], dtype=np.float32))
    right_6d = matrix9_to_rot6d(np.asarray(sequence_group["sbj_smpl_rh"][:], dtype=np.float32))
    pose_6d = np.concatenate([body_6d, left_6d, right_6d], axis=1).reshape(betas.shape[0], -1)

    transl = np.asarray(sequence_group["sbj_smpl_transl"][:], dtype=np.float32)
    return np.concatenate([betas, global_6d, pose_6d, transl], axis=1).astype(np.float32)


def build_object_feature_sequence(sequence_group: h5py.Group) -> np.ndarray:
    obj_rot_6d = matrix9_to_rot6d(np.asarray(sequence_group["obj_R"][:], dtype=np.float32))
    obj_c = np.asarray(sequence_group["obj_c"][:], dtype=np.float32)
    return np.concatenate([obj_rot_6d, obj_c], axis=1).astype(np.float32)


def build_human_pose9_sequence(sequence_group: h5py.Group) -> np.ndarray:
    if "sbj_smpl_pose" in sequence_group:
        pose9 = np.asarray(sequence_group["sbj_smpl_pose"][:], dtype=np.float32)
        if pose9.ndim == 3 and pose9.shape[1] == 52 and pose9.shape[2] == 9:
            return pose9

    sbj_global = np.asarray(sequence_group["sbj_smpl_global"][:], dtype=np.float32)
    if sbj_global.ndim == 3 and sbj_global.shape[1] == 1:
        sbj_global = sbj_global[:, 0]
    if sbj_global.ndim != 2 or sbj_global.shape[1] != 9:
        raise ValueError(
            f"Unexpected sbj_smpl_global shape: {sbj_global.shape}. Expected [T,9] or [T,1,9]."
        )

    body = np.asarray(sequence_group["sbj_smpl_body"][:], dtype=np.float32)
    left = np.asarray(sequence_group["sbj_smpl_lh"][:], dtype=np.float32)
    right = np.asarray(sequence_group["sbj_smpl_rh"][:], dtype=np.float32)

    if body.ndim != 3 or body.shape[1:] != (21, 9):
        raise ValueError(f"Unexpected sbj_smpl_body shape: {body.shape}. Expected [T,21,9].")
    if left.ndim != 3 or left.shape[1:] != (15, 9):
        raise ValueError(f"Unexpected sbj_smpl_lh shape: {left.shape}. Expected [T,15,9].")
    if right.ndim != 3 or right.shape[1:] != (15, 9):
        raise ValueError(f"Unexpected sbj_smpl_rh shape: {right.shape}. Expected [T,15,9].")

    return np.concatenate(
        [sbj_global[:, None, :], body, left, right],
        axis=1,
    ).astype(np.float32)


def convert_human_feature_to_outputs(human_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape_params = human_feature[:, :10]
    global_orient_6d = human_feature[:, 10:16].reshape(-1, 1, 6)
    body_pose_6d = human_feature[:, 16:322].reshape(-1, 51, 6)
    transl = human_feature[:, 322:325]

    pose_6d = torch.cat([global_orient_6d, body_pose_6d], dim=1)
    pose_9d = rotation6d_to_matrix(pose_6d).reshape(-1, 52, 9)
    return shape_params, transl, pose_9d


def convert_object_feature_to_outputs(object_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    obj_rot_6d = object_feature[:, :6].reshape(-1, 1, 6)
    obj_rot_9d = rotation6d_to_matrix(obj_rot_6d).reshape(-1, 9)
    obj_center = object_feature[:, 6:9]
    return obj_rot_9d, obj_center


def transform_object_vertices(
    source_obj_vertices: np.ndarray,
    source_obj_rotation_9d: np.ndarray,
    source_obj_center: np.ndarray,
    target_obj_rotation_9d: np.ndarray,
    target_obj_center: np.ndarray,
) -> np.ndarray:
    batch_size = source_obj_vertices.shape[0]
    transformed = np.zeros_like(source_obj_vertices, dtype=np.float32)

    for batch_index in range(batch_size):
        source_rotation = source_obj_rotation_9d[batch_index].reshape(3, 3).astype(np.float32)
        target_rotation = target_obj_rotation_9d[batch_index].reshape(3, 3).astype(np.float32)
        source_center = source_obj_center[batch_index].astype(np.float32)
        target_center = target_obj_center[batch_index].astype(np.float32)

        canonical_vertices = (source_obj_vertices[batch_index] - source_center.reshape(1, 3)) @ source_rotation
        transformed_vertices = canonical_vertices @ target_rotation.T + target_center.reshape(1, 3)
        transformed[batch_index] = transformed_vertices.astype(np.float32)

    return transformed


def select_sequence_frame_indices(
    *,
    frame_count: int,
    max_frames_per_sequence: Optional[int],
    strategy: str,
) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros((0,), dtype=np.int64)

    if max_frames_per_sequence is None or int(max_frames_per_sequence) <= 0:
        return np.arange(frame_count, dtype=np.int64)

    limit = min(frame_count, int(max_frames_per_sequence))
    if limit >= frame_count:
        return np.arange(frame_count, dtype=np.int64)

    if strategy != "uniform":
        raise ValueError(f"Unsupported frame-selection strategy: {strategy}")

    return np.linspace(0, frame_count - 1, num=limit, dtype=np.int64)


def initialize_output_structure(
    source_file: h5py.File,
    output_file: h5py.File,
    *,
    max_sequences: Optional[int] = None,
    max_frames_per_sequence: Optional[int] = None,
    frame_selection_strategy: str = "uniform",
) -> Tuple[List[Tuple[str, str]], int]:
    sequence_pairs: List[Tuple[str, str]] = []
    total_frames = 0

    for subject_key in source_file.keys():
        source_subject_group = source_file[subject_key]
        output_subject_group: Optional[h5py.Group] = None

        for sequence_key in source_subject_group.keys():
            if max_sequences is not None and len(sequence_pairs) >= int(max_sequences):
                return sequence_pairs, total_frames

            source_seq_group = source_subject_group[sequence_key]
            source_frame_count = int(source_seq_group["sbj_smpl_transl"].shape[0])
            frame_indices = select_sequence_frame_indices(
                frame_count=source_frame_count,
                max_frames_per_sequence=max_frames_per_sequence,
                strategy=frame_selection_strategy,
            )
            frame_count = int(frame_indices.shape[0])
            total_frames += frame_count

            if output_subject_group is None:
                output_subject_group = output_file.create_group(subject_key)
            output_seq_group = output_subject_group.create_group(sequence_key)
            output_seq_group.create_dataset("sbj_v", shape=(frame_count, 6890, 3), dtype=np.float32)
            output_seq_group.create_dataset("sbj_f", data=np.asarray(source_seq_group["sbj_f"][:]))
            output_seq_group.create_dataset(
                "obj_v",
                shape=(frame_count, source_seq_group["obj_v"].shape[1], source_seq_group["obj_v"].shape[2]),
                dtype=np.float32,
            )
            output_seq_group.create_dataset("obj_f", data=np.asarray(source_seq_group["obj_f"][:]))
            output_seq_group.create_dataset("sbj_contact_z", shape=(frame_count, 128), dtype=np.float32)
            output_seq_group.create_dataset("sbj_contact", shape=(frame_count, 6890), dtype=np.float32)
            output_seq_group.create_dataset("sbj_smpl_pose", shape=(frame_count, 52, 9), dtype=np.float32)
            output_seq_group.create_dataset("sbj_smpl_transl", shape=(frame_count, 3), dtype=np.float32)
            output_seq_group.create_dataset("sbj_smpl_betas", shape=(frame_count, 10), dtype=np.float32)
            output_seq_group.create_dataset("sbj_j", shape=(frame_count, 73, 3), dtype=np.float32)
            output_seq_group.create_dataset("obj_c", shape=(frame_count, 3), dtype=np.float32)
            output_seq_group.create_dataset("obj_R", shape=(frame_count, 9), dtype=np.float32)
            output_seq_group.create_dataset("frame_indices", data=frame_indices.astype(np.int64))
            output_seq_group.attrs["T"] = frame_count
            output_seq_group.attrs["source_T"] = source_frame_count
            output_seq_group.attrs["frame_selection_strategy"] = str(frame_selection_strategy)

            sequence_pairs.append((subject_key, sequence_key))

    return sequence_pairs, total_frames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample MSFHOI test sets with fixed global batches")
    parser.add_argument("--checkpoint", type=str, default="assets/msf_hoi.pt")
    parser.add_argument("--test-hdf5-files", nargs="+", default=["data/behave_smplh_ground/dataset_test_1fps.hdf5", "data/grab_smplh_ground/dataset_test_1fps.hdf5"])
    parser.add_argument("--pointnext-files", nargs="+", default=["data/behave_smplh_ground/object_pointnext.pkl", "data/grab_smplh_ground/object_pointnext.pkl"])
    parser.add_argument("--contact-index-file", type=str, default="data/smpl_template_decimated_idxs.npy")
    parser.add_argument("--segmentation-file", type=str, default="data/smpl_segmentation.pkl")
    parser.add_argument("--output-root", type=str, default="experiments")
    parser.add_argument("--mode", type=str, default="011")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--frame-selection-strategy", type=str, choices=["uniform"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-steps", type=int, default=32)
    parser.add_argument("--solver-schedule", type=str, choices=["linear"], default="linear")
    parser.add_argument("--smpl-model-folder", type=str, default="data/smplx_models")
    parser.add_argument("--contact-map-threshold", type=float, default=0.4)
    parser.add_argument("--contact-ae-path", type=str, default="assets/gb_contacts.pth")
    parser.add_argument("--geometry-guidance-weight", type=float, default=0.0)
    parser.add_argument("--sdf-guidance-weight", type=float, default=0.0)
    parser.add_argument("--balance-ground-guidance-weight", type=float, default=0.0)
    parser.add_argument("--sdf-penetration-weight", type=float, default=10.0)
    parser.add_argument("--sdf-contact-weight", type=float, default=2.0)
    parser.add_argument("--sdf-contact-margin", type=float, default=0.05)
    parser.add_argument("--balance-ground-inner-weight", type=float, default=1.0)
    parser.add_argument("--balance-com-weight", type=float, default=1.0)
    return parser
def sample_dataset(*, pipeline: MSFHOIPipeline, smplh_models: Dict[str, torch.nn.Module], source_hdf5_path: str, output_hdf5_path: Path, mode: Tuple[int, int, int], pointnext_features: Dict[str, np.ndarray], contact_vertex_indices: np.ndarray, foot_vertex_indices: np.ndarray, batch_size: int, seed: int, contact_map_threshold: float, geometry_guidance_weight: float, sdf_guidance_weight: float, balance_ground_guidance_weight: float, sdf_penetration_weight: float, sdf_contact_weight: float, sdf_contact_margin: float, balance_ground_inner_weight: float, balance_com_weight: float, max_frames_per_sequence: Optional[int], frame_selection_strategy: str, max_sequences: Optional[int]) -> None:
    if int(batch_size) != 1024:
        raise ValueError("batch-size must be exactly 1024")
    contact_index_tensor = torch.from_numpy(contact_vertex_indices).long()
    foot_index_tensor = torch.from_numpy(foot_vertex_indices).long()
    records: List[Dict[str, object]] = []
    with h5py.File(source_hdf5_path, "r") as source_file, h5py.File(str(output_hdf5_path), "w") as output_file:
        sequence_pairs, total_frames = initialize_output_structure(source_file, output_file, max_sequences=max_sequences, max_frames_per_sequence=max_frames_per_sequence, frame_selection_strategy=frame_selection_strategy)
        for subject_key, sequence_key in sequence_pairs:
            sequence_group = source_file[subject_key][sequence_key]
            selected = np.asarray(output_file[subject_key][sequence_key]["frame_indices"][:], dtype=np.int64)
            object_name = sequence_key.split("_")[0].lower()
            frame_count = int(selected.shape[0])
            human_all = build_human_feature_sequence(sequence_group)[selected]
            object_all = build_object_feature_sequence(sequence_group)[selected]
            source_vertices = np.asarray(sequence_group["sbj_v"][:], dtype=np.float32)[selected]
            source_joints = np.asarray(sequence_group["sbj_j"][:], dtype=np.float32)[selected]
            source_betas = np.asarray(sequence_group["sbj_smpl_betas"][:], dtype=np.float32)[selected]
            source_transl = np.asarray(sequence_group["sbj_smpl_transl"][:], dtype=np.float32)[selected]
            source_pose = build_human_pose9_sequence(sequence_group)[selected]
            source_object_vertices = np.asarray(sequence_group["obj_v"][:], dtype=np.float32)[selected]
            source_object_rotation = np.asarray(sequence_group["obj_R"][:], dtype=np.float32)[selected]
            source_object_center = np.asarray(sequence_group["obj_c"][:], dtype=np.float32)[selected]
            rotation_matrix = source_object_rotation.reshape(frame_count, 3, 3)
            canonical_vertices = (source_object_vertices - source_object_center[:, None, :]) @ rotation_matrix
            keypoints = load_object_keypoints_map(source_hdf5_path, [object_name])[object_name]
            canonical_keypoints = np.repeat(keypoints[None, :, :].astype(np.float32), frame_count, axis=0)
            if "ground_z" not in sequence_group:
                raise KeyError(f"required ground_z field missing: {subject_key}/{sequence_key}")
            ground_all = np.asarray(sequence_group["ground_z"][:], dtype=np.float32).reshape(-1)[selected]
            if "sbj_contact" not in sequence_group or "sbj_contact_z" not in sequence_group:
                raise KeyError(f"required contact fields missing: {subject_key}/{sequence_key}")
            source_contact = np.asarray(sequence_group["sbj_contact"][:], dtype=np.float32)[selected]
            source_contact_z = np.asarray(sequence_group["sbj_contact_z"][:], dtype=np.float32)[selected]
            if source_contact.ndim != 2 or source_contact.shape[1] != 6890 or source_contact_z.ndim != 2 or source_contact_z.shape[1] != 128:
                raise ValueError(f"required contact fields have invalid shapes: {subject_key}/{sequence_key}")
            contact_all = build_contact_map_sequence(sequence_group, contact_vertex_indices, selected) if mode[2] == 0 else None
            gender = parse_gender_value(sequence_group.attrs.get("gender", "male"))
            gender_code = 1 if gender == "female" else 0
            if object_name not in pointnext_features:
                raise KeyError(f"PointNeXt feature missing for object: {object_name}")
            pointnext = pointnext_features[object_name].reshape(1, 1024)
            pointnext_all = np.repeat(pointnext, frame_count, axis=0)
            data = {"object_name": object_name, "gender": gender, "gender_code": np.full(frame_count, gender_code, dtype=np.int64), "human": human_all, "object": object_all, "contact": contact_all, "class": np.eye(40, dtype=np.float32)[OBJECT_CLASS_TO_INDEX[object_name]][None, :].repeat(frame_count, axis=0), "pointnext": pointnext_all, "ground": ground_all, "canonical": canonical_vertices, "canonical_keypoints": canonical_keypoints, "source_vertices": source_vertices, "source_joints": source_joints, "source_betas": source_betas, "source_transl": source_transl, "source_pose": source_pose, "source_object_vertices": source_object_vertices, "source_object_rotation": source_object_rotation, "source_object_center": source_object_center, "source_contact": source_contact, "source_contact_z": source_contact_z, "sdf": load_object_sdf_map(source_hdf5_path, [object_name])[object_name]}
            for local_index in range(frame_count):
                records.append({"subject": subject_key, "sequence": sequence_key, "local": local_index, "data": data})
        progress = tqdm(total=len(records), desc=f"sampling {infer_dataset_name(source_hdf5_path)}")
        for start in range(0, len(records), 1024):
            real_count = min(1024, len(records) - start)
            refs = records[start:start + real_count]
            padded_refs = refs + [refs[-1]] * (1024 - real_count)
            def stack(name: str) -> np.ndarray:
                return np.stack([ref["data"][name][ref["local"]] for ref in padded_refs], axis=0).astype(np.float32)
            human_np = stack("human")
            object_np = stack("object")
            class_np = stack("class")
            pointnext_np = stack("pointnext")
            ground_np = stack("ground").reshape(1024)
            canonical_rows = [ref["data"]["canonical"][ref["local"]] for ref in padded_refs]
            keypoint_rows = [ref["data"]["canonical_keypoints"][ref["local"]] for ref in padded_refs]
            max_vertices = max(int(value.shape[0]) for value in canonical_rows)
            max_keypoints = max(int(value.shape[0]) for value in keypoint_rows)
            canonical_np = np.full((1024, max_vertices, 3), 1e6, dtype=np.float32)
            canonical_keypoints_np = np.full((1024, max_keypoints, 3), 1e6, dtype=np.float32)
            for idx, value in enumerate(canonical_rows):
                canonical_np[idx, :value.shape[0]] = value
            for idx, value in enumerate(keypoint_rows):
                canonical_keypoints_np[idx, :value.shape[0]] = value
            gender_np = np.asarray([ref["data"]["gender_code"][ref["local"]] for ref in padded_refs], dtype=np.int64)
            contact_np = None if mode[2] == 1 else stack("contact")
            sdf_entries = [ref["data"]["sdf"] for ref in padded_refs]
            sdf_grid = None
            sdf_min = None
            sdf_max = None
            if any(entry is not None for entry in sdf_entries):
                if any(entry is None for entry in sdf_entries):
                    raise RuntimeError("object SDF is missing for a frame in a packed guidance batch")
                shapes = [entry["sdf"].shape for entry in sdf_entries]
                max_shape = tuple(max(shape[axis] for shape in shapes) for axis in range(3))
                sdf_grid = np.zeros((1024, 1, *max_shape), dtype=np.float32)
                sdf_min = np.zeros((1024, 3), dtype=np.float32)
                sdf_max = np.zeros((1024, 3), dtype=np.float32)
                for idx, entry in enumerate(sdf_entries):
                    shape = entry["sdf"].shape
                    sdf_grid[idx, 0, :shape[0], :shape[1], :shape[2]] = entry["sdf"]
                    sdf_min[idx] = entry["bounds_min"]
                    sdf_max[idx] = entry["bounds_max"]
            pipeline_output = pipeline.sample(
                class_onehot=torch.from_numpy(class_np), pointnext_feature=torch.from_numpy(pointnext_np), mode=mode,
                human_condition=torch.from_numpy(human_np) if mode[0] == 0 else None, object_condition=torch.from_numpy(object_np) if mode[1] == 0 else None,
                contact_map_condition=torch.from_numpy(contact_np) if mode[2] == 0 else None, ground_z=torch.from_numpy(ground_np),
                object_canonical_vertices=torch.from_numpy(canonical_np), object_canonical_keypoints=torch.from_numpy(canonical_keypoints_np), gender_code=torch.from_numpy(gender_np),
                contact_vertex_indices=contact_index_tensor, foot_vertex_indices=foot_index_tensor,
                object_sdf_grid=None if sdf_grid is None else torch.from_numpy(sdf_grid), object_sdf_bounds_min=None if sdf_min is None else torch.from_numpy(sdf_min), object_sdf_bounds_max=None if sdf_max is None else torch.from_numpy(sdf_max),
                object_names=[str(ref["data"]["object_name"]) for ref in padded_refs], seed=seed + start,
                geometry_weight=geometry_guidance_weight, sdf_weight=sdf_guidance_weight, balance_ground_weight=balance_ground_guidance_weight,
                sdf_penetration_weight=sdf_penetration_weight, sdf_contact_weight=sdf_contact_weight, sdf_contact_margin=sdf_contact_margin,
                balance_ground_inner_weight=balance_ground_inner_weight, balance_com_weight=balance_com_weight,
            )
            sampled_human = pipeline_output.states["human"].detach().cpu()
            sampled_object = pipeline_output.states["object"].detach().cpu()
            sampled_contact_latent = pipeline_output.states["contact"].detach().cpu().numpy().astype(np.float32)
            sampled_contact_scores = pipeline_output.decoded["contact"]["map_scores"].detach().cpu().numpy().astype(np.float32)
            sampled_shape, sampled_transl, sampled_pose = convert_human_feature_to_outputs(sampled_human)
            sampled_shape_np = sampled_shape.numpy().astype(np.float32)
            sampled_transl_np = sampled_transl.numpy().astype(np.float32)
            sampled_pose_np = sampled_pose.numpy().astype(np.float32)
            sampled_obj_rotation, sampled_obj_center = convert_object_feature_to_outputs(sampled_object)
            sampled_obj_rotation_np = sampled_obj_rotation.numpy().astype(np.float32)
            sampled_obj_center_np = sampled_obj_center.numpy().astype(np.float32)
            generated_vertices = np.zeros((1024, 6890, 3), dtype=np.float32)
            generated_joints = np.zeros((1024, 73, 3), dtype=np.float32)
            if mode[0] == 1:
                for gender_name in ("male", "female"):
                    indexes = np.asarray([idx for idx, ref in enumerate(padded_refs) if ref["data"]["gender"] == gender_name], dtype=np.int64)
                    if indexes.size == 0:
                        continue
                    smpl_model = smplh_models[gender_name]
                    device = next(smpl_model.parameters()).device
                    pose9 = sampled_pose_np[indexes]
                    with torch.no_grad():
                        result = smpl_model(betas=torch.from_numpy(sampled_shape_np[indexes]).to(device), global_orient=torch.from_numpy(matrix9_to_axis_angle(pose9[:, 0])).to(device), body_pose=torch.from_numpy(matrix9_to_axis_angle(pose9[:, 1:22]).reshape(indexes.size, 63)).to(device), left_hand_pose=torch.from_numpy(matrix9_to_axis_angle(pose9[:, 22:37]).reshape(indexes.size, 45)).to(device), right_hand_pose=torch.from_numpy(matrix9_to_axis_angle(pose9[:, 37:52]).reshape(indexes.size, 45)).to(device), transl=torch.from_numpy(sampled_transl_np[indexes]).to(device), return_verts=True)
                    generated_vertices[indexes] = result.vertices.detach().cpu().numpy()[:, :6890]
                    joints = result.joints.detach().cpu().numpy()
                    generated_joints[indexes, :min(73, joints.shape[1])] = joints[:, :73]
            for idx, ref in enumerate(refs):
                data = ref["data"]
                local = int(ref["local"])
                output_group = output_file[ref["subject"]][ref["sequence"]]
                if mode[0] == 1:
                    output_group["sbj_v"][local] = generated_vertices[idx]
                    output_group["sbj_j"][local] = generated_joints[idx]
                    output_group["sbj_smpl_betas"][local] = sampled_shape_np[idx]
                    output_group["sbj_smpl_transl"][local] = sampled_transl_np[idx]
                    output_group["sbj_smpl_pose"][local] = sampled_pose_np[idx]
                else:
                    output_group["sbj_v"][local] = data["source_vertices"][local]
                    output_group["sbj_j"][local] = data["source_joints"][local]
                    output_group["sbj_smpl_betas"][local] = data["source_betas"][local]
                    output_group["sbj_smpl_transl"][local] = data["source_transl"][local]
                    output_group["sbj_smpl_pose"][local] = data["source_pose"][local]
                if mode[1] == 1:
                    output_group["obj_c"][local] = sampled_obj_center_np[idx]
                    output_group["obj_R"][local] = sampled_obj_rotation_np[idx]
                    output_group["obj_v"][local] = transform_object_vertices(data["source_object_vertices"][local:local + 1], data["source_object_rotation"][local:local + 1], data["source_object_center"][local:local + 1], sampled_obj_rotation_np[idx:idx + 1], sampled_obj_center_np[idx:idx + 1])[0]
                else:
                    output_group["obj_v"][local] = data["source_object_vertices"][local]
                    output_group["obj_c"][local] = data["source_object_center"][local]
                    output_group["obj_R"][local] = data["source_object_rotation"][local]
                output_group["sbj_contact_z"][local] = sampled_contact_latent[idx] if mode[2] == 1 else data["source_contact_z"][local]
                if mode[2] == 1:
                    contact_full = np.zeros(6890, dtype=np.float32)
                    contact_full[contact_vertex_indices] = (sampled_contact_scores[idx, :contact_vertex_indices.shape[0]] > contact_map_threshold).astype(np.float32)
                    output_group["sbj_contact"][local] = contact_full
                else:
                    output_group["sbj_contact"][local] = data["source_contact"][local]
                progress.update(1)
        progress.close()
def main() -> None:
    args = build_arg_parser().parse_args()
    if int(args.batch_size) != 1024:
        raise ValueError("batch-size must be exactly 1024")
    mode = normalize_MSFHOI_mode(args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise KeyError("checkpoint must contain model and config")
    model = MSFHOIFlowModel(**resolve_MSFHOI_model_kwargs_from_config(checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    load_contact_weights(model, str(PROJECT_ROOT / args.contact_ae_path), device)
    solver = MSFHOIODESolver(timestep_scheduler=MSFHOITimestepScheduler(num_steps=args.solver_steps), smpl_model_folder=str(PROJECT_ROOT / args.smpl_model_folder))
    pipeline = MSFHOIPipeline(model=model, solver=solver, device=device)
    smplh_models = build_smplh_model_cache(str(PROJECT_ROOT / args.smpl_model_folder), device) if mode[0] == 1 else {}
    pointnext_paths = [PROJECT_ROOT / value for value in args.pointnext_files]
    pointnext_features = load_pointnext_features([str(value) for value in pointnext_paths])
    contact_vertex_indices = load_contact_vertex_indices(args.contact_index_file)
    foot_vertex_indices = load_human_foot_vertex_indices(args.segmentation_file)
    output_base = resolve_sampling_output_base(PROJECT_ROOT / args.output_root, mode)
    for repetition in range(int(args.repetitions)):
        for source_path_value in args.test_hdf5_files:
            source_path = PROJECT_ROOT / source_path_value
            if not source_path.exists():
                raise FileNotFoundError(f"test HDF5 not found: {source_path}")
            dataset_name = infer_dataset_name(str(source_path))
            output_folder = output_base / dataset_name / mode_to_output_folder_name(mode)
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f"samples_rep_{repetition:02d}.hdf5"
            sample_dataset(pipeline=pipeline, smplh_models=smplh_models, source_hdf5_path=str(source_path), output_hdf5_path=output_path, mode=mode, pointnext_features=pointnext_features, contact_vertex_indices=contact_vertex_indices, foot_vertex_indices=foot_vertex_indices, batch_size=1024, seed=int(args.seed) + repetition * 100000, contact_map_threshold=float(args.contact_map_threshold), geometry_guidance_weight=float(args.geometry_guidance_weight), sdf_guidance_weight=float(args.sdf_guidance_weight), balance_ground_guidance_weight=float(args.balance_ground_guidance_weight), sdf_penetration_weight=float(args.sdf_penetration_weight), sdf_contact_weight=float(args.sdf_contact_weight), sdf_contact_margin=float(args.sdf_contact_margin), balance_ground_inner_weight=float(args.balance_ground_inner_weight), balance_com_weight=float(args.balance_com_weight), max_frames_per_sequence=args.max_frames_per_sequence, frame_selection_strategy=args.frame_selection_strategy, max_sequences=args.max_sequences)
            print(f"saved_sample={output_path}")
    meta = {"checkpoint": str(checkpoint_path), "mode": list(mode), "batch_size": 1024, "contact_index_file": args.contact_index_file, "segmentation_file": args.segmentation_file, "contact_ae_path": args.contact_ae_path, "smpl_model_folder": args.smpl_model_folder, "solver_steps": int(args.solver_steps), "solver_schedule": "linear"}
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
