from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.transformer import MSFHOIFlowModel

try:
    import smplx
except Exception:
    smplx = None

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:
    ConvexHull = None
    QhullError = RuntimeError


MSFHOIMode = Tuple[int, int, int]
MSFHOI_MODE_LIBRARY: Tuple[MSFHOIMode, ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
MSFHOI_MODALITY_NAMES: Tuple[str, str, str] = ("human", "object", "contact")
MSFHOI_OBJECT_CLASS_NAMES: Tuple[str, ...] = (
    "backpack",
    "banana",
    "basketball",
    "binoculars",
    "boxlarge",
    "boxlong",
    "boxmedium",
    "boxsmall",
    "boxtiny",
    "camera",
    "chairblack",
    "chairwood",
    "cup",
    "doorknob",
    "eyeglasses",
    "flashlight",
    "flute",
    "fryingpan",
    "gamecontroller",
    "hammer",
    "headphones",
    "keyboard",
    "knife",
    "lightbulb",
    "monitor",
    "mouse",
    "mug",
    "phone",
    "plasticcontainer",
    "stool",
    "suitcase",
    "tablesmall",
    "tablesquare",
    "teapot",
    "toolbox",
    "toothbrush",
    "trashbin",
    "wineglass",
    "yogaball",
    "yogamat",
)
HUMAN_MASS_GRAMS: float = 70000.0
BALANCE_SUPPORT_HEIGHT_BAND_METERS: float = 0.02
BALANCE_OBJECT_GROUND_CONTACT_MAX_GAP_METERS: float = 0.05
BALANCE_OBJECT_MASS_THRESHOLD_GRAMS: float = 2000.0
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
def normalize_MSFHOI_mode(mode: Union[str, Sequence[int], torch.Tensor]) -> MSFHOIMode:
    if isinstance(mode, str):
        cleaned = mode.replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
        if len(cleaned) != 3 or any(character not in {"0", "1"} for character in cleaned):
            raise ValueError(f"Invalid mode string: {mode}")
        parsed_mode = tuple(int(character) for character in cleaned)
    elif isinstance(mode, torch.Tensor):
        flat = mode.to(dtype=torch.int64).flatten().tolist()
        if len(flat) != 3:
            raise ValueError(f"Mode tensor must have 3 elements, got {len(flat)}")
        parsed_mode = tuple(int(value) for value in flat)
    else:
        if len(mode) != 3:
            raise ValueError(f"Mode must contain 3 entries, got {len(mode)}")
        parsed_mode = tuple(int(value) for value in mode)

    if parsed_mode not in MSFHOI_MODE_LIBRARY:
        raise ValueError(f"Mode {parsed_mode} is unsupported. Available modes: {MSFHOI_MODE_LIBRARY}")
    return parsed_mode


def list_MSFHOI_modes() -> List[MSFHOIMode]:
    return list(MSFHOI_MODE_LIBRARY)


def _expand_sdf_bound(
    bound: Union[torch.Tensor, Sequence[float]],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(bound, torch.Tensor):
        bound_tensor = bound.to(device=device, dtype=dtype)
    else:
        bound_tensor = torch.as_tensor(bound, device=device, dtype=dtype)

    if bound_tensor.dim() == 1:
        if bound_tensor.shape[0] != 3:
            raise ValueError(f"{name} must have shape [3], got {tuple(bound_tensor.shape)}")
        return bound_tensor.unsqueeze(0).expand(batch_size, -1)

    if bound_tensor.dim() == 2 and bound_tensor.shape[1] == 3:
        if bound_tensor.shape[0] == 1:
            return bound_tensor.expand(batch_size, -1)
        if bound_tensor.shape[0] == batch_size:
            return bound_tensor

    raise ValueError(
        f"{name} must have shape [3], [1, 3], or [B, 3]. "
        f"Got {tuple(bound_tensor.shape)} for B={batch_size}."
    )


def _compute_sdf_guidance_loss(
    *,
    human_kp_690: torch.Tensor,
    obj_translation: torch.Tensor,
    obj_rotation: torch.Tensor,
    sdf_grid: torch.Tensor,
    grid_bounds: Tuple[Union[torch.Tensor, Sequence[float]], Union[torch.Tensor, Sequence[float]]],
    contact_indices: Optional[torch.Tensor] = None,
    contact_mask: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    contact_margin: float = 0.05,
) -> Dict[str, torch.Tensor]:
    if human_kp_690.dim() != 3 or human_kp_690.shape[-1] != 3:
        raise ValueError(f"human_kp_690 must have shape [B, N, 3], got {tuple(human_kp_690.shape)}")
    if obj_translation.dim() != 2 or obj_translation.shape[-1] != 3:
        raise ValueError(
            f"obj_translation must have shape [B, 3], got {tuple(obj_translation.shape)}"
        )
    if obj_rotation.dim() != 3 or obj_rotation.shape[1:] != (3, 3):
        raise ValueError(f"obj_rotation must have shape [B, 3, 3], got {tuple(obj_rotation.shape)}")
    if obj_translation.shape[0] != human_kp_690.shape[0] or obj_rotation.shape[0] != human_kp_690.shape[0]:
        raise ValueError("human_kp_690, obj_translation, and obj_rotation must share the same batch size.")
    if sdf_grid.dim() != 5 or sdf_grid.shape[1] != 1:
        raise ValueError(f"sdf_grid must have shape [B, 1, D, H, W], got {tuple(sdf_grid.shape)}")

    batch_size, num_points = human_kp_690.shape[0], human_kp_690.shape[1]
    if sdf_grid.shape[0] == 1 and batch_size > 1:
        sdf_grid = sdf_grid.expand(batch_size, -1, -1, -1, -1)
    elif sdf_grid.shape[0] != batch_size:
        raise ValueError(
            f"sdf_grid batch dimension must be 1 or B={batch_size}, got {sdf_grid.shape[0]}"
        )
    sdf_grid = sdf_grid.to(device=human_kp_690.device, dtype=human_kp_690.dtype)

    bounds_min, bounds_max = grid_bounds
    bounds_min_tensor = _expand_sdf_bound(
        bounds_min,
        batch_size=batch_size,
        device=human_kp_690.device,
        dtype=human_kp_690.dtype,
        name="grid_bounds[0]",
    )
    bounds_max_tensor = _expand_sdf_bound(
        bounds_max,
        batch_size=batch_size,
        device=human_kp_690.device,
        dtype=human_kp_690.dtype,
        name="grid_bounds[1]",
    )
    bound_extent = (bounds_max_tensor - bounds_min_tensor).clamp_min(1e-6)

    effective_weights = {
        "w_pen": 10.0,
        "w_contact": 2.0,
    }
    if weights is not None:
        if "w_pen" in weights:
            effective_weights["w_pen"] = float(weights["w_pen"])
        if "w_contact" in weights:
            effective_weights["w_contact"] = float(weights["w_contact"])

    centered_points = human_kp_690 - obj_translation.unsqueeze(1)
    local_points = torch.bmm(centered_points, obj_rotation)
    normalized_points = 2.0 * (
        (local_points - bounds_min_tensor.unsqueeze(1)) / bound_extent.unsqueeze(1)
    ) - 1.0
    sample_grid = normalized_points.view(batch_size, num_points, 1, 1, 3)
    sdf_values = F.grid_sample(
        sdf_grid,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(batch_size, num_points)

    penetration_residual = F.relu(-sdf_values)
    penetration_loss_per_sample = penetration_residual.square().sum(dim=1)

    if contact_mask is not None and contact_indices is not None:
        raise ValueError("Provide either contact_indices or contact_mask, not both.")

    if contact_mask is not None:
        if contact_mask.shape != sdf_values.shape:
            raise ValueError(
                f"contact_mask must have shape {tuple(sdf_values.shape)}, got {tuple(contact_mask.shape)}"
            )
        contact_weight_mask = contact_mask.to(device=sdf_values.device, dtype=sdf_values.dtype)
        contact_residual = F.relu(sdf_values - float(contact_margin))
        contact_loss_per_sample = (contact_residual.square() * contact_weight_mask).sum(dim=1)
    elif contact_indices is not None:
        if contact_indices.dim() != 2:
            raise ValueError(
                f"contact_indices must have shape [B, N_contact], got {tuple(contact_indices.shape)}"
            )
        if contact_indices.shape[0] != batch_size:
            raise ValueError("contact_indices batch dimension must match human_kp_690.")
        contact_indices = contact_indices.to(device=sdf_values.device, dtype=torch.long)
        if contact_indices.numel() == 0:
            contact_loss_per_sample = torch.zeros_like(penetration_loss_per_sample)
        else:
            min_index = int(contact_indices.min())
            max_index = int(contact_indices.max())
            if min_index < 0 or max_index >= num_points:
                raise ValueError(
                    f"contact_indices must be in [0, {num_points - 1}], got [{min_index}, {max_index}]"
                )
            gathered_sdf = torch.gather(sdf_values, dim=1, index=contact_indices)
            contact_residual = F.relu(gathered_sdf - float(contact_margin))
            contact_loss_per_sample = contact_residual.square().sum(dim=1)
    else:
        contact_loss_per_sample = torch.zeros_like(penetration_loss_per_sample)

    weighted_loss_per_sample = (
        float(effective_weights["w_pen"]) * penetration_loss_per_sample
        + float(effective_weights["w_contact"]) * contact_loss_per_sample
    )

    return {
        "loss_total": weighted_loss_per_sample.mean(),
        "loss_penetration": penetration_loss_per_sample.mean(),
        "loss_contact": contact_loss_per_sample.mean(),
        "loss_per_sample": weighted_loss_per_sample,
        "sdf_values": sdf_values,
        "local_points": local_points,
    }


def compute_sdf_guidance_gradients(
    human_kp_690: torch.Tensor,
    obj_translation: torch.Tensor,
    obj_rotation: torch.Tensor,
    contact_indices: torch.Tensor,
    sdf_grid: torch.Tensor,
    grid_bounds: Tuple[Union[torch.Tensor, Sequence[float]], Union[torch.Tensor, Sequence[float]]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    loss_outputs = _compute_sdf_guidance_loss(
        human_kp_690=human_kp_690,
        obj_translation=obj_translation,
        obj_rotation=obj_rotation,
        contact_indices=contact_indices,
        sdf_grid=sdf_grid,
        grid_bounds=grid_bounds,
        weights=weights,
    )
    gradients = torch.autograd.grad(
        loss_outputs["loss_total"],
        [human_kp_690, obj_translation, obj_rotation],
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    return {
        "loss_total": loss_outputs["loss_total"],
        "loss_penetration": loss_outputs["loss_penetration"],
        "loss_contact": loss_outputs["loss_contact"],
        "sdf_values": loss_outputs["sdf_values"],
        "human_kp_690_grad": gradients[0] if gradients[0] is not None else torch.zeros_like(human_kp_690),
        "obj_translation_grad": gradients[1]
        if gradients[1] is not None
        else torch.zeros_like(obj_translation),
        "obj_rotation_grad": gradients[2] if gradients[2] is not None else torch.zeros_like(obj_rotation),
    }


def _compute_2d_convex_hull_indices(points_xy: torch.Tensor) -> torch.Tensor:
    if points_xy.dim() != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"points_xy must have shape [N, 2], got {tuple(points_xy.shape)}")

    num_points = int(points_xy.shape[0])
    if num_points <= 1:
        return torch.arange(num_points, device=points_xy.device, dtype=torch.long)

    points_cpu = points_xy.detach().cpu()
    if ConvexHull is not None:
        try:
            hull = ConvexHull(points_cpu.numpy())
            hull_indices = torch.as_tensor(
                hull.vertices.tolist(),
                device=points_xy.device,
                dtype=torch.long,
            )
            if hull_indices.numel() > 0:
                return hull_indices
        except QhullError:
            pass

    sorted_indices = sorted(
        range(num_points),
        key=lambda index: (float(points_cpu[index, 0]), float(points_cpu[index, 1])),
    )

    def cross(index_o: int, index_a: int, index_b: int) -> float:
        point_o = points_cpu[index_o]
        point_a = points_cpu[index_a]
        point_b = points_cpu[index_b]
        return float(
            (point_a[0] - point_o[0]) * (point_b[1] - point_o[1])
            - (point_a[1] - point_o[1]) * (point_b[0] - point_o[0])
        )

    lower: List[int] = []
    for point_index in sorted_indices:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point_index) <= 0.0:
            lower.pop()
        lower.append(point_index)

    upper: List[int] = []
    for point_index in reversed(sorted_indices):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point_index) <= 0.0:
            upper.pop()
        upper.append(point_index)

    hull_indices = lower[:-1] + upper[:-1]
    if len(hull_indices) == 0:
        hull_indices = sorted_indices[:1]

    deduplicated_indices: List[int] = []
    seen_indices = set()
    for point_index in hull_indices:
        if point_index in seen_indices:
            continue
        deduplicated_indices.append(point_index)
        seen_indices.add(point_index)

    return torch.as_tensor(deduplicated_indices, device=points_xy.device, dtype=torch.long)


def _compute_balance_ground_guidance_loss(
    human_foot_vertices: torch.Tensor,
    object_center_of_mass: torch.Tensor,
    ground_z: torch.Tensor,
    object_names: Sequence[str],
    ground_weight: float = 1.0,
    balance_weight: float = 1.0,
) -> torch.Tensor:
    if human_foot_vertices.dim() != 3 or human_foot_vertices.shape[-1] != 3:
        raise ValueError("human_foot_vertices must have shape [B, N, 3]")
    ground = ground_z.reshape(-1).to(device=human_foot_vertices.device, dtype=human_foot_vertices.dtype)
    if ground.numel() == 1:
        ground = ground.expand(human_foot_vertices.shape[0])
    ground_loss = F.relu((human_foot_vertices[..., 2] - ground.unsqueeze(-1)).abs() - BALANCE_SUPPORT_HEIGHT_BAND_METERS).square().mean()
    human_center = human_foot_vertices.mean(dim=1)
    masses = torch.as_tensor([OBJECT_MASS_GRAMS.get(str(name).strip().lower(), 1000.0) for name in object_names], device=human_foot_vertices.device, dtype=human_foot_vertices.dtype)
    human_mass = torch.full_like(masses, HUMAN_MASS_GRAMS)
    system_center = (human_mass.unsqueeze(-1) * human_center + masses.unsqueeze(-1) * object_center_of_mass) / (human_mass + masses).unsqueeze(-1)
    support_center = human_foot_vertices[..., :2].mean(dim=1)
    balance_loss = (system_center[..., :2] - support_center).square().sum(dim=-1).mean()
    return float(ground_weight) * ground_loss + float(balance_weight) * balance_loss

@dataclass
class MSFHOITimestepScheduler:
    num_steps: int = 32

    def build(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        return torch.linspace(0.0, 1.0, int(self.num_steps), device=device, dtype=dtype)

@dataclass
class MSFHOIInferenceGuidanceConfig:
    geometry_weight: float = 0.0
    sdf_weight: float = 0.0
    balance_ground_weight: float = 0.0
    sdf_penetration_weight: float = 10.0
    sdf_contact_weight: float = 2.0
    sdf_contact_margin: float = 0.05
    sdf_max_gradient_norm: float = 50.0
    geometry_max_gradient_norm: float = 50.0
    balance_ground_weight_inner: float = 1.0
    balance_com_weight: float = 1.0
    balance_max_gradient_norm: float = 50.0


@dataclass
class MSFHOIInferenceGuidanceContext:
    ground_z: Optional[torch.Tensor] = None
    object_canonical_vertices: Optional[torch.Tensor] = None
    object_canonical_keypoints: Optional[torch.Tensor] = None
    gender_code: Optional[torch.Tensor] = None
    contact_vertex_indices: Optional[torch.Tensor] = None
    foot_vertex_indices: Optional[torch.Tensor] = None
    object_sdf_grid: Optional[torch.Tensor] = None
    object_sdf_bounds_min: Optional[torch.Tensor] = None
    object_sdf_bounds_max: Optional[torch.Tensor] = None
    object_names: Optional[Sequence[str]] = None
    balance_target_human_vertices: Optional[torch.Tensor] = None
    balance_target_object_translation: Optional[torch.Tensor] = None
    balance_target_object_rotation: Optional[torch.Tensor] = None

class MSFHOIODESolver:
    def __init__(
        self,
        method: str = "heun",
        timestep_scheduler: Optional[MSFHOITimestepScheduler] = None,
        smpl_model_folder: str = "data/smplx_models",
    ) -> None:
        if method != "heun":
            raise ValueError("method must be heun")
        self.method = method
        self.timestep_scheduler = timestep_scheduler or MSFHOITimestepScheduler()
        self.smpl_model_folder = Path(smpl_model_folder)
        self.repo_root = Path(__file__).resolve().parent
        self.object_canonical_com_cache_file = self.repo_root / "assets/object_canonical_mass_properties.json"
        self._smpl_model_cache: Dict[str, nn.Module] = {}
        self._smpl_subset_cache: Dict[Tuple[str, str, Tuple[int, ...]], Dict[str, torch.Tensor]] = {}
        self._object_canonical_com_cache_raw: Optional[Dict[str, Dict[str, object]]] = None
        self._alm_feet_loss = None
        self._alm_object_loss = None
        self._alm_signature = None

    @staticmethod
    def _ensure_conditioned_states(
        states: Dict[str, torch.Tensor],
        condition_states: Dict[str, torch.Tensor],
        mode: MSFHOIMode,
    ) -> None:
        for index, modality_name in enumerate(MSFHOI_MODALITY_NAMES):
            if mode[index] == 0:
                states[modality_name] = condition_states[modality_name]

    @staticmethod
    def _ensure_finite_states(states: Dict[str, torch.Tensor], *, label: str) -> None:
        for modality_name, state in states.items():
            if torch.isfinite(state).all():
                continue
            invalid_count = int((~torch.isfinite(state)).sum().detach().cpu().item())
            raise FloatingPointError(
                f"Non-finite values in {label}.{modality_name}: "
                f"shape={tuple(state.shape)} invalid_count={invalid_count}"
            )

    @staticmethod
    def _build_modality_times(
        *,
        mode: MSFHOIMode,
        t_scalar: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        ones_time = torch.ones(batch_size, device=device, dtype=dtype)
        running_time = t_scalar.expand(batch_size)

        return {
            "human": running_time if mode[0] == 1 else ones_time,
            "object": running_time if mode[1] == 1 else ones_time,
            "contact": running_time if mode[2] == 1 else ones_time,
        }

    @staticmethod
    def _random_like(
        reference: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    @staticmethod
    def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
        vector_a = rotation_6d[..., 0:3]
        vector_b = rotation_6d[..., 3:6]
        basis_1 = torch.nn.functional.normalize(vector_a, dim=-1)
        projection = (basis_1 * vector_b).sum(dim=-1, keepdim=True)
        basis_2 = torch.nn.functional.normalize(vector_b - projection * basis_1, dim=-1)
        basis_3 = torch.cross(basis_1, basis_2, dim=-1)
        return torch.stack([basis_1, basis_2, basis_3], dim=-2)

    @staticmethod
    def _matrix_to_axis_angle(rotation_matrix: torch.Tensor) -> torch.Tensor:
        trace = (
            rotation_matrix[..., 0, 0]
            + rotation_matrix[..., 1, 1]
            + rotation_matrix[..., 2, 2]
        )
        trace_cos = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        angle = torch.acos(trace_cos)
        sine = torch.sin(angle)

        axis = torch.stack(
            [
                rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
                rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
                rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
            ],
            dim=-1,
        )
        axis = axis / (2.0 * sine.unsqueeze(-1) + 1e-7)
        axis_angle = axis * angle.unsqueeze(-1)

        small_angle = angle < 1e-4
        small_angle_approx = 0.5 * torch.stack(
            [
                rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
                rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
                rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
            ],
            dim=-1,
        )
        axis_angle = torch.where(small_angle.unsqueeze(-1), small_angle_approx, axis_angle)
        return axis_angle

    def _get_smpl_model(self, gender_name: str, device: torch.device) -> nn.Module:
        if smplx is None:
            raise ImportError("Package `smplx` is required for inference guidance.")

        gender = "female" if gender_name.lower().startswith("f") else "male"
        if gender not in self._smpl_model_cache:
            model_path = self.smpl_model_folder
            if not model_path.exists():
                raise FileNotFoundError(f"SMPL-H model folder not found: {model_path}")
            model = smplx.SMPLHLayer(
                model_path=str(model_path),
                gender=gender,
                ext="pkl",
                use_pca=False,
                num_betas=10,
                flat_hand_mean=True,
            )
            model = model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._smpl_model_cache[gender] = model

        return self._smpl_model_cache[gender].to(device=device)

    def _load_object_canonical_com_cache(self) -> Dict[str, Dict[str, object]]:
        if self._object_canonical_com_cache_raw is not None:
            return self._object_canonical_com_cache_raw
        cache_path = self.object_canonical_com_cache_file
        if not cache_path.exists():
            raise FileNotFoundError(f"object COM file not found: {cache_path}")
        with cache_path.open("r", encoding="utf-8") as file_pointer:
            loaded = json.load(file_pointer)
        objects = loaded.get("objects")
        if not isinstance(objects, dict):
            raise KeyError(f"object COM file must contain objects: {cache_path}")
        normalized: Dict[str, Dict[str, object]] = {}
        for object_name, entry in objects.items():
            if not isinstance(entry, dict):
                raise TypeError(f"object COM entry must be a mapping: {object_name}")
            normalized[str(object_name).strip().lower()] = entry
        self._object_canonical_com_cache_raw = normalized
        return normalized
    def _resolve_object_center_of_mass(
        self,
        *,
        object_names: Sequence[str],
        object_rotation: torch.Tensor,
        object_translation: torch.Tensor,
    ) -> torch.Tensor:
        cache = self._load_object_canonical_com_cache()

        canonical_com_rows: List[List[float]] = []
        for raw_name in object_names:
            name = str(raw_name).strip().lower()
            entry = cache.get(name)
            if entry is None:
                raise KeyError(f"object COM entry missing: {name}")
            com_vector = entry.get("canonical_center_of_mass")
            if not isinstance(com_vector, (list, tuple)) or len(com_vector) != 3:
                raise ValueError(f"object COM entry must contain canonical_center_of_mass [3]: {name}")
            try:
                canonical_com_rows.append([float(value) for value in com_vector])
            except (TypeError, ValueError) as error:
                raise ValueError(f"object COM entry contains non-numeric values: {name}") from error

        canonical_com = torch.as_tensor(
            canonical_com_rows,
            device=object_translation.device,
            dtype=object_translation.dtype,
        )
        return (
            torch.matmul(canonical_com.unsqueeze(1), object_rotation.transpose(1, 2)).squeeze(1)
            + object_translation
        )

    def _get_smpl_vertex_subset_cache(
        self,
        gender_name: str,
        *,
        device: torch.device,
        vertex_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        smpl_model = self._get_smpl_model(gender_name, device=device)
        vertex_indices = vertex_indices.reshape(-1).to(device=device, dtype=torch.long)
        index_key = tuple(int(index) for index in vertex_indices.detach().cpu().tolist())
        cache_key = (gender_name, str(device), index_key)
        cached = self._smpl_subset_cache.get(cache_key)
        if cached is not None:
            return cached

        vertex_count = int(smpl_model.v_template.shape[0])
        if vertex_indices.numel() == 0:
            raise ValueError("vertex_indices must contain at least one entry.")
        if int(vertex_indices.min()) < 0 or int(vertex_indices.max()) >= vertex_count:
            raise ValueError(
                f"vertex_indices must be in [0, {vertex_count - 1}], "
                f"got [{int(vertex_indices.min())}, {int(vertex_indices.max())}]."
            )

        posedirs = smpl_model.posedirs.reshape(smpl_model.posedirs.shape[0], vertex_count, 3)
        joint_template = smplx.lbs.vertices2joints(
            smpl_model.J_regressor,
            smpl_model.v_template.unsqueeze(0),
        ).squeeze(0)
        joint_shapedirs = torch.einsum("jv,vcn->jcn", smpl_model.J_regressor, smpl_model.shapedirs)

        template_vertices = smpl_model.v_template
        selected_template_vertices = template_vertices.index_select(0, vertex_indices)
        nearest_selected = torch.cdist(
            template_vertices.unsqueeze(0),
            selected_template_vertices.unsqueeze(0),
            p=2,
        ).squeeze(0).argmin(dim=1)
        centroid_weights = torch.bincount(
            nearest_selected,
            minlength=vertex_indices.numel(),
        ).to(dtype=template_vertices.dtype)
        centroid_weights = centroid_weights / centroid_weights.sum().clamp_min(1.0)
        uniform_com_weights = centroid_weights

        cached = {
            "vertex_indices": vertex_indices,
            "v_template": selected_template_vertices,
            "shapedirs": smpl_model.shapedirs.index_select(0, vertex_indices),
            "posedirs": posedirs.index_select(1, vertex_indices).reshape(
                smpl_model.posedirs.shape[0],
                -1,
            ),
            "lbs_weights": smpl_model.lbs_weights.index_select(0, vertex_indices),
            "joint_template": joint_template,
            "joint_shapedirs": joint_shapedirs,
            "parents": smpl_model.parents,
            "centroid_weights": centroid_weights,
            "uniform_com_weights": uniform_com_weights,
        }
        self._smpl_subset_cache[cache_key] = cached
        return cached

    def _reconstruct_human_vertex_subset(
        self,
        *,
        human_state: torch.Tensor,
        gender_code: Optional[torch.Tensor],
        vertex_indices: torch.Tensor,
        return_weighted_centroid: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if smplx is None:
            raise ImportError("Package `smplx` is required for inference guidance.")

        batch_size = human_state.shape[0]
        if gender_code is None:
            gender_code = torch.zeros(batch_size, device=human_state.device, dtype=torch.long)
        else:
            gender_code = gender_code.reshape(-1).to(device=human_state.device, dtype=torch.long)
            if gender_code.shape[0] != batch_size:
                raise ValueError("gender_code must have shape [B].")

        vertex_indices = vertex_indices.reshape(-1).to(device=human_state.device, dtype=torch.long)
        subset_size = int(vertex_indices.numel())
        if subset_size <= 0:
            raise ValueError("vertex_indices must contain at least one entry.")

        betas = human_state[:, :10]
        global_orient_6d = human_state[:, 10:16].reshape(batch_size, 1, 6)
        pose_6d = human_state[:, 16:322].reshape(batch_size, 51, 6)
        transl = human_state[:, 322:325]

        global_orient = self._rotation_6d_to_matrix(global_orient_6d)
        body_pose = self._rotation_6d_to_matrix(pose_6d[:, :21])
        left_hand_pose = self._rotation_6d_to_matrix(pose_6d[:, 21:36])
        right_hand_pose = self._rotation_6d_to_matrix(pose_6d[:, 36:51])

        vertices = torch.zeros(
            batch_size,
            subset_size,
            3,
            device=human_state.device,
            dtype=human_state.dtype,
        )
        weighted_centers = (
            torch.zeros(batch_size, 3, device=human_state.device, dtype=human_state.dtype)
            if return_weighted_centroid
            else None
        )
        ident = torch.eye(3, device=human_state.device, dtype=human_state.dtype)

        for gender_value, gender_name in ((0, "male"), (1, "female")):
            sample_indices = torch.nonzero(gender_code == gender_value, as_tuple=False).reshape(-1)
            if sample_indices.numel() == 0:
                continue

            sparse_cache = self._get_smpl_vertex_subset_cache(
                gender_name,
                device=human_state.device,
                vertex_indices=vertex_indices,
            )
            sample_betas = betas.index_select(0, sample_indices)
            sample_rot_mats = torch.cat(
                [
                    global_orient.index_select(0, sample_indices),
                    body_pose.index_select(0, sample_indices),
                    left_hand_pose.index_select(0, sample_indices),
                    right_hand_pose.index_select(0, sample_indices),
                ],
                dim=1,
            )
            sample_joint_template = sparse_cache["joint_template"].unsqueeze(0).expand(
                sample_indices.numel(),
                -1,
                -1,
            )
            sample_joints = sample_joint_template + smplx.lbs.blend_shapes(
                sample_betas,
                sparse_cache["joint_shapedirs"],
            )
            sample_v_shaped = sparse_cache["v_template"].unsqueeze(0) + smplx.lbs.blend_shapes(
                sample_betas,
                sparse_cache["shapedirs"],
            )
            pose_feature = (sample_rot_mats[:, 1:] - ident).reshape(sample_indices.numel(), -1)
            pose_offsets = torch.matmul(pose_feature, sparse_cache["posedirs"]).reshape(
                sample_indices.numel(),
                subset_size,
                3,
            )
            sample_v_posed = sample_v_shaped + pose_offsets
            _, rel_transforms = smplx.lbs.batch_rigid_transform(
                sample_rot_mats,
                sample_joints,
                sparse_cache["parents"],
                dtype=human_state.dtype,
            )
            skinning_transforms = torch.matmul(
                sparse_cache["lbs_weights"].unsqueeze(0).expand(sample_indices.numel(), -1, -1),
                rel_transforms.reshape(sample_indices.numel(), rel_transforms.shape[1], 16),
            ).reshape(sample_indices.numel(), subset_size, 4, 4)
            homogeneous_vertices = torch.cat(
                [
                    sample_v_posed,
                    torch.ones(
                        sample_indices.numel(),
                        subset_size,
                        1,
                        device=human_state.device,
                        dtype=human_state.dtype,
                    ),
                ],
                dim=2,
            )
            sample_vertices = torch.matmul(
                skinning_transforms,
                homogeneous_vertices.unsqueeze(-1),
            )[:, :, :3, 0]
            sample_vertices = sample_vertices + transl.index_select(0, sample_indices).unsqueeze(1)
            vertices.index_copy_(0, sample_indices, sample_vertices)

            if weighted_centers is not None:
                sample_centers = (
                    sample_vertices
                    * sparse_cache["uniform_com_weights"].to(dtype=sample_vertices.dtype).view(1, -1, 1)
                ).sum(dim=1)
                weighted_centers.index_copy_(0, sample_indices, sample_centers)

        return vertices, weighted_centers

    def _reconstruct_human_joints(
        self,
        *,
        human_state: torch.Tensor,
        gender_code: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = human_state.shape[0]
        if gender_code is None:
            gender_code = torch.zeros(batch_size, device=human_state.device, dtype=torch.long)
        else:
            gender_code = gender_code.reshape(-1).to(device=human_state.device, dtype=torch.long)
            if gender_code.shape[0] != batch_size:
                raise ValueError("gender_code must have shape [B].")

        betas = human_state[:, :10]
        global_orient_6d = human_state[:, 10:16].reshape(batch_size, 1, 6)
        pose_6d = human_state[:, 16:322].reshape(batch_size, 51, 6)
        transl = human_state[:, 322:325]
        global_orient = self._rotation_6d_to_matrix(global_orient_6d).reshape(batch_size, 1, 9)
        body_pose = self._rotation_6d_to_matrix(pose_6d[:, :21]).reshape(batch_size, 21, 9)
        left_hand_pose = self._rotation_6d_to_matrix(pose_6d[:, 21:36]).reshape(batch_size, 15, 9)
        right_hand_pose = self._rotation_6d_to_matrix(pose_6d[:, 36:51]).reshape(batch_size, 15, 9)
        joints = torch.zeros(batch_size, 73, 3, device=human_state.device, dtype=human_state.dtype)
        for gender_value, gender_name in ((0, "male"), (1, "female")):
            sample_indices = torch.nonzero(gender_code == gender_value, as_tuple=False).reshape(-1)
            if sample_indices.numel() == 0:
                continue

            smpl_model = self._get_smpl_model(gender_name, device=human_state.device)
            smpl_output = smpl_model(
                betas=betas[sample_indices],
                global_orient=global_orient[sample_indices],
                body_pose=body_pose[sample_indices],
                left_hand_pose=left_hand_pose[sample_indices],
                right_hand_pose=right_hand_pose[sample_indices],
                transl=transl[sample_indices],
                pose2rot=False,
                get_skin=True,
                return_full_pose=True,
                return_verts=False,
            )
            if smpl_output.joints.shape[1] < 73:
                raise RuntimeError(
                    "SMPL-H output must contain at least 73 joints for validation, "
                    f"got {smpl_output.joints.shape[1]}."
                )
            joints[sample_indices] = smpl_output.joints[:, :73]

        return joints

    def _reconstruct_human_vertices(
        self,
        *,
        human_state: torch.Tensor,
        gender_code: Optional[torch.Tensor],
        return_joints: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = human_state.shape[0]
        if gender_code is None:
            gender_code = torch.zeros(batch_size, device=human_state.device, dtype=torch.long)
        else:
            gender_code = gender_code.reshape(-1).to(device=human_state.device, dtype=torch.long)
            if gender_code.shape[0] != batch_size:
                raise ValueError("gender_code must have shape [B].")

        betas = human_state[:, :10]
        global_orient_6d = human_state[:, 10:16].reshape(batch_size, 1, 6)
        pose_6d = human_state[:, 16:322].reshape(batch_size, 51, 6)
        transl = human_state[:, 322:325]
        global_orient = self._rotation_6d_to_matrix(global_orient_6d).reshape(batch_size, 1, 9)
        body_pose = self._rotation_6d_to_matrix(pose_6d[:, :21]).reshape(batch_size, 21, 9)
        left_hand_pose = self._rotation_6d_to_matrix(pose_6d[:, 21:36]).reshape(batch_size, 15, 9)
        right_hand_pose = self._rotation_6d_to_matrix(pose_6d[:, 36:51]).reshape(batch_size, 15, 9)

        vertices = torch.zeros(batch_size, 6890, 3, device=human_state.device, dtype=human_state.dtype)
        joints = (
            torch.zeros(batch_size, 73, 3, device=human_state.device, dtype=human_state.dtype)
            if bool(return_joints)
            else None
        )
        for gender_value, gender_name in ((0, "male"), (1, "female")):
            sample_indices = torch.nonzero(gender_code == gender_value, as_tuple=False).reshape(-1)
            if sample_indices.numel() == 0:
                continue

            smpl_model = self._get_smpl_model(gender_name, device=human_state.device)
            smpl_output = smpl_model(
                betas=betas[sample_indices],
                global_orient=global_orient[sample_indices],
                body_pose=body_pose[sample_indices],
                left_hand_pose=left_hand_pose[sample_indices],
                right_hand_pose=right_hand_pose[sample_indices],
                transl=transl[sample_indices],
                pose2rot=False,
                get_skin=True,
                return_full_pose=True,
                return_verts=True,
            )
            vertices[sample_indices] = smpl_output.vertices
            if joints is not None:
                if smpl_output.joints.shape[1] < 73:
                    raise RuntimeError(
                        "SMPL-H output must contain at least 73 joints, "
                        f"got {smpl_output.joints.shape[1]}."
                    )
                joints[sample_indices] = smpl_output.joints[:, :73]

        if joints is not None:
            return vertices, joints
        return vertices

    def _reconstruct_object_vertices(
        self,
        *,
        object_state: torch.Tensor,
        object_canonical_vertices: torch.Tensor,
    ) -> torch.Tensor:
        object_rotation = self._rotation_6d_to_matrix(object_state[:, :6])
        object_center = object_state[:, 6:9]
        return torch.matmul(object_canonical_vertices, object_rotation.transpose(1, 2)) + object_center.unsqueeze(1)

    @staticmethod
    def _limit_gradient_norm_per_sample(gradient: torch.Tensor, max_norm: float) -> torch.Tensor:
        if max_norm <= 0.0:
            return gradient
        gradient_flat = gradient.reshape(gradient.shape[0], -1)
        gradient_norm = torch.linalg.norm(gradient_flat, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(max_norm / gradient_norm, max=1.0)
        return gradient * scale.view(gradient.shape[0], *([1] * (gradient.dim() - 1)))

    @staticmethod
    def _expand_batch_tensor(value: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = value.to(device=device, dtype=dtype)
        if tensor.dim() > 0 and tensor.shape[0] == batch_size:
            return tensor
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

    def _object_vertices(self, object_state: torch.Tensor, context: MSFHOIInferenceGuidanceContext) -> Optional[torch.Tensor]:
        if context.object_canonical_vertices is None:
            return None
        canonical = context.object_canonical_vertices.to(device=object_state.device, dtype=object_state.dtype)
        if canonical.dim() == 2:
            canonical = canonical.unsqueeze(0).expand(object_state.shape[0], -1, -1)
        return self._reconstruct_object_vertices(object_state=object_state, object_canonical_vertices=canonical)

    def _contact_vertices(self, human_state: torch.Tensor, context: MSFHOIInferenceGuidanceContext) -> Optional[torch.Tensor]:
        if context.contact_vertex_indices is None:
            return None
        indexes = context.contact_vertex_indices.reshape(-1).to(device=human_state.device, dtype=torch.long)
        return self._reconstruct_human_vertex_subset(human_state=human_state, gender_code=context.gender_code, vertex_indices=indexes)[0]

    def _geometry_loss(self, model: MSFHOIFlowModel, states: Dict[str, torch.Tensor], context: MSFHOIInferenceGuidanceContext) -> Optional[torch.Tensor]:
        subject = self._contact_vertices(states["human"], context)
        objects = self._object_vertices(states["object"], context)
        if subject is None or objects is None:
            return None
        contact_scores = model.decode_contact_map(states["contact"], apply_sigmoid=True)[:, :subject.shape[1]]
        distances = torch.cdist(subject, objects).min(dim=-1).values
        return (distances * contact_scores).mean()

    def _sdf_loss(self, model: MSFHOIFlowModel, states: Dict[str, torch.Tensor], context: MSFHOIInferenceGuidanceContext, config: MSFHOIInferenceGuidanceConfig) -> Optional[torch.Tensor]:
        subject = self._contact_vertices(states["human"], context)
        grid = context.object_sdf_grid
        bounds_min = context.object_sdf_bounds_min
        bounds_max = context.object_sdf_bounds_max
        if subject is None or grid is None or bounds_min is None or bounds_max is None:
            return None
        contact_scores = model.decode_contact_map(states["contact"], apply_sigmoid=True)[:, :subject.shape[1]]
        object_rotation = self._rotation_6d_to_matrix(states["object"][:, :6])
        object_center = states["object"][:, 6:9]
        return _compute_sdf_guidance_loss(
            human_kp_690=subject,
            obj_translation=object_center,
            obj_rotation=object_rotation,
            sdf_grid=grid,
            grid_bounds=(bounds_min, bounds_max),
            contact_mask=contact_scores,
            weights={"w_pen": config.sdf_penetration_weight, "w_contact": config.sdf_contact_weight},
            contact_margin=config.sdf_contact_margin,
        )["loss_total"]

    def _balance_loss(self, states: Dict[str, torch.Tensor], context: MSFHOIInferenceGuidanceContext, config: MSFHOIInferenceGuidanceConfig) -> Optional[torch.Tensor]:
        if context.foot_vertex_indices is None or context.ground_z is None:
            return None
        foot_indexes = context.foot_vertex_indices.reshape(-1).to(device=states["human"].device, dtype=torch.long)
        foot_vertices = self._reconstruct_human_vertex_subset(human_state=states["human"], gender_code=context.gender_code, vertex_indices=foot_indexes)[0]
        object_rotation = self._rotation_6d_to_matrix(states["object"][:, :6])
        object_center = states["object"][:, 6:9]
        names = list(context.object_names or ["unknown"] * states["object"].shape[0])
        object_com = self._resolve_object_center_of_mass(object_names=names, object_rotation=object_rotation, object_translation=object_center)
        return _compute_balance_ground_guidance_loss(
            human_foot_vertices=foot_vertices,
            object_center_of_mass=object_com,
            ground_z=context.ground_z,
            object_names=names,
            ground_weight=config.balance_ground_weight_inner,
            balance_weight=config.balance_com_weight,
        )
    def _guidance_vector(self, model: MSFHOIFlowModel, states: Dict[str, torch.Tensor], mode: MSFHOIMode, context: MSFHOIInferenceGuidanceContext, config: MSFHOIInferenceGuidanceConfig) -> Dict[str, torch.Tensor]:
        if not any(float(value) > 0.0 for value in (config.geometry_weight, config.sdf_weight, config.balance_ground_weight)):
            return self._zero_guidance_vector(states)
        guided = {name: states[name].detach().clone().requires_grad_(bool(mode[index])) for index, name in enumerate(MSFHOI_MODALITY_NAMES)}
        losses: List[torch.Tensor] = []
        geometry_loss = self._geometry_loss(model, guided, context) if config.geometry_weight > 0.0 else None
        sdf_loss = self._sdf_loss(model, guided, context, config) if config.sdf_weight > 0.0 else None
        balance_loss = self._balance_loss(guided, context, config) if config.balance_ground_weight > 0.0 else None
        if geometry_loss is not None:
            losses.append(config.geometry_weight * geometry_loss)
        if sdf_loss is not None:
            losses.append(config.sdf_weight * sdf_loss)
        if balance_loss is not None:
            losses.append(config.balance_ground_weight * balance_loss)
        if not losses:
            return self._zero_guidance_vector(states)
        loss = torch.stack(losses).sum()
        active_names = [name for index, name in enumerate(MSFHOI_MODALITY_NAMES) if mode[index]]
        gradients = torch.autograd.grad(loss, [guided[name] for name in active_names], allow_unused=True, retain_graph=False)
        result = self._zero_guidance_vector(states)
        for name, gradient in zip(active_names, gradients):
            if gradient is not None:
                result[name] = -self._limit_gradient_norm_per_sample(gradient, config.geometry_max_gradient_norm)
        return result

    def _predict_vector_field(self, model: MSFHOIFlowModel, states: Dict[str, torch.Tensor], condition_states: Dict[str, torch.Tensor], class_onehot: torch.Tensor, pointnext_feature: torch.Tensor, mode: MSFHOIMode, t_scalar: torch.Tensor, inference_guidance: Optional[MSFHOIInferenceGuidanceConfig], guidance_context: Optional[MSFHOIInferenceGuidanceContext]) -> Dict[str, torch.Tensor]:
        batch_size = states["human"].shape[0]
        times = self._build_modality_times(mode=mode, t_scalar=t_scalar, batch_size=batch_size, device=states["human"].device, dtype=states["human"].dtype)
        vector_field = model.predict_vector_field(human_state=states["human"], object_state=states["object"], contact_state=states["contact"], class_onehot=class_onehot, pointnext_feature=pointnext_feature, human_time=times["human"], object_time=times["object"], contact_time=times["contact"])
        config = inference_guidance or MSFHOIInferenceGuidanceConfig()
        context = guidance_context or MSFHOIInferenceGuidanceContext()
        if guidance_context is not None and any(float(value) > 0.0 for value in (config.geometry_weight, config.sdf_weight, config.balance_ground_weight)):
            guidance = self._guidance_vector(model, states, mode, context, config)
            for name in MSFHOI_MODALITY_NAMES:
                vector_field[name] = vector_field[name] + guidance[name]
        return vector_field

    def _heun_step(self, model, states, condition_states, class_onehot, pointnext_feature, mode, t_value, t_next, inference_guidance, guidance_context):
        dt = t_next - t_value
        first_field = self._predict_vector_field(model, states, condition_states, class_onehot, pointnext_feature, mode, t_value, inference_guidance, guidance_context)
        predictor = {name: states[name] + dt * first_field[name] for name in MSFHOI_MODALITY_NAMES}
        self._ensure_conditioned_states(predictor, condition_states, mode)
        second_field = self._predict_vector_field(model, predictor, condition_states, class_onehot, pointnext_feature, mode, t_next, inference_guidance, guidance_context)
        next_states = {name: states[name] + 0.5 * dt * (first_field[name] + second_field[name]) for name in MSFHOI_MODALITY_NAMES}
        self._ensure_conditioned_states(next_states, condition_states, mode)
        return next_states

    @torch.no_grad()
    def integrate(self, *, model: MSFHOIFlowModel, initial_states: Dict[str, torch.Tensor], condition_states: Dict[str, torch.Tensor], class_onehot: torch.Tensor, pointnext_feature: torch.Tensor, mode: MSFHOIMode, inference_guidance: Optional[MSFHOIInferenceGuidanceConfig] = None, guidance_context: Optional[MSFHOIInferenceGuidanceContext] = None, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        del generator
        parsed_mode = normalize_MSFHOI_mode(mode)
        states = {name: initial_states[name].clone() for name in MSFHOI_MODALITY_NAMES}
        schedule = self.timestep_scheduler.build(device=states["human"].device, dtype=states["human"].dtype)
        with torch.enable_grad():
            for index in range(schedule.shape[0] - 1):
                t_value = schedule[index]
                t_next = schedule[index + 1]
                states = self._heun_step(model, states, condition_states, class_onehot, pointnext_feature, parsed_mode, t_value, t_next, inference_guidance, guidance_context)
                self._ensure_finite_states(states, label="integrated_states")
        return states
