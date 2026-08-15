from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch

from models.transformer import MSFHOIFlowModel
from solver import MSFHOIInferenceGuidanceConfig, MSFHOIInferenceGuidanceContext, MSFHOIODESolver, normalize_MSFHOI_mode


@dataclass
class MSFHOIPipelineOutput:
    mode: Sequence[int]
    states: Dict[str, torch.Tensor]
    decoded: Dict[str, Dict[str, torch.Tensor]]


class MSFHOIPipeline:
    def __init__(self, model: MSFHOIFlowModel, solver: Optional[MSFHOIODESolver] = None, device: Optional[Union[str, torch.device]] = None) -> None:
        self.model = model
        self.solver = solver or MSFHOIODESolver()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device).eval()

    def _dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _condition(self, value: Optional[torch.Tensor], batch_size: int, width: int, name: str, required: bool) -> torch.Tensor:
        if value is None:
            if required:
                raise ValueError(f"{name} condition is required")
            return torch.zeros(batch_size, width, device=self.device, dtype=self._dtype())
        state = value.to(device=self.device, dtype=self._dtype()).detach().clone()
        if state.shape != (batch_size, width):
            raise ValueError(f"{name} condition must have shape [B, {width}], got {tuple(state.shape)}")
        return state

    def _random_state(self, batch_size: int, width: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randn(batch_size, width, device=self.device, dtype=self._dtype(), generator=generator)

    @torch.no_grad()
    def sample(
        self,
        *,
        class_onehot: torch.Tensor,
        pointnext_feature: torch.Tensor,
        mode: Union[str, Sequence[int], torch.Tensor],
        human_condition: Optional[torch.Tensor] = None,
        object_condition: Optional[torch.Tensor] = None,
        contact_condition: Optional[torch.Tensor] = None,
        contact_map_condition: Optional[torch.Tensor] = None,
        ground_z: Optional[torch.Tensor] = None,
        object_canonical_vertices: Optional[torch.Tensor] = None,
        object_canonical_keypoints: Optional[torch.Tensor] = None,
        gender_code: Optional[torch.Tensor] = None,
        contact_vertex_indices: Optional[torch.Tensor] = None,
        foot_vertex_indices: Optional[torch.Tensor] = None,
        object_sdf_grid: Optional[torch.Tensor] = None,
        object_sdf_bounds_min: Optional[torch.Tensor] = None,
        object_sdf_bounds_max: Optional[torch.Tensor] = None,
        object_names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        geometry_weight: float = 0.0,
        sdf_weight: float = 0.0,
        balance_ground_weight: float = 0.0,
        sdf_penetration_weight: float = 10.0,
        sdf_contact_weight: float = 2.0,
        sdf_contact_margin: float = 0.05,
        balance_ground_inner_weight: float = 1.0,
        balance_com_weight: float = 1.0,
    ) -> MSFHOIPipelineOutput:
        parsed_mode = normalize_MSFHOI_mode(mode)
        dtype = self._dtype()
        class_onehot = class_onehot.to(device=self.device, dtype=dtype)
        pointnext_feature = pointnext_feature.to(device=self.device, dtype=dtype)
        if class_onehot.dim() != 2 or class_onehot.shape[1] != self.model.class_condition_dim:
            raise ValueError(f"class_onehot must have shape [B, {self.model.class_condition_dim}]")
        if pointnext_feature.dim() != 2 or pointnext_feature.shape[1] != self.model.pointnext_condition_dim:
            raise ValueError(f"pointnext_feature must have shape [B, {self.model.pointnext_condition_dim}]")
        batch_size = class_onehot.shape[0]
        if pointnext_feature.shape[0] != batch_size:
            raise ValueError("condition batch sizes must match")
        if contact_map_condition is not None:
            contact_map_condition = contact_map_condition.to(device=self.device, dtype=dtype)
            if contact_map_condition.shape != (batch_size, self.model.contact_map_feature_dim):
                raise ValueError(f"contact_map_condition must have shape [B, {self.model.contact_map_feature_dim}]")
            contact_condition = self.model.encode_contact_map(contact_map_condition).detach()
        condition_states = {
            "human": self._condition(human_condition, batch_size, self.model.human_feature_dim, "human", parsed_mode[0] == 0),
            "object": self._condition(object_condition, batch_size, self.model.object_feature_dim, "object", parsed_mode[1] == 0),
            "contact": self._condition(contact_condition, batch_size, self.model.contact_feature_dim, "contact", parsed_mode[2] == 0),
        }
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        initial_states = {
            name: condition_states[name] if not parsed_mode[index] else self._random_state(batch_size, condition_states[name].shape[1], generator)
            for index, name in enumerate(("human", "object", "contact"))
        }
        to_device = lambda value, value_dtype=dtype: None if value is None else value.to(device=self.device, dtype=value_dtype)
        guidance = MSFHOIInferenceGuidanceConfig(
            geometry_weight=float(geometry_weight), sdf_weight=float(sdf_weight), balance_ground_weight=float(balance_ground_weight),
            sdf_penetration_weight=float(sdf_penetration_weight), sdf_contact_weight=float(sdf_contact_weight), sdf_contact_margin=float(sdf_contact_margin),
            balance_ground_weight_inner=float(balance_ground_inner_weight), balance_com_weight=float(balance_com_weight),
        )
        context = MSFHOIInferenceGuidanceContext(
            ground_z=to_device(ground_z), object_canonical_vertices=to_device(object_canonical_vertices), object_canonical_keypoints=to_device(object_canonical_keypoints),
            gender_code=to_device(gender_code, torch.long), contact_vertex_indices=to_device(contact_vertex_indices, torch.long), foot_vertex_indices=to_device(foot_vertex_indices, torch.long),
            object_sdf_grid=to_device(object_sdf_grid), object_sdf_bounds_min=to_device(object_sdf_bounds_min), object_sdf_bounds_max=to_device(object_sdf_bounds_max), object_names=object_names,
        )
        solved_states = self.solver.integrate(
            model=self.model, initial_states=initial_states, condition_states=condition_states,
            class_onehot=class_onehot, pointnext_feature=pointnext_feature, mode=parsed_mode,
            inference_guidance=guidance, guidance_context=context, generator=generator,
        )
        contact_scores = self.model.decode_contact_map(solved_states["contact"], apply_sigmoid=False)
        decoded = {
            "human": self.model.split_human_parameters(solved_states["human"]),
            "object": self.model.split_object_parameters(solved_states["object"]),
            "contact": {"latent": solved_states["contact"], "map_scores": contact_scores},
        }
        return MSFHOIPipelineOutput(mode=parsed_mode, states=solved_states, decoded=decoded)
