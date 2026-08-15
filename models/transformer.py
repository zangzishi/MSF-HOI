from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn

from .attention import MSFHOIJointAttentionBlock
from .encoders import (
    MSFHOIContactMapDecoder,
    MSFHOIContactMapEncoder,
    MSFHOIConditionEncoder,
    MSFHOIModalityDecoders,
    MSFHOIModalityEncoders,
    CLASS_CONDITION_DIM,
    CONTACT_FEATURE_DIM,
    CONTACT_MAP_FEATURE_DIM,
    HUMAN_FEATURE_DIM,
    OBJECT_FEATURE_DIM,
    POINTNEXT_CONDITION_DIM,
)


def _validate_adaln_gate_init(value: object) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("adaln_gate_init must be finite and >= 0.")
    return value


def _resolve_modality_adaln_gate_init(
    value: object,
    name: str,
) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.")
    return value


def resolve_MSFHOI_model_kwargs_from_config(config: Dict[str, object]) -> Dict[str, object]:
    adaln_gate_init = _validate_adaln_gate_init(config["adaln_gate_init"])
    return {
        "hidden_dim": int(config["hidden_dim"]),
        "time_embedding_dim": int(config["time_embedding_dim"]),
        "time_embedding_input_scale": float(config["time_embedding_input_scale"]),
        "condition_context_normalization": str(config["condition_context_normalization"]),
        "num_layers": int(config["num_layers"]),
        "num_heads": int(config["num_heads"]),
        "human_ffn_dim": int(config["human_ffn_dim"]),
        "object_ffn_dim": int(config["object_ffn_dim"]),
        "contact_ffn_dim": int(config["contact_ffn_dim"]),
        "joint_attn_proj_mode": str(config["joint_attn_proj_mode"]),
        "dropout": float(config["dropout"]),
        "qk_norm": str(config["qk_norm"]),
        "contact_map_hidden_dim": int(config["contact_map_hidden_dim"]),
        "adaln_gate_init": adaln_gate_init,
        "human_adaln_gate_init": _resolve_modality_adaln_gate_init(config["human_adaln_gate_init"], "human_adaln_gate_init"),
        "object_adaln_gate_init": _resolve_modality_adaln_gate_init(config["object_adaln_gate_init"], "object_adaln_gate_init"),
        "contact_adaln_gate_init": _resolve_modality_adaln_gate_init(config["contact_adaln_gate_init"], "contact_adaln_gate_init"),
    }

class MSFHOISinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            timesteps = timesteps.reshape(-1)

        half_dim = self.embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0,
            end=half_dim,
            device=timesteps.device,
            dtype=timesteps.dtype,
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        arguments = timesteps.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)

        if self.embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1), mode="constant", value=0.0)
        return embedding


class MSFHOITransformerModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 312,
        num_layers: int = 8,
        num_heads: int = 8,
        human_ffn_dim: int = 2048,
        object_ffn_dim: int = 2048,
        contact_ffn_dim: int = 2048,
        joint_attn_proj_mode: str = "per_modality",
        dropout: float = 0.0,
        qk_norm: str = "layer_norm",
        adaln_gate_init: float = 0.0,
        human_adaln_gate_init: float = 0.0,
        object_adaln_gate_init: float = 0.0,
        contact_adaln_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        adaln_gate_init = _validate_adaln_gate_init(adaln_gate_init)
        human_adaln_gate_init = _resolve_modality_adaln_gate_init(
            human_adaln_gate_init,
            "human_adaln_gate_init",
        )
        object_adaln_gate_init = _resolve_modality_adaln_gate_init(
            object_adaln_gate_init,
            "object_adaln_gate_init",
        )
        contact_adaln_gate_init = _resolve_modality_adaln_gate_init(
            contact_adaln_gate_init,
            "contact_adaln_gate_init",
        )
        self.adaln_gate_init = adaln_gate_init
        self.human_adaln_gate_init = human_adaln_gate_init
        self.object_adaln_gate_init = object_adaln_gate_init
        self.contact_adaln_gate_init = contact_adaln_gate_init
        self.joint_attn_proj_mode = joint_attn_proj_mode
        self.blocks = nn.ModuleList(
            [
                MSFHOIJointAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    human_ffn_dim=human_ffn_dim,
                    object_ffn_dim=object_ffn_dim,
                    contact_ffn_dim=contact_ffn_dim,
                    joint_attn_proj_mode=joint_attn_proj_mode,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    adaln_gate_init=adaln_gate_init,
                    human_adaln_gate_init=human_adaln_gate_init,
                    object_adaln_gate_init=object_adaln_gate_init,
                    contact_adaln_gate_init=contact_adaln_gate_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_human_norm = nn.LayerNorm(hidden_dim)
        self.final_object_norm = nn.LayerNorm(hidden_dim)
        self.final_contact_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        human_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        contact_tokens: torch.Tensor,
        human_temb: torch.Tensor,
        object_temb: torch.Tensor,
        contact_temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            human_tokens, object_tokens, contact_tokens = block(
                human_tokens=human_tokens,
                object_tokens=object_tokens,
                contact_tokens=contact_tokens,
                human_temb=human_temb,
                object_temb=object_temb,
                contact_temb=contact_temb,
            )

        human_tokens = self.final_human_norm(human_tokens)
        object_tokens = self.final_object_norm(object_tokens)
        contact_tokens = self.final_contact_norm(contact_tokens)
        return human_tokens, object_tokens, contact_tokens


class MSFHOIFlowModel(nn.Module):
    

    human_feature_dim: int = HUMAN_FEATURE_DIM
    object_feature_dim: int = OBJECT_FEATURE_DIM
    contact_feature_dim: int = CONTACT_FEATURE_DIM
    contact_map_feature_dim: int = CONTACT_MAP_FEATURE_DIM
    class_condition_dim: int = CLASS_CONDITION_DIM
    pointnext_condition_dim: int = POINTNEXT_CONDITION_DIM

    def __init__(
        self,
        hidden_dim: int = 312,
        time_embedding_dim: int = 128,
        time_embedding_input_scale: float = 1.0,
        condition_context_normalization: str = "none",
        num_layers: int = 8,
        num_heads: int = 8,
        human_ffn_dim: int = 2048,
        object_ffn_dim: int = 2048,
        contact_ffn_dim: int = 2048,
        joint_attn_proj_mode: str = "per_modality",
        dropout: float = 0.0,
        qk_norm: str = "layer_norm",
        contact_map_hidden_dim: int = 256,
        adaln_gate_init: float = 0.0,
        human_adaln_gate_init: float = 0.0,
        object_adaln_gate_init: float = 0.0,
        contact_adaln_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        adaln_gate_init = _validate_adaln_gate_init(adaln_gate_init)
        human_adaln_gate_init = _resolve_modality_adaln_gate_init(
            human_adaln_gate_init,
            "human_adaln_gate_init",
        )
        object_adaln_gate_init = _resolve_modality_adaln_gate_init(
            object_adaln_gate_init,
            "object_adaln_gate_init",
        )
        contact_adaln_gate_init = _resolve_modality_adaln_gate_init(
            contact_adaln_gate_init,
            "contact_adaln_gate_init",
        )
        self.hidden_dim = hidden_dim
        self.joint_attn_proj_mode = joint_attn_proj_mode
        self.adaln_gate_init = adaln_gate_init
        self.human_adaln_gate_init = human_adaln_gate_init
        self.object_adaln_gate_init = object_adaln_gate_init
        self.contact_adaln_gate_init = contact_adaln_gate_init
        if not math.isfinite(float(time_embedding_input_scale)) or float(time_embedding_input_scale) <= 0.0:
            raise ValueError("time_embedding_input_scale must be finite and > 0.")
        self.time_embedding_input_scale = float(time_embedding_input_scale)
        if condition_context_normalization not in {"none", "layer_norm"}:
            raise ValueError(
                "condition_context_normalization must be one of {'none', 'layer_norm'}."
            )
        self.condition_context_normalization = condition_context_normalization

        self.encoders = MSFHOIModalityEncoders(hidden_dim=hidden_dim)
        self.decoders = MSFHOIModalityDecoders(hidden_dim=hidden_dim)
        self.condition_encoder = MSFHOIConditionEncoder(hidden_dim=hidden_dim)
        self.condition_context_normalizer = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            if condition_context_normalization == "layer_norm"
            else nn.Identity()
        )
        self.contact_map_encoder = MSFHOIContactMapEncoder(
            input_dim=self.contact_map_feature_dim,
            hidden_dim=contact_map_hidden_dim,
            latent_dim=self.contact_feature_dim,
        )
        self.contact_map_decoder = MSFHOIContactMapDecoder(
            latent_dim=self.contact_feature_dim,
            hidden_dim=contact_map_hidden_dim,
            output_dim=self.contact_map_feature_dim,
        )

        self.time_embedding = MSFHOISinusoidalTimeEmbedding(embedding_dim=time_embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.human_context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.contact_context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.human_token_embedding = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.object_token_embedding = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.contact_token_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        self.transformer = MSFHOITransformerModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            human_ffn_dim=human_ffn_dim,
            object_ffn_dim=object_ffn_dim,
            contact_ffn_dim=contact_ffn_dim,
            joint_attn_proj_mode=joint_attn_proj_mode,
            dropout=dropout,
            qk_norm=qk_norm,
            adaln_gate_init=adaln_gate_init,
            human_adaln_gate_init=human_adaln_gate_init,
            object_adaln_gate_init=object_adaln_gate_init,
            contact_adaln_gate_init=contact_adaln_gate_init,
        )

    def _project_time(self, time_value: torch.Tensor) -> torch.Tensor:
        if self.time_embedding_input_scale != 1.0:
            time_value = time_value * self.time_embedding_input_scale
        return self.time_projection(self.time_embedding(time_value))

    def _modality_context(
        self,
        time_value: torch.Tensor,
        condition_context: torch.Tensor,
        context_projection: nn.Module,
    ) -> torch.Tensor:
        time_context = self._project_time(time_value)
        merged_context = torch.cat([time_context, condition_context], dim=-1)
        return context_projection(merged_context)

    def encode_modalities(
        self,
        human_state: torch.Tensor,
        object_state: torch.Tensor,
        contact_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        human_tokens = self.encoders.human(human_state) + self.human_token_embedding.unsqueeze(0)
        object_tokens = self.encoders.object(object_state) + self.object_token_embedding.unsqueeze(0)
        contact_tokens = self.encoders.contact(contact_state) + self.contact_token_embedding.unsqueeze(0)
        return human_tokens, object_tokens, contact_tokens

    def decode_modalities(
        self,
        human_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        contact_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "human": self.decoders.human(human_tokens),
            "object": self.decoders.object(object_tokens),
            "contact": self.decoders.contact(contact_tokens),
        }

    def predict_vector_field(
        self,
        human_state: torch.Tensor,
        object_state: torch.Tensor,
        contact_state: torch.Tensor,
        class_onehot: torch.Tensor,
        pointnext_feature: torch.Tensor,
        human_time: torch.Tensor,
        object_time: torch.Tensor,
        contact_time: torch.Tensor,
        drop_condition_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        condition_context = self.condition_encoder(class_onehot, pointnext_feature)
        condition_context = self.condition_context_normalizer(condition_context)
        if drop_condition_context:
            condition_context = torch.zeros_like(condition_context)

        human_temb = self._modality_context(
            time_value=human_time,
            condition_context=condition_context,
            context_projection=self.human_context_projection,
        )
        object_temb = self._modality_context(
            time_value=object_time,
            condition_context=condition_context,
            context_projection=self.object_context_projection,
        )
        contact_temb = self._modality_context(
            time_value=contact_time,
            condition_context=condition_context,
            context_projection=self.contact_context_projection,
        )

        human_tokens, object_tokens, contact_tokens = self.encode_modalities(
            human_state=human_state,
            object_state=object_state,
            contact_state=contact_state,
        )

        human_tokens, object_tokens, contact_tokens = self.transformer(
            human_tokens=human_tokens,
            object_tokens=object_tokens,
            contact_tokens=contact_tokens,
            human_temb=human_temb,
            object_temb=object_temb,
            contact_temb=contact_temb,
        )

        return self.decode_modalities(
            human_tokens=human_tokens,
            object_tokens=object_tokens,
            contact_tokens=contact_tokens,
        )

    def forward(
        self,
        human_state: torch.Tensor,
        object_state: torch.Tensor,
        contact_state: torch.Tensor,
        class_onehot: torch.Tensor,
        pointnext_feature: torch.Tensor,
        human_time: torch.Tensor,
        object_time: torch.Tensor,
        contact_time: torch.Tensor,
        drop_condition_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.predict_vector_field(
            human_state=human_state,
            object_state=object_state,
            contact_state=contact_state,
            class_onehot=class_onehot,
            pointnext_feature=pointnext_feature,
            human_time=human_time,
            object_time=object_time,
            contact_time=contact_time,
            drop_condition_context=drop_condition_context,
        )

    def split_human_parameters(self, human_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoders.human.split_components(human_features)

    def split_object_parameters(self, object_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoders.object.split_components(object_features)

    def encode_contact_map(self, contact_map: torch.Tensor) -> torch.Tensor:
        return self.contact_map_encoder(contact_map)

    def decode_contact_map(
        self,
        contact_latent: torch.Tensor,
        apply_sigmoid: bool = False,
    ) -> torch.Tensor:
        contact_map_scores = self.contact_map_decoder(contact_latent)
        if apply_sigmoid:
            return torch.sigmoid(contact_map_scores)
        return contact_map_scores
