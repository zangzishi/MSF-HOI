from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn

HUMAN_FEATURE_DIM = 325
OBJECT_FEATURE_DIM = 9
CONTACT_FEATURE_DIM = 128
CONTACT_MAP_FEATURE_DIM = 690
CLASS_CONDITION_DIM = 40
POINTNEXT_CONDITION_DIM = 1024
TOTAL_CONDITION_DIM = CLASS_CONDITION_DIM + POINTNEXT_CONDITION_DIM


@dataclass(frozen=True)
class MSFHOIModalityLayout:
    human_splits: Sequence[int] = (10, 6, 306, 3)
    object_splits: Sequence[int] = (6, 3)
    contact_splits: Sequence[int] = (128,)


def _linear_silu_linear(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class MSFHOIProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = _linear_silu_linear(in_dim, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MSFHOIHumanEncoder(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layout = MSFHOIModalityLayout()
        self.projection_shape = MSFHOIProjectionHead(10, hidden_dim, hidden_dim)
        self.projection_orient = MSFHOIProjectionHead(6, hidden_dim, hidden_dim)
        self.projection_pose = MSFHOIProjectionHead(306, hidden_dim, hidden_dim)
        self.projection_transl = MSFHOIProjectionHead(3, hidden_dim, hidden_dim)

    def forward(self, human_features: torch.Tensor) -> torch.Tensor:
        shape_params, global_orient, body_pose, global_transl = torch.split(
            human_features, self.layout.human_splits, dim=-1
        )
        return torch.stack(
            [
                self.projection_shape(shape_params),
                self.projection_orient(global_orient),
                self.projection_pose(body_pose),
                self.projection_transl(global_transl),
            ],
            dim=1,
        )


class MSFHOIObjectEncoder(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layout = MSFHOIModalityLayout()
        self.projection_orient = MSFHOIProjectionHead(6, hidden_dim, hidden_dim)
        self.projection_transl = MSFHOIProjectionHead(3, hidden_dim, hidden_dim)

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        global_orient, global_transl = torch.split(object_features, self.layout.object_splits, dim=-1)
        return torch.stack(
            [
                self.projection_orient(global_orient),
                self.projection_transl(global_transl),
            ],
            dim=1,
        )


class MSFHOIContactEncoder(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection_contact = MSFHOIProjectionHead(128, hidden_dim, hidden_dim)

    def forward(self, contact_features: torch.Tensor) -> torch.Tensor:
        return self.projection_contact(contact_features).unsqueeze(1)


class MSFHOIContactMapEncoder(nn.Module):
    

    def __init__(
        self,
        input_dim: int = CONTACT_MAP_FEATURE_DIM,
        hidden_dim: int = 256,
        latent_dim: int = CONTACT_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, contact_map: torch.Tensor) -> torch.Tensor:
        return self.encoder(contact_map)


class MSFHOIContactMapDecoder(nn.Module):
    

    def __init__(
        self,
        latent_dim: int = CONTACT_FEATURE_DIM,
        hidden_dim: int = 256,
        output_dim: int = CONTACT_MAP_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, contact_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(contact_latent)


class MSFHOIHumanDecoder(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layout = MSFHOIModalityLayout()
        self.decoder_shape = MSFHOIProjectionHead(hidden_dim, hidden_dim, 10)
        self.decoder_orient = MSFHOIProjectionHead(hidden_dim, hidden_dim, 6)
        self.decoder_pose = MSFHOIProjectionHead(hidden_dim, hidden_dim, 306)
        self.decoder_transl = MSFHOIProjectionHead(hidden_dim, hidden_dim, 3)

    def forward(self, human_tokens: torch.Tensor) -> torch.Tensor:
        shape_params = self.decoder_shape(human_tokens[:, 0])
        global_orient = self.decoder_orient(human_tokens[:, 1])
        body_pose = self.decoder_pose(human_tokens[:, 2])
        global_transl = self.decoder_transl(human_tokens[:, 3])
        return torch.cat([shape_params, global_orient, body_pose, global_transl], dim=-1)

    def split_components(self, human_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shape_params, global_orient, body_pose, global_transl = torch.split(
            human_features, self.layout.human_splits, dim=-1
        )
        return {
            "shape": shape_params,
            "orient": global_orient,
            "pose": body_pose,
            "transl": global_transl,
        }


class MSFHOIObjectDecoder(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layout = MSFHOIModalityLayout()
        self.decoder_orient = MSFHOIProjectionHead(hidden_dim, hidden_dim, 6)
        self.decoder_transl = MSFHOIProjectionHead(hidden_dim, hidden_dim, 3)

    def forward(self, object_tokens: torch.Tensor) -> torch.Tensor:
        global_orient = self.decoder_orient(object_tokens[:, 0])
        global_transl = self.decoder_transl(object_tokens[:, 1])
        return torch.cat([global_orient, global_transl], dim=-1)

    def split_components(self, object_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_orient, global_transl = torch.split(object_features, self.layout.object_splits, dim=-1)
        return {
            "orient": global_orient,
            "transl": global_transl,
        }


class MSFHOIContactDecoder(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.decoder_contact = MSFHOIProjectionHead(hidden_dim, hidden_dim, 128)

    def forward(self, contact_tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder_contact(contact_tokens[:, 0])


class MSFHOIConditionEncoder(nn.Module):
    

    def __init__(self, hidden_dim: int, input_dim: int = TOTAL_CONDITION_DIM) -> None:
        super().__init__()
        self.encoder = _linear_silu_linear(input_dim, hidden_dim * 2, hidden_dim)

    def forward(self, class_onehot: torch.Tensor, pointnext_feature: torch.Tensor) -> torch.Tensor:
        condition_vector = torch.cat([class_onehot, pointnext_feature], dim=-1)
        return self.encoder(condition_vector)


class MSFHOIModalityEncoders(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.human = MSFHOIHumanEncoder(hidden_dim)
        self.object = MSFHOIObjectEncoder(hidden_dim)
        self.contact = MSFHOIContactEncoder(hidden_dim)


class MSFHOIModalityDecoders(nn.Module):
    

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.human = MSFHOIHumanDecoder(hidden_dim)
        self.object = MSFHOIObjectDecoder(hidden_dim)
        self.contact = MSFHOIContactDecoder(hidden_dim)
