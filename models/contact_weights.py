from __future__ import annotations

from pathlib import Path

import torch


def load_contact_weights(model, checkpoint_path: str, map_location: torch.device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"contact checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint type: {type(checkpoint)}")
    model.contact_map_encoder.load_state_dict(checkpoint["model_enc"], strict=True)
    model.contact_map_decoder.load_state_dict(checkpoint["model_dec"], strict=True)
    return model
