from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MSFHOIJointAttentionBlock(nn.Module):
    

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        human_ffn_dim: int = 2048,
        object_ffn_dim: int = 2048,
        contact_ffn_dim: int = 2048,
        joint_attn_proj_mode: str = "per_modality",
        dropout: float = 0.0,
        qk_norm: str = "layer_norm",
        adaln_gate_init: float = 0.0,
        human_adaln_gate_init: Optional[float] = None,
        object_adaln_gate_init: Optional[float] = None,
        contact_adaln_gate_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        if qk_norm not in {"none", "layer_norm"}:
            raise ValueError("qk_norm must be one of {'none', 'layer_norm'}.")
        if joint_attn_proj_mode not in {"shared", "per_modality"}:
            raise ValueError("joint_attn_proj_mode must be one of {'shared', 'per_modality'}.")
        adaln_gate_init = float(adaln_gate_init)
        if not math.isfinite(adaln_gate_init) or adaln_gate_init < 0.0:
            raise ValueError("adaln_gate_init must be finite and >= 0.")

        def _resolve_modality_gate_init(value: Optional[float], name: str) -> float:
            if value is None:
                return adaln_gate_init
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.")
            return value

        human_adaln_gate_init = _resolve_modality_gate_init(
            human_adaln_gate_init,
            "human_adaln_gate_init",
        )
        object_adaln_gate_init = _resolve_modality_gate_init(
            object_adaln_gate_init,
            "object_adaln_gate_init",
        )
        contact_adaln_gate_init = _resolve_modality_gate_init(
            contact_adaln_gate_init,
            "contact_adaln_gate_init",
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = float(dropout)
        self.qk_norm = qk_norm
        self.joint_attn_proj_mode = joint_attn_proj_mode
        self.adaln_gate_init = adaln_gate_init
        self.human_adaln_gate_init = human_adaln_gate_init
        self.object_adaln_gate_init = object_adaln_gate_init
        self.contact_adaln_gate_init = contact_adaln_gate_init

        if self.joint_attn_proj_mode == "shared":
            self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.human_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            self.object_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            self.contact_qkv = nn.Linear(hidden_dim, hidden_dim * 3)

            self.human_out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.object_out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.contact_out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm == "layer_norm" else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm == "layer_norm" else nn.Identity()

        self.human_norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.object_norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.contact_norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.human_norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.object_norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.contact_norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.human_ffn = nn.Sequential(
            nn.Linear(hidden_dim, human_ffn_dim),
            nn.SiLU(),
            nn.Linear(human_ffn_dim, hidden_dim),
        )
        self.object_ffn = nn.Sequential(
            nn.Linear(hidden_dim, object_ffn_dim),
            nn.SiLU(),
            nn.Linear(object_ffn_dim, hidden_dim),
        )
        self.contact_ffn = nn.Sequential(
            nn.Linear(hidden_dim, contact_ffn_dim),
            nn.SiLU(),
            nn.Linear(contact_ffn_dim, hidden_dim),
        )

        self.human_modulation = nn.Linear(hidden_dim, hidden_dim * 6)
        self.object_modulation = nn.Linear(hidden_dim, hidden_dim * 6)
        self.contact_modulation = nn.Linear(hidden_dim, hidden_dim * 6)

        self._init_modulation()

    def _init_modulation(self) -> None:
        nn.init.zeros_(self.human_modulation.weight)
        nn.init.zeros_(self.human_modulation.bias)
        nn.init.zeros_(self.object_modulation.weight)
        nn.init.zeros_(self.object_modulation.bias)
        nn.init.zeros_(self.contact_modulation.weight)
        nn.init.zeros_(self.contact_modulation.bias)
        if self.adaln_gate_init != 0.0:
            hidden_dim = self.hidden_dim
            with torch.no_grad():
                for modulation in (
                    self.human_modulation,
                    self.object_modulation,
                    self.contact_modulation,
                ):
                    modulation.bias[2 * hidden_dim : 3 * hidden_dim].fill_(self.adaln_gate_init)
                    modulation.bias[5 * hidden_dim : 6 * hidden_dim].fill_(self.adaln_gate_init)

        hidden_dim = self.hidden_dim
        with torch.no_grad():
            modality_gates = (
                (self.human_modulation, self.human_adaln_gate_init),
                (self.object_modulation, self.object_adaln_gate_init),
                (self.contact_modulation, self.contact_adaln_gate_init),
            )
            for modulation, gate_init in modality_gates:
                modulation.bias[2 * hidden_dim : 3 * hidden_dim].fill_(gate_init)
                modulation.bias[5 * hidden_dim : 6 * hidden_dim].fill_(gate_init)

    def _split_modulation(self, modulation: torch.Tensor) -> Dict[str, torch.Tensor]:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(6, dim=-1)
        return {
            "shift_attn": shift_attn,
            "scale_attn": scale_attn,
            "gate_attn": gate_attn,
            "shift_ffn": shift_ffn,
            "scale_ffn": scale_ffn,
            "gate_ffn": gate_ffn,
        }

    def _adaptive_norm(
        self,
        x: torch.Tensor,
        norm_layer: nn.LayerNorm,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        x = norm_layer(x)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _project_qkv(self, tokens: torch.Tensor, projection: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = tokens.shape
        qkv = projection(tokens)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv.unbind(dim=0)

    def _shared_joint_attention(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = joint_tokens.shape
        qkv = self.qkv(joint_tokens)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, self.hidden_dim)
        return self.out_proj(attention_output)

    def _joint_attention(
        self,
        human_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        contact_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        human_q, human_k, human_v = self._project_qkv(human_tokens, self.human_qkv)
        object_q, object_k, object_v = self._project_qkv(object_tokens, self.object_qkv)
        contact_q, contact_k, contact_v = self._project_qkv(contact_tokens, self.contact_qkv)

        q = torch.cat([human_q, object_q, contact_q], dim=2)
        k = torch.cat([human_k, object_k, contact_k], dim=2)
        v = torch.cat([human_v, object_v, contact_v], dim=2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attention_output = attention_output.transpose(1, 2).contiguous()
        batch_size = attention_output.shape[0]
        total_length = attention_output.shape[1]
        attention_output = attention_output.view(batch_size, total_length, self.hidden_dim)

        human_length = human_tokens.shape[1]
        object_length = object_tokens.shape[1]
        contact_length = contact_tokens.shape[1]
        human_output, object_output, contact_output = torch.split(
            attention_output,
            [human_length, object_length, contact_length],
            dim=1,
        )
        return (
            self.human_out_proj(human_output),
            self.object_out_proj(object_output),
            self.contact_out_proj(contact_output),
        )

    def _modality_update(
        self,
        x: torch.Tensor,
        attn_update: torch.Tensor,
        norm_ffn: nn.LayerNorm,
        ffn: nn.Module,
        mod_parts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = x + mod_parts["gate_attn"].unsqueeze(1) * attn_update
        x_ffn = self._adaptive_norm(
            x,
            norm_layer=norm_ffn,
            shift=mod_parts["shift_ffn"],
            scale=mod_parts["scale_ffn"],
        )
        x = x + mod_parts["gate_ffn"].unsqueeze(1) * ffn(x_ffn)
        return x

    def forward(
        self,
        human_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        contact_tokens: torch.Tensor,
        human_temb: torch.Tensor,
        object_temb: torch.Tensor,
        contact_temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        human_mod = self._split_modulation(self.human_modulation(human_temb))
        object_mod = self._split_modulation(self.object_modulation(object_temb))
        contact_mod = self._split_modulation(self.contact_modulation(contact_temb))

        human_attn_in = self._adaptive_norm(
            human_tokens,
            norm_layer=self.human_norm_attn,
            shift=human_mod["shift_attn"],
            scale=human_mod["scale_attn"],
        )
        object_attn_in = self._adaptive_norm(
            object_tokens,
            norm_layer=self.object_norm_attn,
            shift=object_mod["shift_attn"],
            scale=object_mod["scale_attn"],
        )
        contact_attn_in = self._adaptive_norm(
            contact_tokens,
            norm_layer=self.contact_norm_attn,
            shift=contact_mod["shift_attn"],
            scale=contact_mod["scale_attn"],
        )

        if self.joint_attn_proj_mode == "shared":
            human_length = human_tokens.shape[1]
            object_length = object_tokens.shape[1]
            contact_length = contact_tokens.shape[1]
            joint_tokens = torch.cat([human_attn_in, object_attn_in, contact_attn_in], dim=1)
            joint_attention_output = self._shared_joint_attention(joint_tokens)
            human_attn_out, object_attn_out, contact_attn_out = torch.split(
                joint_attention_output,
                [human_length, object_length, contact_length],
                dim=1,
            )
        else:
            human_attn_out, object_attn_out, contact_attn_out = self._joint_attention(
                human_tokens=human_attn_in,
                object_tokens=object_attn_in,
                contact_tokens=contact_attn_in,
            )

        human_tokens = self._modality_update(
            x=human_tokens,
            attn_update=human_attn_out,
            norm_ffn=self.human_norm_ffn,
            ffn=self.human_ffn,
            mod_parts=human_mod,
        )
        object_tokens = self._modality_update(
            x=object_tokens,
            attn_update=object_attn_out,
            norm_ffn=self.object_norm_ffn,
            ffn=self.object_ffn,
            mod_parts=object_mod,
        )
        contact_tokens = self._modality_update(
            x=contact_tokens,
            attn_update=contact_attn_out,
            norm_ffn=self.contact_norm_ffn,
            ffn=self.contact_ffn,
            mod_parts=contact_mod,
        )

        return human_tokens, object_tokens, contact_tokens
