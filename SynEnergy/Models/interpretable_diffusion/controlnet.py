"""ControlNet side branch: mirrored backbone Encoder + z_a hint -> zero-init controls."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from Models.interpretable_diffusion.transformer import Encoder


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-init all parameters so control branch starts neutral."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ZControlNet(nn.Module):
    """
    Trainable copy of the backbone Encoder (weights synced after backbone load).

    Input: x_emb (noisy x_t, same as frozen encoder) + global z_a hint + optional
    AE decode dense map (p_hat, v_hat per timestep).
    Conditioning: label_emb (tract), same adaLN path as the frozen encoder.
    Output: per-layer (B, T, d_model) controls through zero-init linear layers.
    """

    def __init__(
        self,
        z_a_dim: int = 64,
        seq_length: int = 180,
        d_model: int = 64,
        n_layer_enc: int = 2,
        n_heads: int = 4,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        mlp_hidden_times: int = 4,
        use_decode_map: bool = True,
        decode_map_channels: int = 2,
        use_p_hat_gate: bool = False,
        gate_z_hint: bool = False,
        gate_control_output: bool = True,
        p_hat_gate_mode: str = "linear",
        p_hat_gate_power: float = 1.0,
        p_hat_gate_floor: float = 0.0,
        p_hat_gate_detach: bool = True,
        p_hat_gate_soft_tau: float = 0.1,
        p_hat_gate_soft_k: float = 20.0,
    ):
        super().__init__()
        self.z_a_dim = z_a_dim
        self.seq_length = seq_length
        self.d_model = d_model
        self.use_decode_map = bool(use_decode_map)
        self.decode_map_channels = int(decode_map_channels)
        self.use_p_hat_gate = bool(use_p_hat_gate)
        self.gate_z_hint = bool(gate_z_hint)
        self.gate_control_output = bool(gate_control_output)
        self.p_hat_gate_mode = str(p_hat_gate_mode)
        self.p_hat_gate_power = float(p_hat_gate_power)
        self.p_hat_gate_floor = float(p_hat_gate_floor)
        self.p_hat_gate_detach = bool(p_hat_gate_detach)
        self.p_hat_gate_soft_tau = float(p_hat_gate_soft_tau)
        self.p_hat_gate_soft_k = float(p_hat_gate_soft_k)
        if self.use_p_hat_gate and not self.use_decode_map:
            raise ValueError("use_p_hat_gate=True requires use_decode_map=True (AE decoder).")
        if self.use_p_hat_gate and not (self.gate_z_hint or self.gate_control_output):
            raise ValueError(
                "use_p_hat_gate=True requires gate_z_hint and/or gate_control_output."
            )
        self.ae_decoder = None

        self.z_norm = nn.LayerNorm(z_a_dim)
        self.z_proj = nn.Sequential(
            nn.Linear(z_a_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_length * d_model),
        )

        if self.use_decode_map:
            self.map_proj = nn.Sequential(
                nn.Linear(self.decode_map_channels, d_model),
                nn.GELU(),
                zero_module(nn.Linear(d_model, d_model)),
            )

        self.control_encoder = Encoder(
            n_layer=n_layer_enc,
            n_embd=d_model,
            n_head=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
        )
        self.zero_convs = nn.ModuleList(
            [zero_module(nn.Linear(d_model, d_model)) for _ in range(n_layer_enc)]
        )
        self.null_z = nn.Parameter(torch.zeros(1, z_a_dim))

    def register_ae_decoder(self, ae_decoder: nn.Module) -> None:
        """Attach frozen AE for z_a -> dense (p_hat, v_hat) decode map."""
        self.ae_decoder = ae_decoder
        self.ae_decoder.eval()
        for param in self.ae_decoder.parameters():
            param.requires_grad = False

    def _decode_p_hat_v_hat(self, z_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, D) -> p_hat, v_hat each (B, T, 1) using frozen AE.decode."""
        if self.ae_decoder is None:
            raise RuntimeError(
                "AE decoder is not registered on ZControlNet."
            )
        with torch.no_grad():
            _, _, p_hat, v_hat = self.ae_decoder.decode(z_a)
        return p_hat, v_hat

    def _build_p_hat_gate(self, p_hat: torch.Tensor) -> torch.Tensor:
        """Map AE soft gate p_hat (B, T, 1) to control gate in [floor, 1]."""
        gate = p_hat
        if self.p_hat_gate_detach:
            gate = gate.detach()

        mode = self.p_hat_gate_mode
        if mode == "linear":
            gate = gate.clamp(0.0, 1.0)
        elif mode == "power":
            gate = gate.clamp(0.0, 1.0).pow(self.p_hat_gate_power)
        elif mode == "soft":
            gate = torch.sigmoid(
                self.p_hat_gate_soft_k * (gate - self.p_hat_gate_soft_tau)
            )
        else:
            raise ValueError(
                f"p_hat_gate_mode must be linear, power, or soft; got {mode!r}"
            )

        floor = self.p_hat_gate_floor
        if floor > 0.0:
            gate = floor + (1.0 - floor) * gate
        return gate

    def _dense_map_from_p_v(
        self, p_hat: torch.Tensor, v_hat: torch.Tensor
    ) -> torch.Tensor:
        if self.decode_map_channels == 1:
            return p_hat
        if self.decode_map_channels == 2:
            return torch.cat([p_hat, v_hat], dim=-1)
        raise ValueError(
            f"decode_map_channels must be 1 or 2, got {self.decode_map_channels}"
        )

    def mirror_from_backbone(self, backbone_encoder: nn.Module) -> None:
        """Copy frozen backbone Encoder weights into the trainable control encoder."""
        self.control_encoder.load_state_dict(backbone_encoder.state_dict(), strict=True)

    def forward(
        self,
        z_a: torch.Tensor,
        t: torch.Tensor,
        x_emb: torch.Tensor,
        label_emb: Optional[torch.Tensor] = None,
        z_a_dropout: float = 0.0,
        padding_masks: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        z_a: (B, D) or (B, D, 1)
        t: (B,) long
        x_emb: (B, T, d_model) backbone encoder input (emb + pos_enc of noisy x_t)
        label_emb: (B, d_model) tract condition, same as frozen encoder adaLN
        Returns list of length n_layer_enc with tensors (B, T, d_model).
        """
        if z_a.ndim == 3:
            z_a = z_a.squeeze(-1)
        b = z_a.shape[0]
        if x_emb.shape[0] != b:
            raise ValueError(
                f"x_emb batch {x_emb.shape[0]} must match z_a batch {b}"
            )
        if x_emb.shape[1] != self.seq_length or x_emb.shape[2] != self.d_model:
            raise ValueError(
                f"x_emb expected (B, {self.seq_length}, {self.d_model}), got {tuple(x_emb.shape)}"
            )

        if self.training and z_a_dropout > 0.0:
            drop = torch.rand(b, device=z_a.device) < z_a_dropout
            if drop.any():
                z_a = z_a.clone()
                z_a[drop] = self.null_z.expand(drop.sum(), -1)

        z_raw = z_a
        z_a = self.z_norm(z_a)
        z_hint = self.z_proj(z_a).view(b, self.seq_length, self.d_model)

        p_hat_gate: Optional[torch.Tensor] = None
        if self.use_decode_map:
            p_hat, v_hat = self._decode_p_hat_v_hat(z_raw)
            if self.use_p_hat_gate:
                p_hat_gate = self._build_p_hat_gate(p_hat)

        if p_hat_gate is not None and self.gate_z_hint:
            z_hint = z_hint * p_hat_gate

        h = x_emb + z_hint
        if self.use_decode_map:
            dense_map = self._dense_map_from_p_v(p_hat, v_hat)
            h = h + self.map_proj(dense_map)

        controls: List[torch.Tensor] = []
        x = h
        for block_idx in range(len(self.control_encoder.blocks)):
            x, _ = self.control_encoder.blocks[block_idx](
                x, t, mask=padding_masks, label_emb=label_emb
            )
            controls.append(self.zero_convs[block_idx](x))

        if p_hat_gate is not None and self.gate_control_output:
            controls = [c * p_hat_gate for c in controls]
        return controls
