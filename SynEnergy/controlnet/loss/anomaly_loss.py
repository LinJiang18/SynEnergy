"""ControlNet anomaly loss in normalized data space."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from controlnet.ae.residual_learner import (
    ZeroAwareResidualAutoencoder,
    filtered_residual_from_data_torch,
)


def build_filter_cfg(controlnet_cfg: dict, seq_length: Optional[int] = None) -> dict:
    cfg = {
        "num_days": controlnet_cfg.get("num_days", 30),
        "slots_per_day": controlnet_cfg.get("slots_per_day", 6),
        "residual_eps": controlnet_cfg.get("residual_eps", 1e-6),
        "min_scale": controlnet_cfg.get("min_scale", 1.0),
        "use_mad_consistency": controlnet_cfg.get("use_mad_consistency", True),
        "filter_threshold": controlnet_cfg.get("filter_threshold", 0.12),
        "filter_mode": controlnet_cfg.get("filter_mode", "soft"),
        "filter_gamma": controlnet_cfg.get("filter_gamma", 50.0),
        "occurrence_threshold": controlnet_cfg.get("occurrence_threshold", 0.12),
        "loss_eps": controlnet_cfg.get("loss_eps", 1e-6),
    }
    if seq_length is not None:
        cfg["seq_len"] = int(seq_length)
    return cfg


def _split_region_masks(
    weight_soft: torch.Tensor,
    mask_hard: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "hard":
        anom_m = mask_hard
        zero_m = 1.0 - mask_hard
    elif mode == "soft":
        anom_m = weight_soft
        zero_m = 1.0 - weight_soft
    else:
        raise ValueError(f"anom_mask_mode must be 'hard' or 'soft', got {mode!r}")
    return anom_m, zero_m


def _region_losses(
    a_hat: torch.Tensor,
    target_r: torch.Tensor,
    anom_m: torch.Tensor,
    zero_m: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_anom = (anom_m * (a_hat - target_r) ** 2).sum() / (anom_m.sum() + eps)
    loss_zero = (zero_m * (a_hat ** 2)).sum() / (zero_m.sum() + eps)
    return loss_anom, loss_zero


class ControlNetAnomalyLoss(nn.Module):
    """
    Auxiliary losses on predicted x0 in norm space (AE frozen; grad -> x_gen / ControlNet).

    - loss_z: || AE.encode(filter(x_gen)) - z_a ||^2  (control latent consistency)
    - loss_anom: soft/hard mask from x_ori; MSE(a_hat(x_gen), data_f_ori) on anom region
    - loss_zero: same mask from x_ori; MSE(a_hat(x_gen), 0) on normal region

    anom_mask_mode=soft uses filter sigmoid weight w(|r|); zero_m = 1 - w.
    """

    def __init__(
        self,
        ae_model: ZeroAwareResidualAutoencoder,
        controlnet_cfg: dict,
        seq_length: Optional[int] = None,
        zone_baseline: Optional[torch.Tensor] = None,
        zone_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.filter_cfg = build_filter_cfg(controlnet_cfg, seq_length=seq_length)
        self.anom_mask_mode = str(controlnet_cfg.get("anom_mask_mode", "hard")).lower()
        self.zone_baseline = zone_baseline
        self.zone_scale = zone_scale

        from controlnet.ae.cond_grouped_residual import uses_cond_grouped_residual

        if uses_cond_grouped_residual(self.filter_cfg) and (
            zone_baseline is None or zone_scale is None
        ):
            raise ValueError(
                "cond-grouped residual requires zone_baseline and zone_scale tensors"
            )

        self.ae = ae_model
        for p in self.ae.parameters():
            p.requires_grad = False
        self.ae.eval()

    def _format_cond(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 3:
            cond = cond.squeeze(-1)
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        return cond.float()

    def _format_z_a(self, z_a: torch.Tensor) -> torch.Tensor:
        if z_a.ndim == 3:
            z_a = z_a.squeeze(-1)
        return z_a

    def forward(
        self,
        x_gen: torch.Tensor,
        x_ori: torch.Tensor,
        cond: torch.Tensor,
        z_a: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x_gen, x_ori: (B, T, 1) in normalized space.
        cond: (B, 1) categorical indices (cast to float for AE).
        z_a: (B, D) paired control latent target (detached inside).
        """
        zero = torch.zeros((), device=x_gen.device, dtype=x_gen.dtype)

        data_f_gen, weight_gen, _ = filtered_residual_from_data_torch(
            x_gen,
            self.filter_cfg,
            detach_stats=True,
            cond=cond,
            zone_baseline=self.zone_baseline,
            zone_scale=self.zone_scale,
            baseline=baseline,
        )
        cond_b = self._format_cond(cond)

        loss_z = zero
        if z_a is not None:
            z_gen = self.ae.encode(data_f_gen, weight_gen, cond_b)
            z_target = self._format_z_a(z_a).detach()
            loss_z = F.mse_loss(z_gen, z_target)

        with torch.no_grad():
            data_f_ori, weight_ori, mask_hard = filtered_residual_from_data_torch(
                x_ori,
                self.filter_cfg,
                detach_stats=True,
                cond=cond,
                zone_baseline=self.zone_baseline,
                zone_scale=self.zone_scale,
                baseline=baseline,
            )

        out = self.ae(data_f_gen, weight_gen, cond_b)
        anom_m, zero_m = _split_region_masks(
            weight_ori, mask_hard, self.anom_mask_mode
        )
        eps = float(self.filter_cfg["loss_eps"])
        loss_anom, loss_zero = _region_losses(
            out["a_hat"], data_f_ori, anom_m, zero_m, eps
        )

        return {
            "loss_z": loss_z,
            "loss_anom": loss_anom,
            "loss_zero": loss_zero,
        }
