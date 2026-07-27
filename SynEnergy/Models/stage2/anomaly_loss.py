"""Legacy Stage 2 anomaly loss (unnorm space). Prefer controlnet.loss.anomaly_loss."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from controlnet.ae.residual_learner import (
    filtered_residual_from_data_torch,
    load_ae_checkpoint,
)


def build_filter_cfg(stage2_cfg: dict) -> dict:
    return {
        "num_days": stage2_cfg.get("num_days", 30),
        "slots_per_day": stage2_cfg.get("slots_per_day", 6),
        "residual_eps": stage2_cfg.get("residual_eps", 1e-6),
        "min_scale": stage2_cfg.get("min_scale", 1.0),
        "use_mad_consistency": stage2_cfg.get("use_mad_consistency", True),
        "filter_threshold": stage2_cfg.get("filter_threshold", 0.12),
        "filter_mode": stage2_cfg.get("filter_mode", "soft"),
        "filter_gamma": stage2_cfg.get("filter_gamma", 50.0),
        "occurrence_threshold": stage2_cfg.get("occurrence_threshold", 0.12),
        "loss_eps": stage2_cfg.get("loss_eps", 1e-6),
    }


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


class Stage2AnomalyLoss(nn.Module):
    def __init__(
        self,
        ae_ckpt: str,
        stage2_cfg: dict,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.filter_cfg = build_filter_cfg(stage2_cfg)
        self.anom_mask_mode = str(stage2_cfg.get("anom_mask_mode", "hard")).lower()
        self.ae = load_ae_checkpoint(ae_ckpt, device=device)
        for p in self.ae.parameters():
            p.requires_grad = False
        self.ae.eval()

    def _format_cond(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 3:
            cond = cond.squeeze(-1)
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        return cond.float()

    def forward(
        self,
        x_gen: torch.Tensor,
        x_ori: torch.Tensor,
        cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            data_f_ori, weight_ori, mask_hard = filtered_residual_from_data_torch(
                x_ori, self.filter_cfg, detach_stats=True
            )

        data_f_gen, weight_gen, _ = filtered_residual_from_data_torch(
            x_gen, self.filter_cfg, detach_stats=True
        )
        cond_b = self._format_cond(cond)

        out = self.ae(data_f_gen, weight_gen, cond_b)
        anom_m, zero_m = _split_region_masks(
            weight_ori, mask_hard, self.anom_mask_mode
        )
        eps = float(self.filter_cfg["loss_eps"])
        loss_anom, loss_zero = _region_losses(
            out["a_hat"], data_f_ori, anom_m, zero_m, eps
        )

        return {
            "loss_anom": loss_anom,
            "loss_zero": loss_zero,
        }
