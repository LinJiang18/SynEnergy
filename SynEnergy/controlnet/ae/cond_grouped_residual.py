"""Residual extraction when num_days * slots_per_day != seq_length (cond-grouped reshape)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from controlnet.ae.residual_learner import (
    extract_standardized_residual,
    filter_residual_deviation_torch,
)

COND_GROUP_STATS_FNAME = "cond_group_residual_stats.npz"


def uses_cond_grouped_residual(cfg: dict) -> bool:
    """True when residual must group rows by cond before reshape (e.g. NY daily)."""
    if "seq_len" not in cfg:
        return False
    expected = int(cfg["num_days"]) * int(cfg["slots_per_day"])
    return expected != int(cfg["seq_len"])


def _cond_flat(cond: np.ndarray) -> np.ndarray:
    return np.asarray(cond).reshape(-1).astype(np.int64)


def extract_standardized_residual_by_cond(
    data: np.ndarray,
    cond: np.ndarray,
    num_days: int,
    slots_per_day: int,
    eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Group daily rows by cond, reshape each zone to (1, num_days, slots_per_day),
    run extract_standardized_residual, map back to (N, slots_per_day, 1).
    """
    from controlnet.ae.residual_learner import extract_standardized_residual

    cond_v = _cond_flat(cond)
    n_total, seq_len, _ = data.shape
    if seq_len != slots_per_day:
        raise ValueError(
            f"data seq_len {seq_len} must equal slots_per_day {slots_per_day}"
        )

    data_r = np.empty_like(data)
    max_id = int(cond_v.max())
    zone_baseline = np.zeros((max_id + 1, slots_per_day), dtype=np.float32)
    zone_scale = np.ones((max_id + 1, slots_per_day), dtype=np.float32)

    for z in np.unique(cond_v):
        idx = np.where(cond_v == z)[0]
        idx = np.sort(idx)
        block = data[idx]
        if block.shape[0] != num_days:
            raise ValueError(
                f"cond {z}: expected exactly {num_days} daily rows, got {block.shape[0]}"
            )

        flat = block.reshape(1, num_days * slots_per_day, 1)
        data_r_z, baseline, scale, _, _ = extract_standardized_residual(
            flat,
            num_days=num_days,
            slots_per_day=slots_per_day,
            eps=eps,
            min_scale=min_scale,
            use_mad_consistency=use_mad_consistency,
        )
        data_r[idx] = data_r_z.reshape(num_days, slots_per_day, 1)
        zone_baseline[z] = baseline.reshape(slots_per_day)
        zone_scale[z] = scale.reshape(slots_per_day)

    zone_stats = {
        "baseline": zone_baseline,
        "scale": zone_scale,
        "cond_ids": np.unique(cond_v).astype(np.int64),
        "num_days": np.int32(num_days),
        "slots_per_day": np.int32(slots_per_day),
    }
    return data_r.astype(np.float32), zone_stats


def standardized_residual_by_cond_torch(
    data: torch.Tensor,
    cond: torch.Tensor,
    zone_baseline: torch.Tensor,
    zone_scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply precomputed per-cond baseline/scale to daily batches (B, T, 1)."""
    if cond.ndim == 3:
        cond = cond.squeeze(-1)
    if cond.ndim == 2 and cond.shape[1] == 1:
        cond = cond.squeeze(1)
    cond_idx = cond.long().reshape(-1)

    b = zone_baseline[cond_idx].unsqueeze(-1)
    s = zone_scale[cond_idx].unsqueeze(-1)
    return (data - b) / (s + eps)


def filtered_residual_from_cond_grouped_torch(
    data: torch.Tensor,
    cond: torch.Tensor,
    zone_baseline: torch.Tensor,
    zone_scale: torch.Tensor,
    cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch: data -> data_r (lookup) -> data_f, weight, mask_hard."""
    eps = float(cfg.get("residual_eps", 1e-6))
    data_r = standardized_residual_by_cond_torch(
        data, cond, zone_baseline, zone_scale, eps=eps
    )
    data_f, weight = filter_residual_deviation_torch(
        data_r,
        threshold=cfg["filter_threshold"],
        mode=cfg["filter_mode"],
        gamma=cfg["filter_gamma"],
        return_mask=True,
    )
    occ_threshold = float(cfg.get("occurrence_threshold", cfg["filter_threshold"]))
    mask_hard = (torch.abs(data_r) > occ_threshold).to(data_r.dtype)
    return data_f, weight, mask_hard


def save_cond_group_stats(zone_stats: Dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        baseline=zone_stats["baseline"],
        scale=zone_stats["scale"],
        cond_ids=zone_stats["cond_ids"],
        num_days=zone_stats["num_days"],
        slots_per_day=zone_stats["slots_per_day"],
    )
    return path


def load_cond_group_stats(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cond-group residual stats not found: {path}")
    z = np.load(path)
    return {
        "baseline": z["baseline"].astype(np.float32),
        "scale": z["scale"].astype(np.float32),
        "cond_ids": z["cond_ids"].astype(np.int64),
        "num_days": int(z["num_days"]),
        "slots_per_day": int(z["slots_per_day"]),
    }


def cond_group_stats_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "intermediate" / COND_GROUP_STATS_FNAME


def fit_cond_group_zone_stats(
    data: np.ndarray,
    cond: np.ndarray,
    num_days: int,
    slots_per_day: int,
    eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
) -> Dict[str, np.ndarray]:
    """Fit per-zone baseline/scale from ori (e.g. 365 daily rows per cond)."""
    _, zone_stats = extract_standardized_residual_by_cond(
        data,
        cond,
        num_days=num_days,
        slots_per_day=slots_per_day,
        eps=eps,
        min_scale=min_scale,
        use_mad_consistency=use_mad_consistency,
    )
    return zone_stats


def apply_cond_group_residual(
    data: np.ndarray,
    cond: np.ndarray,
    zone_stats: Dict[str, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply precomputed zone baseline/scale to daily rows (N, slots_per_day, 1)."""
    data = np.asarray(data, dtype=np.float32)
    cond_v = _cond_flat(cond)
    if data.shape[0] != cond_v.shape[0]:
        raise ValueError(
            f"data rows {data.shape[0]} must match cond rows {cond_v.shape[0]}"
        )
    baseline = np.asarray(zone_stats["baseline"], dtype=np.float32)
    scale = np.asarray(zone_stats["scale"], dtype=np.float32)
    b = baseline[cond_v]
    s = scale[cond_v]
    data_r = (data[..., 0] - b) / (s + eps)
    return data_r[..., np.newaxis].astype(np.float32)
