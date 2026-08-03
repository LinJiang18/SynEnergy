"""Read/write AE + z_a intermediates under OUTPUT/{name}/intermediate/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from controlnet.ae.cond_grouped_residual import (
    COND_GROUP_STATS_FNAME,
    save_cond_group_stats,
)
from controlnet.ae.residual_learner import (
    ZeroAwareResidualAutoencoder,
    build_ae_cfg,
    load_ae_checkpoint,
    save_ae_checkpoint,
)

AE_CKPT = "ae.pt"
Z_A_NPY = "z_a.npy"
DATA_F_NPY = "data_f.npy"
WEIGHT_NPY = "weight.npy"
MASK_HARD_NPY = "mask_hard.npy"


def intermediate_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "intermediate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_intermediates(
    output_dir: str | Path,
    ae_model: ZeroAwareResidualAutoencoder,
    intermediates: Dict[str, Any],
    config: dict,
) -> Path:
    """Persist AE checkpoint and numpy tables from the AE phase."""
    out = intermediate_dir(output_dir)
    ae_cfg = build_ae_cfg(config)
    save_ae_checkpoint(ae_model, out / AE_CKPT, ae_cfg)

    np.save(out / Z_A_NPY, np.asarray(intermediates["z_a"], dtype=np.float32))
    np.save(out / DATA_F_NPY, np.asarray(intermediates["data_f"], dtype=np.float32))
    np.save(out / WEIGHT_NPY, np.asarray(intermediates["weight"], dtype=np.float32))
    np.save(out / MASK_HARD_NPY, np.asarray(intermediates["mask_hard"], dtype=np.float32))

    cond_group_stats = intermediates.get("cond_group_stats")
    if cond_group_stats is not None:
        save_cond_group_stats(cond_group_stats, out / COND_GROUP_STATS_FNAME)

    return out


def load_intermediates(
    output_dir: str | Path,
    config: dict,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load AE model and z_a table written during training."""
    out = intermediate_dir(output_dir)
    ae_path = out / AE_CKPT
    z_path = out / Z_A_NPY
    if not ae_path.is_file():
        raise FileNotFoundError(f"AE checkpoint not found: {ae_path}")
    if not z_path.is_file():
        raise FileNotFoundError(f"z_a table not found: {z_path}")

    ae_model = load_ae_checkpoint(ae_path, device=device, cfg=build_ae_cfg(config))
    z_a = np.load(z_path)
    result: Dict[str, Any] = {"ae_model": ae_model, "z_a": z_a}

    for name, fname in (
        ("data_f", DATA_F_NPY),
        ("weight", WEIGHT_NPY),
        ("mask_hard", MASK_HARD_NPY),
    ):
        path = out / fname
        if path.is_file():
            result[name] = np.load(path)
    return result
