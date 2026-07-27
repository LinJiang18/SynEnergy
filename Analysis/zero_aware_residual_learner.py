"""
Zero-aware Residual Representation Learner — Stage 1
=====================================================
ExperimentFirst-aligned pipeline (norm space + A4 gate loss):

  raw data + cond
    -> backbone MinMax normalize
    -> data_r -> data_f, weight (soft), mask_hard (eval)
    -> train ZeroAwareResidualAutoencoder (A4 loss)
    -> z_a

Core algorithms live in ``SynEnergy/controlnet/ae/residual_learner.py``.
This module provides Analysis paths, ``run_stage1()``, and backward-compatible re-exports.

Run:
    python zero_aware_residual_learner.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
SYNENERGY_ROOT = (SCRIPT_DIR / ".." / "SynEnergy").resolve()
if str(SYNENERGY_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNENERGY_ROOT))

from controlnet.ae.residual_learner import (  # noqa: E402
    DEFAULT_AE_CFG,
    ResidualDataset,
    ZeroAwareResidualAutoencoder,
    extract_anomaly_latent,
    extract_standardized_residual,
    filter_residual_deviation,
    filtered_residual_from_data_torch,
    load_ae_checkpoint,
    occurrence_mask_from_data_r,
    prepare_filtered_residual,
    save_ae_checkpoint,
    train_zero_aware_autoencoder,
    zero_aware_loss,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Result"

# Paths + Stage 1 overrides (defaults match ExperimentFirst / power_controlnet.yaml).
CONFIG: Dict = {
    "data_path": "../SynEnergy/Data/Power/Tallahassee_seq_180_sample_10000.npy",
    "cond_path": "../SynEnergy/Data/Power/Tallahassee_seq_180_sample_10000_condition.npy",
    "backbone_output_dir": "../SynEnergy/OUTPUT/power",
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "output_prefix": "Tallahassee_seq_180_sample_10000",
    "checkpoint_name": "checkpoint_zero_aware_ae.pt",
    **DEFAULT_AE_CFG,
}


def _resolve_path(path: Union[str, Path], base: Path = SCRIPT_DIR) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _resolve_device(cfg: Dict) -> torch.device:
    device = cfg.get("device")
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        return torch.device(device)
    return device


def _format_cond(cond: np.ndarray) -> np.ndarray:
    cond = np.asarray(cond, dtype=np.float32)
    if cond.ndim == 3 and cond.shape[-1] == 1:
        cond = cond.squeeze(-1)
    if cond.ndim == 1:
        cond = cond[:, None]
    return cond


def load_and_prepare_data(
    cfg: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw data + cond; normalize data with backbone normalizer (norm space);
    return data_norm, data_f, weight, mask_hard, cond.
    """
    cfg = dict(cfg or CONFIG)
    data_path = _resolve_path(cfg["data_path"])
    cond_path = _resolve_path(cfg["cond_path"])
    backbone_dir = _resolve_path(cfg["backbone_output_dir"])

    from Data.build_dataloader import load_normalizer_from_disk

    data_raw = np.load(data_path).astype(np.float32)
    normalizer = load_normalizer_from_disk(str(backbone_dir))
    if normalizer is None:
        raise FileNotFoundError(
            f"Backbone normalizer not found under {backbone_dir}/normalizer.joblib. "
            "Train backbone first."
        )

    flat = data_raw.reshape(-1, data_raw.shape[-1])
    data = normalizer.normalize(flat).reshape(data_raw.shape).astype(np.float32)

    data_f, weight, mask_hard, _data_r, _ = prepare_filtered_residual(data, cfg)
    cond = _format_cond(np.load(cond_path))

    assert data_f.shape == weight.shape == mask_hard.shape
    assert data_f.shape[0] == cond.shape[0]

    if verbose:
        print("=== Stage 1 data (norm space) ===")
        print(f"data_raw:   {data_raw.shape}")
        print(f"data_norm:  {data.shape}  range [{data.min():.3f}, {data.max():.3f}]")
        print(f"data_f:     {data_f.shape}")
        print(f"weight:     {weight.shape}  mean={weight.mean():.6f}")
        print(f"mask_hard:  {mask_hard.shape}  rate={mask_hard.mean():.6f}")
        print(f"cond:       {cond.shape}")
        print(
            f"filter_threshold={cfg['filter_threshold']}, "
            f"occurrence_threshold={cfg.get('occurrence_threshold', cfg['filter_threshold'])}, "
            f"filter_gamma={cfg['filter_gamma']}, filter_mode={cfg['filter_mode']}"
        )
        print(
            f"A4 loss: lambda_gate_zero={cfg['lambda_gate_zero']}, "
            f"lambda_gate_anom={cfg['lambda_gate_anom']}, "
            f"gate_normal_threshold={cfg['gate_normal_threshold']}, "
            f"gate_anom_threshold={cfg['gate_anom_threshold']}"
        )
        print()

    return data, data_f, weight, mask_hard, cond


def pack_anomaly_outputs(
    data: np.ndarray,
    cond: np.ndarray,
    model: ZeroAwareResidualAutoencoder,
    cfg: Optional[Dict] = None,
    data_f: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    mask_hard: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Returns data, cond, data_f, weight, mask_hard, z_a (z_a shape N×D×1)."""
    cfg = dict(cfg or CONFIG)
    cond = _format_cond(cond)

    if weight is None and mask_hard is not None:
        weight = mask_hard
    if data_f is None or weight is None or mask_hard is None:
        data_f, weight, mask_hard, _, _ = prepare_filtered_residual(data, cfg)
    else:
        data_f = np.asarray(data_f, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        mask_hard = np.asarray(mask_hard, dtype=np.float32)

    z_flat = extract_anomaly_latent(
        model,
        data_f,
        weight,
        cond,
        batch_size=batch_size or cfg.get("extract_batch_size", 256),
        device=_resolve_device(cfg),
    )
    z_a = np.expand_dims(z_flat, axis=-1).astype(np.float32)

    return {
        "data": data,
        "cond": cond,
        "data_f": data_f,
        "weight": weight,
        "mask": weight,
        "mask_hard": mask_hard,
        "z_a": z_a,
    }


class AnomalyPackDecoder(nn.Module):
    """Differentiable soft filter + optional frozen latent encoder."""

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        latent_model: Optional[ZeroAwareResidualAutoencoder] = None,
    ):
        super().__init__()
        self.cfg = dict(cfg or CONFIG)
        self.latent_model = latent_model
        if latent_model is not None:
            for p in latent_model.parameters():
                p.requires_grad = False

    def filter_data(
        self,
        data: torch.Tensor,
        detach_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return filtered_residual_from_data_torch(
            data, self.cfg, detach_stats=detach_stats
        )

    @torch.no_grad()
    def encode_z_a(
        self,
        data_f: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if self.latent_model is None:
            raise RuntimeError("latent_model is not set; load checkpoint first.")
        self.latent_model.eval()
        z = self.latent_model.encode(data_f, mask, cond)
        return z.unsqueeze(-1)


def save_checkpoint(
    model: ZeroAwareResidualAutoencoder,
    path: Union[str, Path],
    cfg: Optional[Dict] = None,
) -> Path:
    path = save_ae_checkpoint(model, path, cfg or CONFIG)
    print(f"Saved checkpoint to {path}")
    return path


def load_latent_encoder(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
    cfg: Optional[Dict] = None,
) -> ZeroAwareResidualAutoencoder:
    """Load Stage 1 checkpoint (alias for ``load_ae_checkpoint``)."""
    return load_ae_checkpoint(path, device=device, cfg=cfg)


def save_anomaly_pack(
    pack: Dict[str, np.ndarray],
    out_dir: Union[str, Path],
    prefix: str,
) -> Dict[str, Path]:
    """``*_mask.npy`` = soft weight; ``*_mask_hard.npy`` = occurrence for eval."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight = pack.get("weight", pack["mask"])
    paths = {
        "latent": out_dir / f"{prefix}_latent.npy",
        "data": out_dir / f"{prefix}.npy",
        "condition": out_dir / f"{prefix}_condition.npy",
        "residual": out_dir / f"{prefix}_residual.npy",
        "weight": out_dir / f"{prefix}_mask.npy",
        "mask_hard": out_dir / f"{prefix}_mask_hard.npy",
    }

    np.save(paths["latent"], pack["z_a"])
    np.save(paths["data"], pack["data"])
    np.save(paths["condition"], pack["cond"])
    np.save(paths["residual"], pack["data_f"])
    np.save(paths["weight"], weight)
    np.save(paths["mask_hard"], pack["mask_hard"])

    print("=== Saved anomaly pack ===")
    for name, p in paths.items():
        print(f"{name}: {p}")
    return paths


def run_stage1(cfg: Optional[Dict] = None) -> Dict:
    """Full Stage 1: norm preprocess -> A4 AE train -> checkpoint + npy pack."""
    cfg = dict(cfg or CONFIG)

    out_dir = _resolve_path(cfg["output_dir"])
    prefix = cfg["output_prefix"]
    ckpt_path = out_dir / cfg["checkpoint_name"]

    print("=" * 60)
    print("Stage 1: Zero-aware Residual Learner (ExperimentFirst / A4)")
    print("=" * 60)
    print()

    data, data_f, weight, mask_hard, cond = load_and_prepare_data(cfg, verbose=True)

    model, history = train_zero_aware_autoencoder(data_f, weight, cond, cfg)

    save_checkpoint(model, ckpt_path, cfg)

    pack = pack_anomaly_outputs(
        data=data,
        cond=cond,
        model=model,
        cfg=cfg,
        data_f=data_f,
        weight=weight,
        mask_hard=mask_hard,
    )
    save_anomaly_pack(pack, out_dir, prefix)

    print()
    print(f"z_a.shape: {pack['z_a'].shape}")
    print("Stage 1 complete.")
    print("=" * 60)

    return {
        "model": model,
        "history": history,
        "pack": pack,
        "checkpoint_path": ckpt_path,
        "output_dir": out_dir,
    }


__all__ = [
    "CONFIG",
    "SCRIPT_DIR",
    "DEFAULT_AE_CFG",
    "ZeroAwareResidualAutoencoder",
    "ResidualDataset",
    "extract_standardized_residual",
    "filter_residual_deviation",
    "occurrence_mask_from_data_r",
    "prepare_filtered_residual",
    "filtered_residual_from_data_torch",
    "zero_aware_loss",
    "train_zero_aware_autoencoder",
    "extract_anomaly_latent",
    "load_and_prepare_data",
    "pack_anomaly_outputs",
    "AnomalyPackDecoder",
    "save_checkpoint",
    "load_latent_encoder",
    "save_anomaly_pack",
    "run_stage1",
]


if __name__ == "__main__":
    run_stage1(CONFIG)
