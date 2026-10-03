"""ControlNet data loading: backbone normalizer on data only; cond stays categorical."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from controlnet.ae.historical_info import (
    historical_info_to_norm,
    load_or_build_historical_info,
)
from Data.build_dataloader import (
    ZAwareConditionalDataset,
    load_cond_npy,
    load_latent_npy,
    load_normalizer_from_disk,
    load_npy_or_npz_X,
    split_to_chunks,
    _sample_indices_from_config,
)


def _controlnet_cfg(config: dict) -> dict:
    return config.get("controlnet", config.get("stage2", {}))


def _is_auto_backbone_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in ("auto", "")


def resolve_backbone_run_name(config: dict, args: Any) -> str:
    """
    Resolve which backbone run pairs with the current ControlNet job.

    ``backbone_run_name: auto`` supports two conventions:

    1. **Unified name** (recommended): Stage 1 and Stage 2 use the same ``--name``
       (e.g. ``power_temporal1``). Backbone artifacts live under the same OUTPUT dir.

    2. **Legacy suffix**: ``--name power_controlnet_temporal1`` strips ``_controlnet``
       and maps to ``power_temporal1``.

    Set ``controlnet.backbone_run_name`` to a fixed string to override.
    """
    cn = _controlnet_cfg(config)
    explicit = cn.get("backbone_run_name")
    if not _is_auto_backbone_value(explicit):
        return str(explicit)

    run_name = str(getattr(args, "name", "") or "")
    if "_controlnet" in run_name:
        return run_name.replace("_controlnet", "")
    if run_name:
        return run_name
    return "power"


def resolve_backbone_output_dir(config: dict, args: Any) -> str:
    cn = _controlnet_cfg(config)
    explicit = cn.get("backbone_output_dir")
    if not _is_auto_backbone_value(explicit):
        return str(explicit)
    return os.path.join("OUTPUT", resolve_backbone_run_name(config, args))


def _backbone_output_dir(config: dict, args: Any) -> str:
    return resolve_backbone_output_dir(config, args)


def load_training_bundle(config: dict, args: Any) -> dict:
    """
    Load raw data + cond; normalize data with backbone's normalizer.

    Returns dict with origin_norm, cond_array, normalizer, n_samples.
    """
    train_params = config["train"]["params"]
    npy_path = train_params["data_path"]
    seq_length = train_params["seq_length"]

    data = load_npy_or_npz_X(npy_path)
    if data.ndim == 3:
        if data.shape[1] != seq_length:
            raise ValueError(
                f"3D data L={data.shape[1]} != seq_length={seq_length}"
            )
    elif data.ndim == 2:
        n_keep = (data.shape[0] // seq_length) * seq_length
        data = data[:n_keep]

    backbone_output = _backbone_output_dir(config, args)
    normalizer = load_normalizer_from_disk(backbone_output)
    if normalizer is None:
        raise FileNotFoundError(
            f"Backbone normalizer not found under {backbone_output}/normalizer.joblib. "
            "Train backbone first with the same data."
        )

    origin_unnorm = split_to_chunks(data, seq_length)
    n_samples = origin_unnorm.shape[0]
    if data.ndim == 2:
        data_norm_full = normalizer.normalize(data)
        origin_norm = split_to_chunks(data_norm_full, seq_length)
    else:
        flat = data.reshape(-1, data.shape[-1])
        data_norm_full = normalizer.normalize(flat).reshape(data.shape)
        origin_norm = data_norm_full

    model_params = config.get("model", {}).get("params", {})
    cond_dim = int(model_params.get("cond_dim", 1))
    num_classes_per_cond = model_params.get("num_classes_per_cond")
    cond_path = train_params.get("cond_path")
    if not cond_path:
        raise ValueError("train.params.cond_path is required for controlnet.")
    cond_array = load_cond_npy(
        cond_path,
        cond_dim=cond_dim,
        expected_n=n_samples,
        num_classes_per_cond=num_classes_per_cond,
    )

    cn = _controlnet_cfg(config)
    num_days = int(cn.get("num_days", 30))
    slots_per_day = int(cn.get("slots_per_day", 6))
    baseline_norm = None
    if num_days * slots_per_day == int(seq_length):
        historical_info = load_or_build_historical_info(
            npy_path,
            origin_unnorm,
            num_days=num_days,
            slots_per_day=slots_per_day,
            path=cn.get("historical_info_path"),
        )
        baseline_norm = historical_info_to_norm(historical_info, normalizer)

    return {
        "origin_norm": origin_norm,
        "origin_unnorm": origin_unnorm,
        "cond_array": cond_array,
        "normalizer": normalizer,
        "n_samples": n_samples,
        "baseline_norm": baseline_norm,
    }


def build_controlnet_dataloader(
    config: dict,
    args: Any,
    origin_norm: np.ndarray,
    cond_array: np.ndarray,
    z_array: np.ndarray,
    baseline_norm: Optional[np.ndarray] = None,
) -> DataLoader:
    train_params = config["train"]["params"]
    batch_size = train_params["batch_size"]
    dataset = ZAwareConditionalDataset(origin_norm, cond_array, z_array, baseline_norm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def _z_a_path(output_dir: str) -> str:
    return os.path.join(output_dir, "intermediate", "z_a.npy")


def load_controlnet_sample_pairs(
    config: dict,
    output_dir: str,
    num_samples: int,
    indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned (cond, z_a) for controlnet sampling."""
    sample_params = config.get("sample", {}).get("params", {})
    cond_path = sample_params.get("cond_path")
    if not cond_path:
        raise ValueError("sample.params.cond_path is required for controlnet sampling.")

    latent_path = _z_a_path(output_dir)
    if not os.path.isfile(latent_path):
        cn = _controlnet_cfg(config)
        latent_path = cn.get("latent_path", latent_path)
    if not os.path.isfile(latent_path):
        raise FileNotFoundError(
            f"z_a table not found at {latent_path}. Run controlnet training first."
        )

    model_params = config.get("model", {}).get("params", {})
    cond_dim = int(model_params.get("cond_dim", 1))
    num_classes_per_cond = model_params.get("num_classes_per_cond")

    cond = load_cond_npy(
        cond_path,
        cond_dim=cond_dim,
        expected_n=None,
        num_classes_per_cond=num_classes_per_cond,
    )
    z = load_latent_npy(latent_path)
    n = cond.shape[0]
    if z.shape[0] != n:
        raise ValueError(
            f"cond rows {n} and z_a rows {z.shape[0]} must match for aligned sampling."
        )

    if indices is None:
        indices = _sample_indices_from_config(cond, num_samples, config)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape[0] != num_samples:
            raise ValueError(
                f"indices length {indices.shape[0]} must match num_samples {num_samples}."
            )
        if indices.min() < 0 or indices.max() >= n:
            raise ValueError(f"indices out of range for N={n}.")

    return cond[indices], z[indices], indices
