"""
Historical info: per-sample reference level used as the residual baseline.

Stored in raw (unnormalized) units with the same shape as the data, (N, T, 1), next to
the data file as ``<data_stem>_historical_info.npy``. Any (N, T, 1) array in raw units can
replace the default file; residuals are then ``x - historical_info`` (in norm space).

Default builder: slot-wise median over the days of each sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

HISTORICAL_INFO_SUFFIX = "_historical_info.npy"


def historical_info_path(data_path: Union[str, Path]) -> Path:
    """``Data/X.npy`` -> ``Data/X_historical_info.npy``."""
    p = Path(data_path)
    return p.with_name(p.stem + HISTORICAL_INFO_SUFFIX)


def build_historical_info(
    data: np.ndarray,
    num_days: int = 30,
    slots_per_day: int = 6,
) -> np.ndarray:
    """Default historical info (N, T, 1): per-slot median over days, tiled back to T."""
    data = np.asarray(data)
    expected_len = int(num_days) * int(slots_per_day)
    if data.ndim != 3 or data.shape[1] != expected_len or data.shape[2] != 1:
        raise ValueError(f"data must have shape (N, {expected_len}, 1), got {data.shape}")
    n = data.shape[0]
    x = data.reshape(n, num_days, slots_per_day)
    baseline = np.median(x, axis=1, keepdims=True)
    info = np.broadcast_to(baseline, x.shape).reshape(n, expected_len, 1)
    return np.ascontiguousarray(info, dtype=np.float32)


def save_historical_info(info: np.ndarray, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(info, dtype=np.float32))
    return path


def load_historical_info(
    path: Union[str, Path],
    expected_shape: Optional[tuple] = None,
) -> np.ndarray:
    info = np.load(Path(path))
    if info.ndim == 2:
        info = info[..., np.newaxis]
    if expected_shape is not None and tuple(info.shape) != tuple(expected_shape):
        raise ValueError(
            f"historical info {path} has shape {info.shape}, expected {tuple(expected_shape)}"
        )
    return info.astype(np.float32, copy=False)


def load_or_build_historical_info(
    data_path: Union[str, Path],
    data: np.ndarray,
    num_days: int = 30,
    slots_per_day: int = 6,
    path: Optional[Union[str, Path]] = None,
    logger=None,
) -> np.ndarray:
    """Load historical info for ``data`` (raw, (N, T, 1)); build and save it if missing."""
    path = Path(path) if path else historical_info_path(data_path)
    if path.is_file():
        info = load_historical_info(path, expected_shape=np.asarray(data).shape)
        if logger is not None:
            logger.log_info(f"Loaded historical info from {path}")
        return info
    info = build_historical_info(data, num_days=num_days, slots_per_day=slots_per_day)
    save_historical_info(info, path)
    if logger is not None:
        logger.log_info(f"Saved historical info to {path}")
    return info


def historical_info_to_norm(info: np.ndarray, normalizer) -> np.ndarray:
    """Map raw-unit historical info into the backbone's normalized space."""
    info = np.asarray(info)
    flat = info.reshape(-1, info.shape[-1])
    return normalizer.normalize(flat).reshape(info.shape).astype(np.float32)
