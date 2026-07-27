"""
Invert the preprocessing chain used in main_temperature_example.py:

  °C -> monthly (year,month) z-score -> global StandardScaler on full series
  -> sliding windows -> rFFT -> boost high frequencies -> irFFT -> model space

Generated samples live in the final (irFFT) space; this module maps them back to °C
using the same calendar path as each training window (row i <-> window i).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import irfft, rfft, rfftfreq


def invert_fft_enhancement(
    x_time: np.ndarray,
    seq_len: int,
    c: float,
    percentage_of_freq_enhanced: int,
) -> np.ndarray:
    """Undo high-frequency boost applied before training (same indices as forward)."""
    x_time = np.asarray(x_time, dtype=np.float64)
    X = rfft(x_time, axis=1)
    freq = rfftfreq(seq_len)
    top_n = max(1, int((X.shape[1] * percentage_of_freq_enhanced) / 100))
    top_indices = np.argsort(freq)[-top_n:]
    X_un = X.copy()
    X_un[:, top_indices] /= c
    return irfft(X_un, n=seq_len, axis=1).astype(np.float32)


def invert_global_standard_scaler(x: np.ndarray, mean: float, scale: float) -> np.ndarray:
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    return (np.asarray(x, dtype=np.float64) * scale + mean).astype(np.float32)


def invert_monthly_standardization(
    monthly_z: np.ndarray,
    window_cal: np.ndarray,
    monthly_stats: pd.DataFrame,
) -> np.ndarray:
    """
    monthly_z: (N, L) after global scaler inverse — still monthly z-score space.
    window_cal: (N, L, 2) int, [year, month] for each position.
    monthly_stats: columns year, month, mean, std (raw °C climatology).
    """
    N, L = monthly_z.shape
    flat = window_cal.reshape(-1, 2)
    df_pos = pd.DataFrame(flat, columns=["year", "month"])
    stats = monthly_stats[["year", "month", "mean", "std"]].copy()
    merged = df_pos.merge(stats, on=["year", "month"], how="left")
    if merged["mean"].isna().any():
        missing = int(merged["mean"].isna().sum())
        raise ValueError(
            f"invert_monthly_standardization: {missing} (year,month) pairs missing from monthly_stats"
        )
    mu = merged["mean"].to_numpy(dtype=np.float64).reshape(N, L)
    sig = merged["std"].to_numpy(dtype=np.float64).reshape(N, L)
    sig = np.where((sig == 0) | ~np.isfinite(sig), 1.0, sig)
    return (np.asarray(monthly_z, dtype=np.float64) * sig + mu).astype(np.float32)


def model_space_to_celsius(
    seq_model_space: np.ndarray,
    *,
    seq_len: int,
    c: float,
    percentage_of_freq_enhanced: int,
    global_mean: float,
    global_scale: float,
    window_cal: np.ndarray,
    monthly_stats: pd.DataFrame,
) -> np.ndarray:
    """
    Full inverse: DDPM output -> daily min temperature °C.
    Accepts (N, L) or (N, L, 1); returns the same rank as input (channel dim preserved).
    """
    x = np.asarray(seq_model_space)
    trailing_c = x.ndim == 3 and x.shape[-1] == 1
    if trailing_c:
        x = x.squeeze(-1)
    out = invert_fft_enhancement(x, seq_len, c, percentage_of_freq_enhanced)
    out = invert_global_standard_scaler(out, global_mean, global_scale)
    out = invert_monthly_standardization(out, window_cal, monthly_stats)
    if trailing_c:
        out = out[..., np.newaxis]
    return out
