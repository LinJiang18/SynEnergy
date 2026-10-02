"""
Unpaired global time-series distribution metrics (ori vs gen).

Focus: temporal structure without flattening all timesteps into one marginal.
Extreme / anomaly tails are evaluated separately (see anomaly_distribution_metrics).

ori, gen: (N1, T, C) or (N1, T); N1 and N2 may differ.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.stats import wasserstein_distance
except ImportError:  # pragma: no cover
    wasserstein_distance = None


def _require_scipy_w1() -> None:
    if wasserstein_distance is None:
        raise ImportError("scipy is required for Wasserstein distance")


def _check_ts_shape(ori: np.ndarray, gen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ori = np.asarray(ori, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if ori.ndim == 2:
        ori = ori[..., np.newaxis]
    if gen.ndim == 2:
        gen = gen[..., np.newaxis]
    if ori.ndim != 3 or gen.ndim != 3:
        raise ValueError("ori and gen must have shape (N, T, C) or (N, T)")
    if ori.shape[1] != gen.shape[1] or ori.shape[2] != gen.shape[2]:
        raise ValueError("ori and gen must have the same T and C")
    if ori.shape[0] == 0 or gen.shape[0] == 0:
        raise ValueError("ori and gen must contain at least one sample")
    return ori, gen


def _w1_1d(a: np.ndarray, b: np.ndarray) -> float:
    _require_scipy_w1()
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def dist_timewise_marginal_w1(
    ori: np.ndarray,
    gen: np.ndarray,
    *,
    channel: int = 0,
) -> Dict[str, Any]:
    """
    Per-timestep marginal W1: W1(P(x_t | ori), P(x_t | gen)) for each t.

    Returns TW1_mean, TW1_max, and TW1_per_t (length T).
    """
    ori, gen = _check_ts_shape(ori, gen)
    t_len = ori.shape[1]
    per_t = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        per_t[t] = _w1_1d(ori[:, t, channel], gen[:, t, channel])
    finite = per_t[np.isfinite(per_t)]
    if finite.size == 0:
        return {"TW1_mean": float("nan"), "TW1_max": float("nan"), "TW1_per_t": per_t}
    return {
        "TW1_mean": float(finite.mean()),
        "TW1_max": float(finite.max()),
        "TW1_per_t": per_t,
    }


def dist_increment_w1(
    ori: np.ndarray,
    gen: np.ndarray,
    *,
    channel: int = 0,
) -> Dict[str, Any]:
    """
    Per-step increment W1 on Δx_t = x_{t+1} - x_t.

    Returns DW1_mean, DW1_max, and DW1_per_t (length T-1).
    """
    ori, gen = _check_ts_shape(ori, gen)
    d_ori = np.diff(ori[:, :, channel], axis=1)
    d_gen = np.diff(gen[:, :, channel], axis=1)
    t_len = d_ori.shape[1]
    per_t = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        per_t[t] = _w1_1d(d_ori[:, t], d_gen[:, t])
    finite = per_t[np.isfinite(per_t)]
    if finite.size == 0:
        return {"DW1_mean": float("nan"), "DW1_max": float("nan"), "DW1_per_t": per_t}
    return {
        "DW1_mean": float(finite.mean()),
        "DW1_max": float(finite.max()),
        "DW1_per_t": per_t,
    }


def dist_slot_marginal_w1(
    ori: np.ndarray,
    gen: np.ndarray,
    *,
    slots_per_day: int = 6,
    channel: int = 0,
) -> Dict[str, Any]:
    """
    Pool timesteps by intra-period slot (e.g. 6 slots/day) and compute W1 per slot.

    For T=180 and slots_per_day=6, slot s pools indices t where t % 6 == s.
    """
    ori, gen = _check_ts_shape(ori, gen)
    t_len = ori.shape[1]
    if slots_per_day < 1:
        raise ValueError("slots_per_day must be >= 1")
    per_slot = np.zeros(slots_per_day, dtype=np.float64)
    for s in range(slots_per_day):
        idx = np.arange(s, t_len, slots_per_day)
        if idx.size == 0:
            per_slot[s] = float("nan")
            continue
        per_slot[s] = _w1_1d(ori[:, idx, channel].ravel(), gen[:, idx, channel].ravel())
    finite = per_slot[np.isfinite(per_slot)]
    if finite.size == 0:
        return {
            "slot_TW1_mean": float("nan"),
            "slot_TW1_max": float("nan"),
            "slot_TW1_per_slot": per_slot,
        }
    return {
        "slot_TW1_mean": float(finite.mean()),
        "slot_TW1_max": float(finite.max()),
        "slot_TW1_per_slot": per_slot,
    }


def _acf_at_lag(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    t = len(x)
    if lag < 1 or t < lag + 2:
        return float("nan")
    x = x - x.mean()
    num = np.dot(x[:-lag], x[lag:])
    den = np.dot(x, x)
    if den <= 1e-12:
        return float("nan")
    return float(num / den)


def dist_acd(ori: np.ndarray, gen: np.ndarray, max_lag: int = 50) -> float:
    """Mean over lags of |mean ACF_ori(lag) - mean ACF_gen(lag)| (unpaired)."""
    ori, gen = _check_ts_shape(ori, gen)
    _, t, c = ori.shape
    m = min(max_lag, max(0, t - 2))
    if m < 1:
        return 0.0
    diffs = []
    for lag in range(1, m + 1):
        ao, ag = [], []
        for i in range(ori.shape[0]):
            for k in range(c):
                r = _acf_at_lag(ori[i, :, k], lag)
                if np.isfinite(r):
                    ao.append(r)
        for i in range(gen.shape[0]):
            for k in range(c):
                r = _acf_at_lag(gen[i, :, k], lag)
                if np.isfinite(r):
                    ag.append(r)
        if ao and ag:
            diffs.append(abs(np.mean(ao) - np.mean(ag)))
    return float(np.mean(diffs)) if diffs else 0.0


def dist_mmd_rbf(
    ori: np.ndarray,
    gen: np.ndarray,
    sigma: Optional[float] = None,
    n_repeats: int = 5,
    max_n: int = 10000,
    random_state: int = 42,
) -> float:
    """Squared MMD (MMD^2, biased) with RBF kernel on flattened (T,C) vectors per sample (auxiliary)."""
    ori_full = np.asarray(ori).reshape(ori.shape[0], -1).astype(np.float64)
    gen_full = np.asarray(gen).reshape(gen.shape[0], -1).astype(np.float64)
    mmds = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(random_state + rep)
        o = ori_full
        g = gen_full
        if o.shape[0] > max_n:
            o = o[rng.choice(o.shape[0], size=max_n, replace=False)]
        if g.shape[0] > max_n:
            g = g[rng.choice(g.shape[0], size=max_n, replace=False)]

        def rbf(x, y, s):
            xy = np.dot(x, y.T)
            xx = np.sum(x ** 2, axis=1)[:, None]
            yy = np.sum(y ** 2, axis=1)[None, :]
            return np.exp(-(xx + yy - 2 * xy) / (2 * s ** 2))

        sig = sigma
        if sig is None:
            n_sub = min(500, o.shape[0])
            idx = rng.choice(o.shape[0], size=n_sub, replace=False)
            sub = o[idx]
            dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
            sig = max(float(np.median(dists)), 1e-6)
        koo = rbf(o, o, sig)
        kgg = rbf(g, g, sig)
        kog = rbf(o, g, sig)
        mmd2 = koo.mean() + kgg.mean() - 2 * kog.mean()
        mmds.append(float(max(mmd2, 0.0)))
    return float(np.mean(mmds))


def dist_wasserstein_flat(ori: np.ndarray, gen: np.ndarray) -> float:
    """Legacy: 1D Wasserstein on all values pooled (no temporal structure)."""
    ori = np.asarray(ori).reshape(-1).astype(np.float64)
    gen = np.asarray(gen).reshape(-1).astype(np.float64)
    return _w1_1d(ori, gen)


def compute_distribution_distance(
    ori: np.ndarray,
    gen: np.ndarray,
    *,
    slots_per_day: int = 6,
    max_lag: int = 50,
    mmd_repeats: int = 5,
    include_legacy_flat: bool = False,
) -> Dict[str, float]:
    """
    Global time-series distribution distance (unpaired, temporal-aware).

    Primary metrics
    ---------------
    TW1_mean, TW1_max : per-timestep marginal W1 (load level profile)
    DW1_mean, DW1_max : per-step increment W1 (ramp / dynamics)
    slot_TW1_mean, slot_TW1_max : slot-pooled marginal W1 (intra-day phase)
    ACF_dist : pure autocorrelation distance (dist_acd)
    MMD : trajectory-level RBF-MMD^2 (squared; auxiliary)

    Legacy (optional)
    -----------------
    W_flat : flattened marginal W1 if include_legacy_flat=True
    """
    tw = dist_timewise_marginal_w1(ori, gen)
    dw = dist_increment_w1(ori, gen)
    slot = dist_slot_marginal_w1(ori, gen, slots_per_day=slots_per_day)

    out: Dict[str, float] = {
        "TW1_mean": float(tw["TW1_mean"]),
        "TW1_max": float(tw["TW1_max"]),
        "DW1_mean": float(dw["DW1_mean"]),
        "DW1_max": float(dw["DW1_max"]),
        "slot_TW1_mean": float(slot["slot_TW1_mean"]),
        "slot_TW1_max": float(slot["slot_TW1_max"]),
        "ACF_dist": float(dist_acd(ori, gen, max_lag=max_lag)),
        "MMD": float(dist_mmd_rbf(ori, gen, n_repeats=mmd_repeats)),
    }
    if include_legacy_flat:
        out["W_flat"] = float(dist_wasserstein_flat(ori, gen))
    return out


def compute_distribution_distance_detail(
    ori: np.ndarray,
    gen: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Like compute_distribution_distance but includes per-t curves."""
    slots_per_day = int(kwargs.pop("slots_per_day", 6))
    max_lag = int(kwargs.pop("max_lag", 50))
    mmd_repeats = int(kwargs.pop("mmd_repeats", 5))
    include_legacy_flat = bool(kwargs.pop("include_legacy_flat", False))

    summary = compute_distribution_distance(
        ori,
        gen,
        slots_per_day=slots_per_day,
        max_lag=max_lag,
        mmd_repeats=mmd_repeats,
        include_legacy_flat=include_legacy_flat,
    )
    tw = dist_timewise_marginal_w1(ori, gen)
    dw = dist_increment_w1(ori, gen)
    slot = dist_slot_marginal_w1(ori, gen, slots_per_day=slots_per_day)
    return {
        **summary,
        "TW1_per_t": tw["TW1_per_t"],
        "DW1_per_t": dw["DW1_per_t"],
        "slot_TW1_per_slot": slot["slot_TW1_per_slot"],
    }


def compare_global_distribution_multi(
    ori: np.ndarray,
    baselines: Dict[str, np.ndarray],
    **kwargs: Any,
) -> "pd.DataFrame":
    """Run compute_distribution_distance for each baseline; return summary table."""
    import pandas as pd

    rows = []
    for name, gen in baselines.items():
        row = compute_distribution_distance(ori, gen, **kwargs)
        row["baseline"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("baseline")


def print_global_distribution_report(result: Dict[str, float]) -> None:
    """Pretty-print compute_distribution_distance output."""
    order = [
        "TW1_mean", "TW1_max",
        "DW1_mean", "DW1_max",
        "slot_TW1_mean", "slot_TW1_max",
        "ACF_dist", "MMD", "W_flat",
    ]
    print("=" * 48)
    for key in order:
        if key in result:
            print(f"  {key:16s}: {result[key]:.6f}")
    print("=" * 48)


__all__ = [
    "compute_distribution_distance",
    "compute_distribution_distance_detail",
    "compare_global_distribution_multi",
    "print_global_distribution_report",
    "dist_timewise_marginal_w1",
    "dist_increment_w1",
    "dist_slot_marginal_w1",
    "dist_acd",
    "dist_mmd_rbf",
    "dist_wasserstein_flat",
]
