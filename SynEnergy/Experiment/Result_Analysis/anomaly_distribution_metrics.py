"""
Unpaired anomaly-distribution comparison for ori vs gen time series.

Both corpora are mapped to standardized residuals r = (x - local_baseline) / scale;
anomaly points satisfy |r| > tau. No sample pairing, no shape matching — only
population-level marginal comparisons (Wasserstein on per-series scalars + pooled quantiles).

ori: (N1, T, 1), gen: (N2, T, 1); N1 and N2 may differ.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

_SYNENERGY_ROOT = Path(__file__).resolve().parents[2]
if str(_SYNENERGY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYNENERGY_ROOT))

from controlnet.ae.cond_grouped_residual import (  # noqa: E402
    apply_cond_group_residual,
    fit_cond_group_zone_stats,
)
from controlnet.ae.residual_learner import extract_standardized_residual  # noqa: E402

try:
    from scipy.stats import wasserstein_distance
except ImportError:  # pragma: no cover
    wasserstein_distance = None


DEFAULT_TAU = 0.12
DEFAULT_NUM_DAYS = 30
DEFAULT_SLOTS_PER_DAY = 6
DEFAULT_RESIDUAL_EPS = 1e-6
DEFAULT_MIN_SCALE = 1.0

SAMPLE_FEATURE_KEYS = (
    "anomaly_rate",
    "event_count",
    "max_abs_r",
    "mean_abs_on_anom",
    "pos_ratio",
    "anomaly_energy",
    "longest_run",
)

# Per-sample scalars shown in ``anomaly_distribution_summary_table`` (excludes max_abs_r).
SUMMARY_TABLE_FEATURE_KEYS = tuple(
    k for k in SAMPLE_FEATURE_KEYS if k != "max_abs_r"
)

POOLED_QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)

ORI_COLOR = "#2166ac"
BASELINE_COLORS = ("#b2182b", "#fdae61", "#4daf4a", "#984ea3", "#7570b3")


def _validate_series_batch(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 1:
        raise ValueError(f"{name} must have shape (N, T, 1), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return arr


def _event_count_1d(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0:
        return 0
    padded = np.concatenate(([False], m, [False])).astype(np.int8)
    return int(np.sum(np.diff(padded) == 1))


def _standardized_residual_batch(
    data: np.ndarray,
    num_days: int,
    slots_per_day: int,
    residual_eps: float,
    min_scale: float,
    use_mad_consistency: bool,
    *,
    cond: Optional[np.ndarray] = None,
    zone_stats: Optional[Dict[str, np.ndarray]] = None,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    if zone_stats is not None:
        if cond is None:
            raise ValueError("cond is required when zone_stats is provided")
        data_r = apply_cond_group_residual(
            data.astype(np.float32),
            cond,
            zone_stats,
            eps=residual_eps,
        )
        return np.asarray(data_r, dtype=np.float64)

    data_r, _, _, _, _ = extract_standardized_residual(
        data.astype(np.float32),
        num_days=num_days,
        slots_per_day=slots_per_day,
        eps=residual_eps,
        min_scale=min_scale,
        use_mad_consistency=use_mad_consistency,
        baseline=baseline,
    )
    return np.asarray(data_r, dtype=np.float64)


def _per_sample_features(data_r: np.ndarray, tau: float, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    n = data_r.shape[0]
    abs_r = np.abs(data_r[..., 0])
    r = data_r[..., 0]
    mask = abs_r > float(tau)

    anomaly_rate = mask.mean(axis=1)
    max_abs_r = abs_r.max(axis=1)
    anomaly_energy = np.sum((r ** 2) * mask, axis=1)

    event_count = np.zeros(n, dtype=np.float64)
    longest_run = np.zeros(n, dtype=np.float64)
    mean_abs_on_anom = np.full(n, np.nan, dtype=np.float64)
    pos_ratio = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        m = mask[i]
        event_count[i] = _event_count_1d(m)
        if m.any():
            cur = max_len = 0
            for v in m:
                if v:
                    cur += 1
                    max_len = max(max_len, cur)
                else:
                    cur = 0
            longest_run[i] = max_len
            mean_abs_on_anom[i] = abs_r[i, m].mean()
            pos_ratio[i] = (r[i, m] > 0).mean()
        else:
            longest_run[i] = 0.0

    return {
        "anomaly_rate": anomaly_rate,
        "event_count": event_count,
        "max_abs_r": max_abs_r,
        "mean_abs_on_anom": mean_abs_on_anom,
        "pos_ratio": pos_ratio,
        "anomaly_energy": anomaly_energy,
        "longest_run": longest_run,
    }


def _finite_values(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if wasserstein_distance is None:
        raise ImportError("scipy is required for Wasserstein distance")
    fa = _finite_values(a)
    fb = _finite_values(b)
    if fa.size == 0 or fb.size == 0:
        return float("nan")
    return float(wasserstein_distance(fa, fb))


def _summarize_vector(x: np.ndarray) -> Dict[str, float]:
    v = _finite_values(x)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "n": int(v.size),
    }


def _quantile_dict(values: np.ndarray, quantiles: Sequence[float]) -> Dict[str, float]:
    v = _finite_values(values)
    if v.size == 0:
        return {f"q{int(q * 100) if q < 1 else 100}": float("nan") for q in quantiles}
    out: Dict[str, float] = {}
    for q in quantiles:
        key = "max" if q >= 1.0 else f"q{int(round(q * 100))}"
        out[key] = float(np.quantile(v, q))
    return out


def _build_comparison_from_residuals(
    r_ori: np.ndarray,
    feats_ori: Dict[str, np.ndarray],
    r_gen: np.ndarray,
    tau: float,
    quantiles: Sequence[float],
    store_pooled_arrays: bool,
) -> Dict[str, Any]:
    feats_gen = _per_sample_features(r_gen, tau=tau)

    sample_features: Dict[str, Dict[str, Any]] = {}
    for key in SAMPLE_FEATURE_KEYS:
        sample_features[key] = {
            "ori": _summarize_vector(feats_ori[key]),
            "gen": _summarize_vector(feats_gen[key]),
            "wasserstein": _w1(feats_ori[key], feats_gen[key]),
        }

    abs_ori = np.abs(r_ori).ravel()
    abs_gen = np.abs(r_gen).ravel()
    mask_ori = abs_ori > tau
    mask_gen = abs_gen > tau
    abs_anom_ori = abs_ori[mask_ori]
    abs_anom_gen = abs_gen[mask_gen]

    pooled = {
        "abs_r": {
            "ori": _quantile_dict(abs_ori, quantiles),
            "gen": _quantile_dict(abs_gen, quantiles),
            "wasserstein": _w1(abs_ori, abs_gen),
        },
        "abs_r_anomaly": {
            "ori": _quantile_dict(abs_anom_ori, quantiles),
            "gen": _quantile_dict(abs_anom_gen, quantiles),
            "wasserstein": _w1(abs_anom_ori, abs_anom_gen),
            "n_points_ori": int(abs_anom_ori.size),
            "n_points_gen": int(abs_anom_gen.size),
        },
    }

    out: Dict[str, Any] = {
        "n_ori": int(r_ori.shape[0]),
        "n_gen": int(r_gen.shape[0]),
        "sample_features": sample_features,
        "pooled": pooled,
        "per_sample": {"ori": feats_ori, "gen": feats_gen},
    }
    if store_pooled_arrays:
        out["pooled_arrays"] = {
            "abs_r_ori": abs_ori,
            "abs_r_gen": abs_gen,
            "abs_r_anom_ori": abs_anom_ori,
            "abs_r_anom_gen": abs_anom_gen,
        }
    return out


def compare_anomaly_distribution(
    ori: np.ndarray,
    gen: np.ndarray,
    tau: float = DEFAULT_TAU,
    num_days: int = DEFAULT_NUM_DAYS,
    slots_per_day: int = DEFAULT_SLOTS_PER_DAY,
    residual_eps: float = DEFAULT_RESIDUAL_EPS,
    min_scale: float = DEFAULT_MIN_SCALE,
    use_mad_consistency: bool = True,
    quantiles: Sequence[float] = POOLED_QUANTILES,
    store_pooled_arrays: bool = False,
    cond: Optional[np.ndarray] = None,
    gen_cond: Optional[np.ndarray] = None,
    cond_group_num_days: Optional[int] = None,
    ori_baseline: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare unpaired anomaly distributions of ori and gen.

    Parameters
    ----------
    ori, gen : array (N, T, 1)
        Two corpora in the **same value space** (both norm or both unnorm).
    tau : float
        Anomaly threshold on |standardized residual|.

    store_pooled_arrays : bool
        If True, attach ``pooled_arrays`` (all ``|r|`` and anomaly ``|r|``) for plotting.

    Returns
    -------
    dict with keys: config, n_ori, n_gen, sample_features, pooled, per_sample (raw arrays).
    """
    ori = _validate_series_batch(ori, "ori")
    gen = _validate_series_batch(gen, "gen")
    if ori.shape[1] != gen.shape[1]:
        raise ValueError(
            f"seq length mismatch: ori T={ori.shape[1]} vs gen T={gen.shape[1]}"
        )

    zone_stats = None
    if cond_group_num_days is not None:
        if cond is None:
            raise ValueError("cond is required when cond_group_num_days is set")
        zone_stats = fit_cond_group_zone_stats(
            ori,
            cond,
            num_days=int(cond_group_num_days),
            slots_per_day=int(slots_per_day),
            eps=residual_eps,
            min_scale=min_scale,
            use_mad_consistency=use_mad_consistency,
        )
        gen_cond = gen_cond if gen_cond is not None else cond

    r_ori = _standardized_residual_batch(
        ori,
        num_days,
        slots_per_day,
        residual_eps,
        min_scale,
        use_mad_consistency,
        cond=cond,
        zone_stats=zone_stats,
        baseline=ori_baseline,
    )
    r_gen = _standardized_residual_batch(
        gen,
        num_days,
        slots_per_day,
        residual_eps,
        min_scale,
        use_mad_consistency,
        cond=gen_cond,
        zone_stats=zone_stats,
    )

    feats_ori = _per_sample_features(r_ori, tau=tau)
    built = _build_comparison_from_residuals(
        r_ori, feats_ori, r_gen, tau, quantiles, store_pooled_arrays
    )

    return {
        "config": {
            "tau": float(tau),
            "num_days": int(num_days),
            "slots_per_day": int(slots_per_day),
            "residual_eps": float(residual_eps),
            "min_scale": float(min_scale),
            "use_mad_consistency": bool(use_mad_consistency),
            "seq_length": int(ori.shape[1]),
            "cond_group_num_days": (
                int(cond_group_num_days) if cond_group_num_days is not None else None
            ),
        },
        **built,
    }


def compare_anomaly_distribution_multi(
    ori: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    tau: float = DEFAULT_TAU,
    num_days: int = DEFAULT_NUM_DAYS,
    slots_per_day: int = DEFAULT_SLOTS_PER_DAY,
    residual_eps: float = DEFAULT_RESIDUAL_EPS,
    min_scale: float = DEFAULT_MIN_SCALE,
    use_mad_consistency: bool = True,
    quantiles: Sequence[float] = POOLED_QUANTILES,
    store_pooled_arrays: bool = False,
    cond: Optional[np.ndarray] = None,
    baseline_conds: Optional[Mapping[str, np.ndarray]] = None,
    cond_group_num_days: Optional[int] = None,
    ori_baseline: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare ori against multiple generated baselines (unpaired).

    Parameters
    ----------
    baselines : mapping
        e.g. ``{"gen": gen_arr, "gen1": gen1_arr}`` — names used in reports/plots.
    """
    if not baselines:
        raise ValueError("baselines must contain at least one array")

    ori = _validate_series_batch(ori, "ori")
    zone_stats = None
    if cond_group_num_days is not None:
        if cond is None:
            raise ValueError("cond is required when cond_group_num_days is set")
        zone_stats = fit_cond_group_zone_stats(
            ori,
            cond,
            num_days=int(cond_group_num_days),
            slots_per_day=int(slots_per_day),
            eps=residual_eps,
            min_scale=min_scale,
            use_mad_consistency=use_mad_consistency,
        )

    r_ori = _standardized_residual_batch(
        ori,
        num_days,
        slots_per_day,
        residual_eps,
        min_scale,
        use_mad_consistency,
        cond=cond,
        zone_stats=zone_stats,
        baseline=ori_baseline,
    )
    feats_ori = _per_sample_features(r_ori, tau=tau)

    config = {
        "tau": float(tau),
        "num_days": int(num_days),
        "slots_per_day": int(slots_per_day),
        "residual_eps": float(residual_eps),
        "min_scale": float(min_scale),
        "use_mad_consistency": bool(use_mad_consistency),
        "seq_length": int(ori.shape[1]),
        "cond_group_num_days": (
            int(cond_group_num_days) if cond_group_num_days is not None else None
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, gen in baselines.items():
        gen = _validate_series_batch(gen, name)
        if gen.shape[1] != ori.shape[1]:
            raise ValueError(
                f"{name} seq length mismatch: T={gen.shape[1]} vs ori T={ori.shape[1]}"
            )
        gen_cond = None
        if zone_stats is not None:
            if baseline_conds is not None and name in baseline_conds:
                gen_cond = baseline_conds[name]
            elif cond is not None and gen.shape[0] == cond.shape[0]:
                gen_cond = cond
            else:
                raise ValueError(
                    f"baseline_conds['{name}'] required for cond-grouped residual "
                    f"(gen rows={gen.shape[0]})"
                )
        r_gen = _standardized_residual_batch(
            gen,
            num_days,
            slots_per_day,
            residual_eps,
            min_scale,
            use_mad_consistency,
            cond=gen_cond,
            zone_stats=zone_stats,
        )
        built = _build_comparison_from_residuals(
            r_ori, feats_ori, r_gen, tau, quantiles, store_pooled_arrays
        )
        results[name] = {"config": config, **built}

    return {"config": config, "n_ori": int(ori.shape[0]), "baselines": results}


def _single_result_from_multi(
    multi_result: Dict[str, Any], baseline_name: str
) -> Dict[str, Any]:
    block = multi_result["baselines"][baseline_name]
    return {"config": multi_result["config"], **block}


def print_anomaly_distribution_multi_report(
    multi_result: Dict[str, Any],
    baseline_names: Optional[Sequence[str]] = None,
) -> None:
    """Print text reports for each baseline vs ori."""
    names = list(baseline_names or multi_result["baselines"].keys())
    for name in names:
        print_anomaly_distribution_report(
            _single_result_from_multi(multi_result, name),
            title=f"Anomaly distribution: ori vs {name}",
        )
        print()


def anomaly_distribution_summary_table(
    multi_result: Dict[str, Any],
    baseline_names: Optional[Sequence[str]] = None,
    include_ori: bool = True,
    feature_keys: Optional[Sequence[str]] = None,
) -> "pd.DataFrame":
    """Build a compact W1 summary table (requires pandas).

    When ``include_ori=True``, prepend a reference row with ori means and W1=0.
    Default columns use ``SUMMARY_TABLE_FEATURE_KEYS`` (no max_abs_r).
    """
    import pandas as pd

    keys = list(feature_keys or SUMMARY_TABLE_FEATURE_KEYS)
    names = list(baseline_names or multi_result["baselines"].keys())
    if not names:
        raise ValueError("multi_result has no baselines to summarize")

    ref = multi_result["baselines"][names[0]]
    rows: List[Dict[str, Any]] = []

    if include_ori:
        sf_ref = ref["sample_features"]
        ori_row: Dict[str, Any] = {"baseline": "ori"}
        for key in keys:
            ori_row[f"{key}_w1"] = 0.0
            ori_row[f"{key}_gen_mean"] = sf_ref[key]["ori"]["mean"]
        ori_row["w1_abs_r"] = 0.0
        ori_row["w1_abs_r_anom"] = 0.0
        rows.append(ori_row)

    for name in names:
        sf = multi_result["baselines"][name]["sample_features"]
        row: Dict[str, Any] = {"baseline": name}
        for key in keys:
            row[f"{key}_w1"] = sf[key]["wasserstein"]
            row[f"{key}_gen_mean"] = sf[key]["gen"]["mean"]
        row["w1_abs_r"] = multi_result["baselines"][name]["pooled"]["abs_r"]["wasserstein"]
        row["w1_abs_r_anom"] = multi_result["baselines"][name]["pooled"]["abs_r_anomaly"]["wasserstein"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("baseline")


def _plot_feature_panel(
    ax,
    ori_vals: np.ndarray,
    baseline_vals: Mapping[str, np.ndarray],
    *,
    key: str,
    w1_by_name: Mapping[str, float],
    colors: Sequence[str],
    bins: int,
    kde: bool,
    log_energy: bool,
) -> None:
    use_log_x = log_energy and key == "anomaly_energy"
    ori_v = _finite_values(ori_vals)
    base_v = {n: _finite_values(v) for n, v in baseline_vals.items()}
    if use_log_x:
        ori_v = ori_v[ori_v > 0]
        base_v = {n: v[v > 0] for n, v in base_v.items()}

    all_vals = [ori_v] + list(base_v.values())
    all_vals = [v for v in all_vals if v.size > 0]
    if not all_vals:
        ax.set_visible(False)
        return

    lo = min(v.min() for v in all_vals)
    hi = max(v.max() for v in all_vals)
    if lo == hi:
        hi = lo + 1.0
    hist_bins = np.linspace(lo, hi, bins + 1)

    ax.hist(ori_v, bins=hist_bins, density=True, alpha=0.30, color=ORI_COLOR, label="ori")
    if kde and ori_v.size >= 2:
        try:
            from scipy.stats import gaussian_kde

            xs = np.linspace(lo, hi, 200)
            ax.plot(xs, gaussian_kde(ori_v)(xs), color=ORI_COLOR, lw=1.5)
        except Exception:
            pass

    for color, (name, vals) in zip(colors, base_v.items()):
        if vals.size == 0:
            continue
        ax.hist(vals, bins=hist_bins, density=True, alpha=0.25, color=color, label=name)
        if kde and vals.size >= 2:
            try:
                from scipy.stats import gaussian_kde

                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, gaussian_kde(vals)(xs), color=color, lw=1.5)
            except Exception:
                pass

    w1_text = ", ".join(f"{n}:{_format_float(w1_by_name[n], 3)}" for n in baseline_vals)
    ax.set_title(f"{key}\nW1({w1_text})", fontsize=9)
    if use_log_x:
        ax.set_xscale("log")
    ax.tick_params(labelsize=8)


def plot_anomaly_distribution_multi(
    multi_result: Dict[str, Any],
    *,
    baseline_names: Optional[Sequence[str]] = None,
    features: Sequence[str] = SAMPLE_FEATURE_KEYS,
    figsize_per_panel: tuple = (3.4, 2.6),
    bins: int = 40,
    kde: bool = True,
    log_energy: bool = True,
    pooled_subsample: int = 200_000,
    tail_xmax: Optional[float] = None,
    qq_points: int = 5000,
    random_state: int = 42,
):
    """
    Visualize ori vs multiple baselines on shared axes.

    Use ``baseline_names`` to plot a subset (e.g. only ``["gen", "gen1"]``) while
    ``multi_result`` may contain more baselines for tabular summary.

    Returns (fig_features, fig_ccdf, fig_qq) when ``store_pooled_arrays=True`` was used,
    otherwise only fig_features.
    """
    import matplotlib.pyplot as plt

    names = list(baseline_names or multi_result["baselines"].keys())
    colors = BASELINE_COLORS[: len(names)]
    rng = np.random.default_rng(random_state)

    first = multi_result["baselines"][names[0]]
    per_ori = first["per_sample"]["ori"]

    n_feat = len(features)
    ncols = 4
    nrows = int(np.ceil(n_feat / ncols))
    fig1, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, key in zip(axes.ravel(), features):
        baseline_vals = {
            n: multi_result["baselines"][n]["per_sample"]["gen"][key] for n in names
        }
        w1_by_name = {
            n: multi_result["baselines"][n]["sample_features"][key]["wasserstein"]
            for n in names
        }
        _plot_feature_panel(
            ax,
            per_ori[key],
            baseline_vals,
            key=key,
            w1_by_name=w1_by_name,
            colors=colors,
            bins=bins,
            kde=kde,
            log_energy=log_energy,
        )
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig1.suptitle(
        f"Marginals (N_ori={multi_result['n_ori']})",
        y=0.995,
        fontsize=11,
    )
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(4, len(names) + 1),
        frameon=False,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.94])

    has_pooled = all(
        "pooled_arrays" in multi_result["baselines"][n] for n in names
    )
    if not has_pooled:
        fig1.text(
            0.5,
            0.01,
            "Tip: pass store_pooled_arrays=True to also get CCDF / QQ plots.",
            ha="center",
            fontsize=9,
            transform=fig1.transFigure,
        )
        return fig1

    ref = multi_result["baselines"][names[0]]["pooled_arrays"]
    abs_ori = _subsample_1d(ref["abs_r_ori"], pooled_subsample, rng)
    anom_ori = _subsample_1d(ref["abs_r_anom_ori"], pooled_subsample, rng)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    tau = multi_result["config"]["tau"]
    for ax, (ori_arr, title, key) in zip(
        axes2,
        (
            (abs_ori, "All timesteps: CCDF of |r|", "abs_r_gen"),
            (anom_ori, f"Anomaly only (|r|>{tau}): CCDF", "abs_r_anom_gen"),
        ),
    ):
        x, y = _ccdf(ori_arr)
        if tail_xmax is not None:
            keep = x <= tail_xmax
            x, y = x[keep], y[keep]
        ax.plot(x, y, label="ori", color=ORI_COLOR, lw=1.8)
        for color, name in zip(colors, names):
            pooled = multi_result["baselines"][name]["pooled_arrays"]
            vals = _subsample_1d(pooled[key], pooled_subsample, rng)
            xb, yb = _ccdf(vals)
            if tail_xmax is not None:
                keep = xb <= tail_xmax
                xb, yb = xb[keep], yb[keep]
            ax.plot(xb, yb, label=name, color=color, lw=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("|r|")
        ax.set_ylabel("P(|r| > x)")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
    fig2.suptitle("Tail survival (log y)", y=1.02, fontsize=11)
    fig2.tight_layout()

    ncols_qq = len(names)
    fig3, axes3 = plt.subplots(1, ncols_qq, figsize=(4.2 * ncols_qq, 4.2), squeeze=False)
    q_grid = np.linspace(0.001, 0.999, qq_points)
    q_ori = np.quantile(abs_ori, q_grid)
    for ax, color, name in zip(axes3.ravel(), colors, names):
        pooled = multi_result["baselines"][name]["pooled_arrays"]
        abs_gen = _subsample_1d(pooled["abs_r_gen"], pooled_subsample, rng)
        q_gen = np.quantile(abs_gen, q_grid)
        ax.scatter(q_ori, q_gen, s=6, alpha=0.5, c=color, edgecolors="none")
        lim = max(q_ori.max(), q_gen.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#999999", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("ori quantile")
        ax.set_ylabel(f"{name} quantile")
        ax.set_title(f"QQ: {name} vs ori", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig3.tight_layout()

    return fig1, fig2, fig3


def build_baselines(
    sources: Mapping[str, Union[np.ndarray, str, Path]],
    active: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Select baselines by name from in-memory arrays or .npy paths.

    Parameters
    ----------
    sources : mapping
        name -> array (N,T,1) or path to .npy
    active : sequence, optional
        Subset/order of names to compare; default = all keys in ``sources``.
    """
    names = list(active) if active is not None else list(sources.keys())
    if not names:
        raise ValueError("active baselines must contain at least one name")
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name not in sources:
            raise KeyError(f"baseline {name!r} not found in sources: {list(sources)}")
        src = sources[name]
        if isinstance(src, (str, Path)):
            out[name] = np.load(str(src))
        else:
            out[name] = np.asarray(src)
    return out


def compare_and_print_multi(
    ori: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run multi-baseline comparison, print reports, return result dict."""
    result = compare_anomaly_distribution_multi(ori, baselines, **kwargs)
    print_anomaly_distribution_multi_report(result)
    return result


def _format_float(x: Union[float, int], digits: int = 4) -> str:
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "nan"
    return f"{float(x):.{digits}f}"


def print_anomaly_distribution_report(
    result: Dict[str, Any],
    title: str = "Anomaly distribution comparison (unpaired)",
) -> None:
    """Print a compact text report from ``compare_anomaly_distribution`` output."""
    cfg = result["config"]
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(
        f"  N_ori={result['n_ori']}  N_gen={result['n_gen']}  "
        f"T={cfg['seq_length']}  tau={cfg['tau']}"
    )
    print("-" * 72)
    print(f"{'feature':24s} {'ori_mean':>10s} {'gen_mean':>10s} {'W1':>10s}")
    print("-" * 72)
    for key in SAMPLE_FEATURE_KEYS:
        block = result["sample_features"][key]
        print(
            f"{key:24s} "
            f"{_format_float(block['ori']['mean']):>10s} "
            f"{_format_float(block['gen']['mean']):>10s} "
            f"{_format_float(block['wasserstein']):>10s}"
        )

    print("-" * 72)
    print("Pooled |r| quantiles (all timesteps)")
    for label in ("ori", "gen"):
        q = result["pooled"]["abs_r"][label]
        parts = [f"{k}={_format_float(v)}" for k, v in q.items()]
        print(f"  {label:4s}: " + ", ".join(parts))
    print(f"  W1(all |r|) = {_format_float(result['pooled']['abs_r']['wasserstein'])}")

    print("-" * 72)
    pa = result["pooled"]["abs_r_anomaly"]
    print(
        f"Pooled |r| on anomaly points (|r|>tau); "
        f"n_ori={pa['n_points_ori']} n_gen={pa['n_points_gen']}"
    )
    for label in ("ori", "gen"):
        q = pa[label]
        parts = [f"{k}={_format_float(v)}" for k, v in q.items()]
        print(f"  {label:4s}: " + ", ".join(parts))
    print(f"  W1(anomaly |r|) = {_format_float(pa['wasserstein'])}")
    print("=" * 72)


def compare_and_print(
    ori: np.ndarray,
    gen: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run comparison and print report; returns the result dict."""
    result = compare_anomaly_distribution(ori, gen, **kwargs)
    print_anomaly_distribution_report(result)
    return result


def _subsample_1d(x: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    v = _finite_values(x)
    if v.size <= max_points:
        return v
    idx = rng.choice(v.size, size=max_points, replace=False)
    return v[idx]


def _ecdf(values: np.ndarray):
    v = np.sort(_finite_values(values))
    if v.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, v.size + 1) / v.size
    return v, y


def _ccdf(values: np.ndarray):
    v = np.sort(_finite_values(values))
    if v.size == 0:
        return np.array([]), np.array([])
    y = 1.0 - np.arange(1, v.size + 1) / v.size
    y = np.maximum(y, 1.0 / v.size)
    return v, y


def plot_anomaly_distribution(
    result: Dict[str, Any],
    *,
    features: Sequence[str] = SAMPLE_FEATURE_KEYS,
    figsize_per_panel: tuple = (3.4, 2.6),
    bins: int = 40,
    kde: bool = True,
    log_energy: bool = True,
    pooled_subsample: int = 200_000,
    tail_xmax: Optional[float] = None,
    qq_points: int = 5000,
    random_state: int = 42,
):
    """
    Visualize unpaired anomaly-distribution comparison.

    Recommended layout (3 figures returned):

    1. **Per-sample marginals** — KDE/hist overlay for each scalar feature.
       Answers: "Does each series-level statistic match ori?"
    2. **Pooled |r| CCDF** — log-scale tail survival curves (all points + anomaly subset).
       Answers: "Is the extreme tail heavy enough?" (your main gap)
    3. **QQ plot** — quantiles of gen vs ori on pooled |r| (subsampled).
       Answers: "Where does gen deviate from ori?" (diagonal = match)

    Parameters
    ----------
    result : dict
        Output of ``compare_anomaly_distribution(..., store_pooled_arrays=True)``.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_state)
    per_ori = result["per_sample"]["ori"]
    per_gen = result["per_sample"]["gen"]

    n_feat = len(features)
    ncols = 4
    nrows = int(np.ceil(n_feat / ncols))
    fig1, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, key in zip(axes.ravel(), features):
        a = _finite_values(per_ori[key])
        b = _finite_values(per_gen[key])
        w1 = result["sample_features"][key]["wasserstein"]
        use_log_x = log_energy and key == "anomaly_energy"
        if use_log_x:
            a = a[a > 0]
            b = b[b > 0]
        if a.size == 0 or b.size == 0:
            ax.set_visible(False)
            continue
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        if lo == hi:
            hi = lo + 1.0
        hist_bins = np.linspace(lo, hi, bins + 1)
        ax.hist(a, bins=hist_bins, density=True, alpha=0.35, color="#2166ac", label="ori")
        ax.hist(b, bins=hist_bins, density=True, alpha=0.35, color="#b2182b", label="gen")
        if kde:
            try:
                from scipy.stats import gaussian_kde

                xs = np.linspace(lo, hi, 200)
                if a.size >= 2:
                    ax.plot(xs, gaussian_kde(a)(xs), color="#2166ac", lw=1.5)
                if b.size >= 2:
                    ax.plot(xs, gaussian_kde(b)(xs), color="#b2182b", lw=1.5)
            except Exception:
                pass
        ax.set_title(f"{key}\nW1={_format_float(w1, 3)}", fontsize=10)
        ax.set_xlabel("log10" if use_log_x else "value")
        if use_log_x:
            ax.set_xscale("log")
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig1.suptitle(
        f"Marginals (N_ori={result['n_ori']}, N_gen={result['n_gen']})",
        y=0.995,
        fontsize=11,
    )
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.94])

    pooled = result.get("pooled_arrays")
    if pooled is None:
        fig1.text(
            0.5,
            0.01,
            "Tip: pass store_pooled_arrays=True to also get CCDF / QQ plots.",
            ha="center",
            fontsize=9,
            transform=fig1.transFigure,
        )
        return fig1

    abs_ori = _subsample_1d(pooled["abs_r_ori"], pooled_subsample, rng)
    abs_gen = _subsample_1d(pooled["abs_r_gen"], pooled_subsample, rng)
    anom_ori = _subsample_1d(pooled["abs_r_anom_ori"], pooled_subsample, rng)
    anom_gen = _subsample_1d(pooled["abs_r_anom_gen"], pooled_subsample, rng)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (a, b, title) in zip(
        axes2,
        (
            (abs_ori, abs_gen, "All timesteps: CCDF of |r|"),
            (anom_ori, anom_gen, f"Anomaly only (|r|>{result['config']['tau']}): CCDF"),
        ),
    ):
        for vals, label, color in ((a, "ori", "#2166ac"), (b, "gen", "#b2182b")):
            x, y = _ccdf(vals)
            if x.size == 0:
                continue
            if tail_xmax is not None:
                keep = x <= tail_xmax
                x, y = x[keep], y[keep]
            ax.plot(x, y, label=label, color=color, lw=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("|r|")
        ax.set_ylabel("P(|r| > x)")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
    fig2.suptitle("Tail survival (log y): upper-right = heavier tail", y=1.02, fontsize=11)
    fig2.tight_layout()

    n = min(qq_points, abs_ori.size, abs_gen.size)
    q_grid = np.linspace(0.001, 0.999, n)
    q_ori = np.quantile(abs_ori, q_grid)
    q_gen = np.quantile(abs_gen, q_grid)
    fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
    ax3.scatter(q_ori, q_gen, s=6, alpha=0.5, c="#4d4d4d", edgecolors="none")
    lim = max(q_ori.max(), q_gen.max()) * 1.05
    ax3.plot([0, lim], [0, lim], "--", color="#999999", lw=1)
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.set_xlabel("ori quantile of |r|")
    ax3.set_ylabel("gen quantile of |r|")
    ax3.set_title("QQ: pooled |r| (subsampled)", fontsize=10)
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, alpha=0.25)
    fig3.tight_layout()

    return fig1, fig2, fig3


DEFAULT_PLOT_BASELINES = ("gen", "gen1")


def plot_anomaly_distribution_from_table(
    multi_result: Dict[str, Any],
    plot_baselines: Sequence[str] = DEFAULT_PLOT_BASELINES,
    features: Sequence[str] = SUMMARY_TABLE_FEATURE_KEYS,
    **kwargs: Any,
):
    """Plot ori vs a fixed small set of baselines (default: gen, gen1).

    Feature panels match ``anomaly_distribution_summary_table`` (no max_abs_r).
    """
    return plot_anomaly_distribution_multi(
        multi_result,
        baseline_names=list(plot_baselines),
        features=features,
        **kwargs,
    )


__all__ = [
    "compare_anomaly_distribution",
    "compare_anomaly_distribution_multi",
    "compare_and_print",
    "compare_and_print_multi",
    "print_anomaly_distribution_report",
    "print_anomaly_distribution_multi_report",
    "anomaly_distribution_summary_table",
    "plot_anomaly_distribution",
    "plot_anomaly_distribution_multi",
    "plot_anomaly_distribution_from_table",
    "build_baselines",
    "SAMPLE_FEATURE_KEYS",
    "SUMMARY_TABLE_FEATURE_KEYS",
    "DEFAULT_TAU",
]
