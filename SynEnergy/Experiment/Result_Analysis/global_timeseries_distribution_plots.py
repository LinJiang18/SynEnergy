"""
Temporal-aware global distribution plots for ori vs gen time series.

Complements scalar metrics in global_timeseries_distribution_metrics.py.
Designed for long sequences (e.g. T=180, 30 days x 6 slots) where flattened KDE / PCA
lose phase and dynamics structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from Experiment.Result_Analysis.anomaly_distribution_metrics import (
    BASELINE_COLORS,
    ORI_COLOR,
)
from Experiment.Result_Analysis.global_timeseries_distribution_metrics import (
    _acf_at_lag,
    compute_distribution_distance_detail,
    dist_wasserstein_flat,
)

ArrayLike = Union[np.ndarray, Sequence[float]]

LINE_STYLES = ("-", "--", "-.", ":")


def _as_3d(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T) or (N, T, C), got {arr.shape}")
    return arr


def _series_2d(x: np.ndarray, channel: int = 0) -> np.ndarray:
    return _as_3d(x)[..., channel]


def _num_days(t_len: int, slots_per_day: int) -> int:
    if t_len % slots_per_day != 0:
        raise ValueError(f"T={t_len} not divisible by slots_per_day={slots_per_day}")
    return t_len // slots_per_day


def _reshape_days_slots(x2d: np.ndarray, slots_per_day: int) -> np.ndarray:
    """(N, T) -> (N, num_days, slots_per_day)."""
    n, t = x2d.shape
    d = _num_days(t, slots_per_day)
    return x2d.reshape(n, d, slots_per_day)


def _mean_acf_curve(data2d: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean ACF across samples at lags 1..max_lag."""
    _, t = data2d.shape
    m = min(max_lag, max(0, t - 2))
    lags = np.arange(1, m + 1, dtype=int)
    if m < 1:
        return lags, np.zeros(0, dtype=np.float64)
    vals = np.zeros(m, dtype=np.float64)
    for j, lag in enumerate(lags):
        per = [_acf_at_lag(data2d[i], int(lag)) for i in range(data2d.shape[0])]
        per = [p for p in per if np.isfinite(p)]
        vals[j] = float(np.mean(per)) if per else float("nan")
    return lags, vals


def _baseline_style(index: int) -> Tuple[str, str]:
    return BASELINE_COLORS[index % len(BASELINE_COLORS)], LINE_STYLES[index % len(LINE_STYLES)]


def _kde_line(ax, values: np.ndarray, *, color: str, label: str, lw: float = 1.8, ls: str = "-") -> None:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    try:
        from scipy.stats import gaussian_kde

        lo, hi = float(v.min()), float(v.max())
        if lo == hi:
            hi = lo + 1.0
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, gaussian_kde(v)(xs), color=color, lw=lw, ls=ls, label=label)
    except Exception:
        ax.hist(v, bins=40, density=True, alpha=0.35, color=color, label=label)


def _ecdf(ax, values: np.ndarray, *, color: str, label: str, lw: float = 1.8, ls: str = "-") -> None:
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    y = np.arange(1, v.size + 1) / v.size
    ax.plot(v, y, color=color, lw=lw, ls=ls, label=label)


def plot_global_distribution_overlay(
    ori: ArrayLike,
    baselines: Mapping[str, ArrayLike],
    *,
    baseline_names: Optional[Sequence[str]] = None,
    slots_per_day: int = 6,
    max_lag: int = 50,
    channel: int = 0,
    n_spaghetti: int = 30,
    violin_timesteps: Optional[Sequence[int]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Multi-figure global distribution report with **ori + all baselines on the same axes**.

    Returns dict: detail_by_baseline, figures, names, figure_dict.
    """
    import matplotlib.pyplot as plt

    names = list(baseline_names or baselines.keys())
    if not names:
        raise ValueError("baselines must contain at least one entry")

    ori2 = _series_2d(ori, channel)
    gens2: Dict[str, np.ndarray] = {n: _series_2d(baselines[n], channel) for n in names}
    n_ori, t_len = ori2.shape
    rng = np.random.default_rng(random_state)
    t = np.arange(t_len)
    slots = np.arange(slots_per_day)

    details: Dict[str, Dict[str, Any]] = {}
    for name in names:
        details[name] = compute_distribution_distance_detail(
            ori2[..., None],
            gens2[name][..., None],
            slots_per_day=slots_per_day,
            max_lag=max_lag,
        )

    if violin_timesteps is None:
        violin_timesteps = list(np.linspace(0, t_len - 1, 6, dtype=int))

    figures: List[Any] = []
    fig_names: List[str] = []

    # ----- Fig 1: profile -----
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 8))

    ax = axes1[0, 0]
    q10, q50, q90 = np.quantile(ori2, [0.1, 0.5, 0.9], axis=0)
    ax.plot(q50, color=ORI_COLOR, lw=2.2, label="ori median")
    ax.fill_between(t, q10, q90, color=ORI_COLOR, alpha=0.12)
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        g = gens2[name]
        q10, q50, q90 = np.quantile(g, [0.1, 0.5, 0.9], axis=0)
        ax.plot(q50, color=color, lw=2, ls=ls, label=f"{name} median")
        ax.fill_between(t, q10, q90, color=color, alpha=0.08)
    ax.set_title("Median trajectory + IQR (ori vs all gens)")
    ax.set_xlabel("timestep")
    ax.legend(frameon=False, fontsize=8)

    ax = axes1[0, 1]
    ax.plot(np.median(ori2, axis=0), color=ORI_COLOR, lw=2, label="ori")
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        ax.plot(np.median(gens2[name], axis=0), color=color, lw=2, ls=ls, label=name)
    ax.set_title("Median trajectories overlay")
    ax.set_xlabel("timestep")
    ax.legend(frameon=False, fontsize=8)

    ax = axes1[1, 0]
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        tw = details[name]["TW1_per_t"]
        ax.plot(tw, color=color, lw=1.8, ls=ls, label=f"{name} (μ={details[name]['TW1_mean']:.3f})")
    ax.set_title("Per-timestep marginal W1 (lower is better)")
    ax.set_xlabel("timestep")
    ax.set_ylabel("W1")
    ax.legend(frameon=False, fontsize=8)

    ax = axes1[1, 1]
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        ax.plot(gens2[name].mean(axis=0) - ori2.mean(axis=0), color=color, lw=1.8, ls=ls, label=name)
    ax.axhline(0.0, color="#999999", ls="--", lw=1)
    ax.set_title("Mean gap (gen − ori) per timestep")
    ax.set_xlabel("timestep")
    ax.legend(frameon=False, fontsize=8)
    fig1.tight_layout()
    figures.append(fig1)
    fig_names.append("profile_TW1")

    # ----- Fig 2: dynamics -----
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes2[0]
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        ax.plot(details[name]["DW1_per_t"], color=color, lw=1.8, ls=ls, label=f"{name} (μ={details[name]['DW1_mean']:.3f})")
    ax.set_title("Increment W1 per step")
    ax.set_xlabel("step index")
    ax.legend(frameon=False, fontsize=8)

    ax = axes2[1]
    _kde_line(ax, np.diff(ori2, axis=1).ravel(), color=ORI_COLOR, label="ori Δx", lw=2)
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        _kde_line(ax, np.diff(gens2[name], axis=1).ravel(), color=color, label=f"{name} Δx", ls=ls)
    ax.set_title("Pooled increment Δx")
    ax.legend(frameon=False, fontsize=8)

    ax = axes2[2]
    lags, acf_o = _mean_acf_curve(ori2, max_lag)
    if lags.size:
        ax.plot(lags, acf_o, color=ORI_COLOR, lw=2.2, label="ori")
        for i, name in enumerate(names):
            color, ls = _baseline_style(i)
            _, acf_g = _mean_acf_curve(gens2[name], max_lag)
            ax.plot(lags, acf_g, color=color, lw=1.8, ls=ls, label=f"{name} (d={details[name]['ACF_dist']:.4f})")
    ax.set_title("Mean ACF")
    ax.set_xlabel("lag")
    ax.legend(frameon=False, fontsize=7)
    fig2.tight_layout()
    figures.append(fig2)
    fig_names.append("dynamics_ACF")

    # ----- Fig 3: slot structure -----
    n_g = len(names)
    fig3, axes3 = plt.subplots(3, 3, figsize=(15, 11))

    ax_bar = axes3[0, 0]
    width = 0.75 / max(n_g, 1)
    for i, name in enumerate(names):
        offset = (i - (n_g - 1) / 2) * width
        ax_bar.bar(slots + offset, details[name]["slot_TW1_per_slot"], width=width, color=_baseline_style(i)[0], label=name)
    ax_bar.set_title("Slot-pooled W1 by baseline")
    ax_bar.set_xlabel("slot (t mod S)")
    ax_bar.set_ylabel("W1")
    ax_bar.legend(frameon=False, fontsize=8)

    slot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
    for s, (r, c) in zip(slots, slot_positions):
        ax = axes3[r, c]
        idx = np.arange(s, t_len, slots_per_day)
        _kde_line(ax, ori2[:, idx].ravel(), color=ORI_COLOR, label="ori", lw=1.8)
        for i, name in enumerate(names):
            color, ls = _baseline_style(i)
            _kde_line(ax, gens2[name][:, idx].ravel(), color=color, label=name, lw=1.5, ls=ls)
        ax.set_title(f"Slot {s}")
        if s == 0:
            ax.legend(frameon=False, fontsize=7)
    for ax in (axes3[2, 1], axes3[2, 2]):
        ax.set_visible(False)
    fig3.tight_layout()
    figures.append(fig3)
    fig_names.append("slot_marginals")

    # ----- Fig 4: heatmaps (ori + each gen + diffs) -----
    mean_ori = _reshape_days_slots(ori2, slots_per_day).mean(axis=0)
    n_cols = 1 + n_g + n_g
    fig4, axes4 = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.2))
    if n_cols == 1:
        axes4 = [axes4]
    vmax = mean_ori.max()
    for g in gens2.values():
        vmax = max(vmax, _reshape_days_slots(g, slots_per_day).mean(axis=0).max())
    vmax = max(vmax, 1e-6)

    im0 = axes4[0].imshow(mean_ori, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    axes4[0].set_title("ori mean")
    axes4[0].set_xlabel("slot")
    axes4[0].set_ylabel("day")
    fig4.colorbar(im0, ax=axes4[0], fraction=0.046)

    for i, name in enumerate(names):
        mean_g = _reshape_days_slots(gens2[name], slots_per_day).mean(axis=0)
        ax = axes4[1 + i]
        im = ax.imshow(mean_g, aspect="auto", cmap="Oranges", vmin=0, vmax=vmax)
        ax.set_title(f"{name} mean")
        ax.set_xlabel("slot")
        fig4.colorbar(im, ax=ax, fraction=0.046)

    for i, name in enumerate(names):
        diff = _reshape_days_slots(gens2[name], slots_per_day).mean(axis=0) - mean_ori
        lim = max(np.abs(diff).max(), 1e-6)
        ax = axes4[1 + n_g + i]
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{name} − ori")
        ax.set_xlabel("slot")
        fig4.colorbar(im, ax=ax, fraction=0.046)
    fig4.tight_layout()
    figures.append(fig4)
    fig_names.append("heatmap_day_slot")

    # ----- Fig 5: daily / pooled -----
    fig5, axes5 = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes5[0, 0]
    _kde_line(ax, _reshape_days_slots(ori2, slots_per_day).mean(axis=2).ravel(), color=ORI_COLOR, label="ori", lw=2)
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        d = _reshape_days_slots(gens2[name], slots_per_day).mean(axis=2).ravel()
        _kde_line(ax, d, color=color, label=name, ls=ls)
    ax.set_title("Daily mean load KDE")
    ax.legend(frameon=False, fontsize=8)

    ax = axes5[0, 1]
    d_ori = _reshape_days_slots(ori2, slots_per_day).mean(axis=2)
    days = np.arange(d_ori.shape[1])
    ax.plot(days, np.median(d_ori, axis=0), color=ORI_COLOR, lw=2, label="ori")
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        d = _reshape_days_slots(gens2[name], slots_per_day).mean(axis=2)
        ax.plot(days, np.median(d, axis=0), color=color, lw=2, ls=ls, label=name)
    ax.set_title("Daily mean trajectory (medians)")
    ax.set_xlabel("day")
    ax.legend(frameon=False, fontsize=8)

    ax = axes5[1, 0]
    _ecdf(ax, ori2.ravel(), color=ORI_COLOR, label="ori", lw=2)
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        w = dist_wasserstein_flat(ori2[..., None], gens2[name][..., None])
        _ecdf(ax, gens2[name].ravel(), color=color, label=f"{name} (W_flat={w:.3f})", ls=ls)
    ax.set_title("Pooled value ECDF")
    ax.legend(frameon=False, fontsize=7)

    ax = axes5[1, 1]
    metrics = ["TW1_mean", "DW1_mean", "slot_TW1_mean", "ACF_dist", "MMD"]
    x = np.arange(len(metrics))
    width = 0.75 / max(n_g, 1)
    for i, name in enumerate(names):
        vals = [details[name][m] for m in metrics]
        ax.bar(x + (i - (n_g - 1) / 2) * width, vals, width=width, color=_baseline_style(i)[0], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["TW1", "DW1", "slot_TW1", "ACF", "MMD"], rotation=15, ha="right")
    ax.set_title("Scalar metrics (lower better)")
    ax.legend(frameon=False, fontsize=8)
    fig5.tight_layout()
    figures.append(fig5)
    fig_names.append("daily_pooled_metrics")

    # ----- Fig 6: mean spaghetti overlay -----
    fig6, ax6 = plt.subplots(figsize=(14, 4.5))
    n_show = min(n_spaghetti, n_ori)
    idx_o = rng.choice(n_ori, size=n_show, replace=False) if n_ori else np.array([], dtype=int)
    for i in idx_o:
        ax6.plot(t, ori2[i], color=ORI_COLOR, alpha=0.06, lw=0.6)
    ax6.plot(t, ori2.mean(axis=0), color=ORI_COLOR, lw=2.5, label="ori mean")
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        g = gens2[name]
        n_show_g = min(n_spaghetti, g.shape[0])
        idx_g = rng.choice(g.shape[0], size=n_show_g, replace=False)
        for j in idx_g:
            ax6.plot(t, g[j], color=color, alpha=0.05, lw=0.6)
        ax6.plot(t, g.mean(axis=0), color=color, lw=2.2, ls=ls, label=f"{name} mean")
    ax6.set_title("Spaghetti (faint) + mean curves overlay")
    ax6.set_xlabel("timestep")
    ax6.legend(frameon=False, fontsize=9)
    fig6.tight_layout()
    figures.append(fig6)
    fig_names.append("spaghetti_overlay")

    # ----- Fig 7: grouped violins -----
    fig7, ax7 = plt.subplots(figsize=(max(12, 2.2 * len(violin_timesteps)), 4.8))
    group_width = 1 + n_g
    all_positions: List[float] = []
    all_data: List[np.ndarray] = []
    all_colors: List[str] = []
    tick_pos: List[float] = []
    tick_lab: List[str] = []

    for gi, ti in enumerate(violin_timesteps):
        base = gi * (group_width + 1)
        tick_pos.append(base + n_g / 2)
        tick_lab.append(f"t={ti}")
        all_positions.append(base)
        all_data.append(ori2[:, ti])
        all_colors.append(ORI_COLOR)
        for i, name in enumerate(names):
            all_positions.append(base + 1 + i)
            all_data.append(gens2[name][:, ti])
            all_colors.append(_baseline_style(i)[0])

    parts = ax7.violinplot(all_data, positions=all_positions, showmeans=False, showmedians=True, widths=0.78)
    for body, c in zip(parts["bodies"], all_colors):
        body.set_facecolor(c)
        body.set_alpha(0.55)
    ax7.set_xticks(tick_pos)
    ax7.set_xticklabels(tick_lab)
    ax7.set_title("Violins at selected t: ori then baselines (each group)")
    ax7.set_ylabel("load")
    handles = [plt.Line2D([0], [0], color=ORI_COLOR, lw=3, label="ori")]
    handles += [plt.Line2D([0], [0], color=_baseline_style(i)[0], lw=3, label=n) for i, n in enumerate(names)]
    ax7.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig7.tight_layout()
    figures.append(fig7)
    fig_names.append("violin_selected_t")

    # ----- Fig 8: TW1 overlay -----
    fig8, ax8 = plt.subplots(figsize=(13, 4))
    for i, name in enumerate(names):
        color, ls = _baseline_style(i)
        ax8.plot(details[name]["TW1_per_t"], color=color, lw=1.8, ls=ls, label=name)
    ax8.set_title("TW1(t) overlay — which baseline drifts from ori where")
    ax8.set_xlabel("timestep")
    ax8.set_ylabel("W1")
    ax8.legend(frameon=False, fontsize=9)
    fig8.tight_layout()
    figures.append(fig8)
    fig_names.append("TW1_overlay")

    return {
        "detail_by_baseline": details,
        "figures": figures,
        "names": fig_names,
        "figure_dict": dict(zip(fig_names, figures)),
        "baseline_names": names,
    }


def plot_global_distribution_comprehensive(
    ori: ArrayLike,
    gen: ArrayLike,
    *,
    gen_name: str = "Synthetic",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Single-baseline wrapper around overlay plots."""
    out = plot_global_distribution_overlay(ori, {gen_name: gen}, baseline_names=[gen_name], **kwargs)
    name = out["baseline_names"][0]
    return {
        "detail": out["detail_by_baseline"][name],
        "figures": out["figures"],
        "names": out["names"],
        "figure_dict": out["figure_dict"],
    }


def plot_global_distribution_multi_comprehensive(
    ori: ArrayLike,
    baselines: Mapping[str, ArrayLike],
    *,
    baseline_names: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """All baselines on shared axes (preferred for gen1 vs gen2 comparison)."""
    out = plot_global_distribution_overlay(
        ori,
        baselines,
        baseline_names=baseline_names,
        **kwargs,
    )
    return {
        "per_baseline": {n: {"detail": out["detail_by_baseline"][n]} for n in out["baseline_names"]},
        "detail_by_baseline": out["detail_by_baseline"],
        "comparison_figure": out["figure_dict"]["daily_pooled_metrics"],
        "figures": out["figures"],
        "names": out["names"],
        "figure_dict": out["figure_dict"],
        "baseline_names": out["baseline_names"],
    }


def show_global_distribution_report(
    ori: ArrayLike,
    baselines: Mapping[str, ArrayLike],
    *,
    baseline_names: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate overlay figures and plt.show() each (notebook-friendly)."""
    import matplotlib.pyplot as plt

    result = plot_global_distribution_overlay(
        ori,
        baselines,
        baseline_names=baseline_names,
        **kwargs,
    )
    for fig_name, fig in result["figure_dict"].items():
        print(f"--- {fig_name} ---")
        plt.figure(fig.number)
        plt.show()
    return result


__all__ = [
    "plot_global_distribution_overlay",
    "plot_global_distribution_comprehensive",
    "plot_global_distribution_multi_comprehensive",
    "show_global_distribution_report",
]
