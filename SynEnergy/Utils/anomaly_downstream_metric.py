"""TSTR anomaly detection / prediction downstream fidelity (PyTorch).

Train a Transformer binary classifier on **generated** sliding windows; evaluate on
**original** windows.  Labels follow ``anomaly_distribution_metrics``:

    standardized residual r from ``extract_standardized_residual``
    y_t = 1[|r_t| > tau]

**Detection** — context ``x[t-lookback:t]`` predicts onset at ``t``:
    ``y_t = 1[|r_t|>τ and |r_{t-1}|≤τ]`` (first point of each anomaly run).
**Prediction** — same context predicts ``y_{t+horizon}`` (future point anomaly).

Returns PR-AUC (higher is better). ``downstream_summary_table`` reports
``Det-PRAUC`` / ``Pred-PRAUC`` with an ``oracle`` reference row.
Pass ``detection_label=\"point\"`` for legacy contemporaneous point detection.
Use ``detection_difficulty`` (``\"hard\"`` / ``\"severe\"``) for stricter onset detection.

Speed defaults: subsampled train/eval windows (50k), 100 training iterations,
batch_size 256, cached ori eval set inside ``downstream_summary_table``.
Pass ``max_train_windows=None`` / ``max_eval_windows=None`` for full data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from Utils.metric_utils import extract_time
from Utils.predictive_metric import _TransformerContextEncoder, _d_model_nhead_dimff

try:
    from controlnet.ae.cond_grouped_residual import (
        apply_cond_group_residual,
        fit_cond_group_zone_stats,
    )
    from controlnet.ae.residual_learner import extract_standardized_residual
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from controlnet.ae.cond_grouped_residual import (  # noqa: E402
        apply_cond_group_residual,
        fit_cond_group_zone_stats,
    )
    from controlnet.ae.residual_learner import extract_standardized_residual  # noqa: E402


TrainSource = Literal["gen", "ori"]
DetectionLabel = Literal["point", "onset"]
DetectionDifficulty = Literal["standard", "hard", "severe"]

DOWNSTREAM_METRIC_KEYS = ("Det-PRAUC", "Pred-PRAUC")
DOWNSTREAM_GAP_KEYS = ("Det-gap", "Pred-gap")

DEFAULT_MAX_TRAIN_WINDOWS = 50_000
DEFAULT_MAX_EVAL_WINDOWS = 50_000
DEFAULT_ITERATIONS = 100
DEFAULT_BATCH_SIZE = 256


class _TransformerBinaryClassifier(nn.Module):
    """Transformer encoder on (B, L, C) context -> scalar logit."""

    def __init__(
        self,
        lookback: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.input_dim = int(input_dim)
        d_model, nhead, dim_ff = _d_model_nhead_dimff(hidden_dim)
        self.encoder = _TransformerContextEncoder(
            lookback, d_model, nhead, dim_ff, num_layers, dropout
        )
        if input_dim != 1:
            self.in_adapter = nn.Linear(input_dim, 1)
        else:
            self.in_adapter = None
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_adapter is not None:
            x = self.in_adapter(x)
        h = self.encoder(x)
        return self.head(h[:, -1, :]).squeeze(-1)


@dataclass
class _WindowPair:
    ctx: np.ndarray
    y: np.ndarray


@dataclass
class _OriEvalCache:
    det: _WindowPair
    pred: _WindowPair


def _anomaly_labels(
    data: np.ndarray,
    tau: float,
    num_days: int = 30,
    slots_per_day: int = 6,
    residual_eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    *,
    cond: Optional[np.ndarray] = None,
    zone_stats: Optional[Dict] = None,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Binary labels (N, T, 1) from |standardized residual| > tau."""
    data = np.asarray(data, dtype=np.float32)
    if zone_stats is not None:
        if cond is None:
            raise ValueError("cond is required when zone_stats is provided")
        data_r = apply_cond_group_residual(
            data, cond, zone_stats, eps=residual_eps
        )
    else:
        data_r, _, _, _, _ = extract_standardized_residual(
            data,
            num_days=num_days,
            slots_per_day=slots_per_day,
            eps=residual_eps,
            min_scale=min_scale,
            use_mad_consistency=use_mad_consistency,
            baseline=baseline,
        )
    return (np.abs(data_r) > float(tau)).astype(np.float32)


@dataclass(frozen=True)
class _DetectionSettings:
    detection_tau: float
    detection_lookback: int
    detection_min_normal_run: int
    detection_label: DetectionLabel
    difficulty: DetectionDifficulty


def _filter_onset_cold_start(
    onset_labels: np.ndarray,
    point_labels: np.ndarray,
    min_normal_run: int,
) -> np.ndarray:
    """Keep onset at ``t`` only if the previous ``min_normal_run`` steps are normal."""
    k = int(min_normal_run)
    if k <= 1:
        return onset_labels
    p = (np.asarray(point_labels, dtype=np.float32) > 0.5).astype(np.float32)
    o = np.asarray(onset_labels, dtype=np.float32).copy()
    for j in range(1, k):
        prev = np.zeros_like(p)
        prev[:, j:, :] = p[:, :-j, :]
        o *= 1.0 - prev
    return o


def _build_detection_y(
    data: np.ndarray,
    label_cfg: Dict,
    *,
    cond: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    detection_tau: float,
    detection_label: DetectionLabel,
    detection_min_normal_run: int,
) -> np.ndarray:
    det_cfg = {**label_cfg, "tau": float(detection_tau)}
    point = _anomaly_labels(data, cond=cond, baseline=baseline, **det_cfg)
    y = _detection_labels(point, detection_label=detection_label)
    if detection_label == "onset" and int(detection_min_normal_run) > 1:
        y = _filter_onset_cold_start(y, point, int(detection_min_normal_run))
    return y


def _resolve_detection_settings(
    *,
    tau: float,
    lookback: int,
    detection_label: DetectionLabel,
    detection_difficulty: DetectionDifficulty,
    detection_tau: Optional[float],
    detection_lookback: Optional[int],
    detection_min_normal_run: Optional[int],
) -> _DetectionSettings:
    if detection_difficulty == "standard":
        det_tau = float(tau)
        det_lb = int(lookback)
        det_run = 1
    elif detection_difficulty == "hard":
        det_tau = max(float(tau) * 1.5, 0.18)
        det_lb = max(int(lookback) // 2, 24)
        det_run = 3
    elif detection_difficulty == "severe":
        det_tau = max(float(tau) * 2.0, 0.22)
        det_lb = max(int(lookback) // 4, 12)
        det_run = 6
    else:
        raise ValueError(
            f"detection_difficulty must be 'standard', 'hard', or 'severe', "
            f"got {detection_difficulty!r}"
        )

    if detection_tau is not None:
        det_tau = float(detection_tau)
    if detection_lookback is not None:
        det_lb = int(detection_lookback)
    if detection_min_normal_run is not None:
        det_run = int(detection_min_normal_run)

    if det_lb < 1:
        raise ValueError("detection_lookback must be >= 1")
    if det_run < 1:
        raise ValueError("detection_min_normal_run must be >= 1")

    return _DetectionSettings(
        detection_tau=det_tau,
        detection_lookback=det_lb,
        detection_min_normal_run=det_run,
        detection_label=detection_label,
        difficulty=detection_difficulty,
    )


def _onset_labels(point_labels: np.ndarray) -> np.ndarray:
    """First timestep of each contiguous anomaly run in ``point_labels`` (N, T, 1)."""
    y = (np.asarray(point_labels, dtype=np.float32) > 0.5).astype(np.float32)
    prev = np.zeros_like(y)
    if y.shape[1] > 1:
        prev[:, 1:, :] = y[:, :-1, :]
    return (y * (1.0 - prev)).astype(np.float32)


def _detection_labels(
    point_labels: np.ndarray,
    *,
    detection_label: DetectionLabel,
) -> np.ndarray:
    if detection_label == "point":
        return np.asarray(point_labels, dtype=np.float32)
    if detection_label == "onset":
        return _onset_labels(point_labels)
    raise ValueError(f"detection_label must be 'point' or 'onset', got {detection_label!r}")


def _collect_windows(
    data: np.ndarray,
    labels: np.ndarray,
    times: List[int],
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows: ctx = x[t-lookback:t], y = label at t (+ horizon)."""
    lb = int(lookback)
    h = int(horizon)
    if lb < 1:
        raise ValueError("lookback must be >= 1")
    if h < 0:
        raise ValueError("horizon must be >= 0")

    _, _, dim = data.shape
    ctx_list: list[np.ndarray] = []
    y_list: list[float] = []

    for n in range(data.shape[0]):
        L = int(times[n])
        t_end = L - h if h > 0 else L
        for t in range(lb, t_end):
            ctx_list.append(data[n, t - lb : t, :].astype(np.float32, copy=False))
            y_list.append(float(labels[n, t + h, 0]))

    if not ctx_list:
        return np.zeros((0, lb, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(ctx_list, axis=0), np.asarray(y_list, dtype=np.float32)


def _subsample_windows(
    ctx: np.ndarray,
    y: np.ndarray,
    max_windows: Optional[int],
    seed: int,
    *,
    stratified: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample windows; stratified keeps label prevalence for eval PR-AUC."""
    n = int(ctx.shape[0])
    if max_windows is None or n <= int(max_windows):
        return ctx, y

    m = int(max_windows)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64).ravel()

    if not stratified:
        idx = rng.choice(n, size=m, replace=False)
        return ctx[idx], y[idx]

    pos_idx = np.flatnonzero(y > 0.5)
    neg_idx = np.flatnonzero(y <= 0.5)
    if pos_idx.size == 0 or neg_idx.size == 0:
        idx = rng.choice(n, size=m, replace=False)
        return ctx[idx], y[idx]

    pos_rate = pos_idx.size / n
    n_pos = int(round(m * pos_rate))
    n_pos = max(1, min(n_pos, pos_idx.size, m - 1))
    n_neg = m - n_pos
    n_neg = max(1, min(n_neg, neg_idx.size))

    idx = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(idx)
    return ctx[idx], y[idx]


def _build_ori_eval_cache(
    ori_data: np.ndarray,
    ori_time: List[int],
    y_det: np.ndarray,
    y_pred: np.ndarray,
    *,
    det_lookback: int,
    pred_lookback: int,
    horizon: int,
    max_eval_windows: Optional[int],
    seed: int,
) -> _OriEvalCache:
    det_ctx, det_y = _collect_windows(ori_data, y_det, ori_time, det_lookback, 0)
    pred_ctx, pred_y = _collect_windows(
        ori_data, y_pred, ori_time, pred_lookback, horizon
    )
    det_ctx, det_y = _subsample_windows(
        det_ctx, det_y, max_eval_windows, seed, stratified=True
    )
    pred_ctx, pred_y = _subsample_windows(
        pred_ctx, pred_y, max_eval_windows, seed + 1, stratified=True
    )
    return _OriEvalCache(
        det=_WindowPair(det_ctx, det_y),
        pred=_WindowPair(pred_ctx, pred_y),
    )


def _pr_skill_score(
    pr: float,
    *,
    baseline_pr: float,
    oracle_pr: float,
) -> float:
    """Normalized PR lift: (PR - baseline) / (oracle_PR - baseline); oracle -> 1."""
    denom = float(oracle_pr) - float(baseline_pr)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    if not np.isfinite(pr):
        return float("nan")
    return float((float(pr) - float(baseline_pr)) / denom)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return float("nan")
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _train_and_eval_pr(
    train_ctx: np.ndarray,
    train_y: np.ndarray,
    eval_ctx: np.ndarray,
    eval_y: np.ndarray,
    *,
    iterations: int,
    batch_size: int,
    eval_batch_size: int,
    hidden_dim: int,
    transformer_layers: int,
    transformer_dropout: float,
    lr: float,
    seed: int,
    device: torch.device,
    desc: str,
) -> float:
    if train_ctx.shape[0] == 0 or eval_ctx.shape[0] == 0:
        return float("nan")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    _, _, dim = train_ctx.shape
    model = _TransformerBinaryClassifier(
        train_ctx.shape[1],
        dim,
        hidden_dim,
        num_layers=transformer_layers,
        dropout=transformer_dropout,
    ).to(device)

    n_pos = float(train_y.sum())
    n_neg = float(train_y.size - n_pos)
    pos_weight = torch.tensor(1.0 if n_pos <= 0 or n_neg <= 0 else n_neg / n_pos, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_cpu = torch.from_numpy(train_ctx)
    y_cpu = torch.from_numpy(train_y)
    m = int(train_ctx.shape[0])

    model.train()
    for _ in range(int(iterations)):
        idx = np.random.choice(m, size=min(int(batch_size), m), replace=False)
        xb = x_cpu[idx].to(device)
        yb = y_cpu[idx].to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    scores: list[np.ndarray] = []
    xe_cpu = torch.from_numpy(eval_ctx)
    ne = int(eval_ctx.shape[0])
    ebs = int(eval_batch_size)
    with torch.no_grad():
        for start in range(0, ne, ebs):
            end = min(start + ebs, ne)
            logits = model(xe_cpu[start:end].to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())

    return _pr_auc(eval_y, np.concatenate(scores, axis=0))


def _label_kwargs(
    tau: float,
    num_days: int,
    slots_per_day: int,
    residual_eps: float,
    min_scale: float,
    use_mad_consistency: bool,
    zone_stats: Optional[Dict] = None,
) -> Dict:
    return dict(
        tau=tau,
        num_days=num_days,
        slots_per_day=slots_per_day,
        residual_eps=residual_eps,
        min_scale=min_scale,
        use_mad_consistency=use_mad_consistency,
        zone_stats=zone_stats,
    )


def _run_task_pr(
    train_data: np.ndarray,
    train_y: np.ndarray,
    train_time: List[int],
    eval_pair: _WindowPair,
    *,
    lookback: int,
    horizon: int,
    max_train_windows: Optional[int],
    seed: int,
    iterations: int,
    batch_size: int,
    eval_batch_size: Optional[int],
    hidden_dim: Optional[int],
    transformer_layers: int,
    transformer_dropout: float,
    lr: float,
    device: Optional[str],
    desc: str,
) -> float:
    train_ctx, train_labels = _collect_windows(
        train_data, train_y, train_time, lookback, horizon
    )
    train_ctx, train_labels = _subsample_windows(
        train_ctx, train_labels, max_train_windows, seed, stratified=False
    )

    dim = train_data.shape[-1]
    if hidden_dim is None:
        hidden_dim = max(32, 4 * dim)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ebs = int(eval_batch_size) if eval_batch_size is not None else int(batch_size)

    return _train_and_eval_pr(
        train_ctx,
        train_labels,
        eval_pair.ctx,
        eval_pair.y,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=ebs,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        seed=seed,
        device=dev,
        desc=desc,
    )


def _run_fidelity_with_cache(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    eval_cache: _OriEvalCache,
    *,
    lookback: int,
    det_lookback: int,
    horizon: int,
    train_source: TrainSource,
    label_cfg: Dict,
    det_settings: _DetectionSettings,
    ori_cond: Optional[np.ndarray] = None,
    gen_cond: Optional[np.ndarray] = None,
    ori_baseline: Optional[np.ndarray] = None,
    gen_baseline: Optional[np.ndarray] = None,
    max_train_windows: Optional[int],
    iterations: int,
    batch_size: int,
    eval_batch_size: Optional[int],
    hidden_dim: Optional[int],
    transformer_layers: int,
    transformer_dropout: float,
    lr: float,
    seed: int,
    device: Optional[str],
) -> Dict[str, float]:
    ori_time, _ = extract_time(ori_data)
    gen_time, _ = extract_time(generated_data)

    y_ori_point = _anomaly_labels(
        ori_data, cond=ori_cond, baseline=ori_baseline, **label_cfg
    )
    y_gen_point = _anomaly_labels(
        generated_data, cond=gen_cond, baseline=gen_baseline, **label_cfg
    )
    y_ori_det = _build_detection_y(
        ori_data,
        label_cfg,
        cond=ori_cond,
        baseline=ori_baseline,
        detection_tau=det_settings.detection_tau,
        detection_label=det_settings.detection_label,
        detection_min_normal_run=det_settings.detection_min_normal_run,
    )
    y_gen_det = _build_detection_y(
        generated_data,
        label_cfg,
        cond=gen_cond,
        baseline=gen_baseline,
        detection_tau=det_settings.detection_tau,
        detection_label=det_settings.detection_label,
        detection_min_normal_run=det_settings.detection_min_normal_run,
    )

    train_data = generated_data if train_source == "gen" else ori_data
    train_y_det = y_gen_det if train_source == "gen" else y_ori_det
    train_y_pred = y_gen_point if train_source == "gen" else y_ori_point
    train_time = gen_time if train_source == "gen" else ori_time

    det_seed = seed
    pred_seed = seed + 17 if seed is not None else None
    det_desc = (
        f"anomaly_onset_detection_{det_settings.difficulty}"
        if det_settings.detection_label == "onset"
        else "anomaly_detection"
    )

    det_pr = _run_task_pr(
        train_data,
        train_y_det,
        train_time,
        eval_cache.det,
        lookback=det_lookback,
        horizon=0,
        max_train_windows=max_train_windows,
        seed=det_seed,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        device=device,
        desc=det_desc,
    )
    pred_pr = _run_task_pr(
        train_data,
        train_y_pred,
        train_time,
        eval_cache.pred,
        lookback=lookback,
        horizon=horizon,
        max_train_windows=max_train_windows,
        seed=pred_seed,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        device=device,
        desc=f"anomaly_prediction_h{int(horizon)}",
    )
    return {"Det-PR": det_pr, "Pred-PR": pred_pr}


def _prepare_arrays(ori_data, generated_data) -> Tuple[np.ndarray, np.ndarray]:
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)
    if ori_data.ndim == 2:
        ori_data = ori_data[..., np.newaxis]
    if generated_data.ndim == 2:
        generated_data = generated_data[..., np.newaxis]
    if ori_data.shape[-1] != generated_data.shape[-1]:
        raise ValueError("ori_data and generated_data must share channel dimension")
    return ori_data, generated_data


def _run_tstr_pr(
    ori_data,
    generated_data,
    *,
    lookback: int,
    horizon: int,
    tau: float,
    num_days: int,
    slots_per_day: int,
    residual_eps: float,
    min_scale: float,
    use_mad_consistency: bool,
    iterations: int,
    batch_size: int,
    eval_batch_size: Optional[int],
    hidden_dim: Optional[int],
    transformer_layers: int,
    transformer_dropout: float,
    lr: float,
    seed: int,
    device: Optional[str],
    train_source: TrainSource,
    max_train_windows: Optional[int],
    max_eval_windows: Optional[int],
    desc: str,
) -> float:
    ori_data, generated_data = _prepare_arrays(ori_data, generated_data)
    ori_time, _ = extract_time(ori_data)
    label_cfg = _label_kwargs(
        tau, num_days, slots_per_day, residual_eps, min_scale, use_mad_consistency
    )
    y_ori = _anomaly_labels(ori_data, **label_cfg)

    eval_ctx, eval_y = _collect_windows(ori_data, y_ori, ori_time, lookback, horizon)
    eval_ctx, eval_y = _subsample_windows(
        eval_ctx, eval_y, max_eval_windows, seed + 2, stratified=True
    )
    eval_pair = _WindowPair(eval_ctx, eval_y)

    gen_time, _ = extract_time(generated_data)
    y_gen = _anomaly_labels(generated_data, **label_cfg)
    train_data = generated_data if train_source == "gen" else ori_data
    train_y = y_gen if train_source == "gen" else y_ori
    train_time = gen_time if train_source == "gen" else ori_time

    return _run_task_pr(
        train_data,
        train_y,
        train_time,
        eval_pair,
        lookback=lookback,
        horizon=horizon,
        max_train_windows=max_train_windows,
        seed=seed,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        device=device,
        desc=desc,
    )


def _speed_kwargs(
    lookback: int,
    tau: float,
    num_days: int,
    slots_per_day: int,
    residual_eps: float,
    min_scale: float,
    use_mad_consistency: bool,
    iterations: int,
    batch_size: int,
    eval_batch_size: Optional[int],
    hidden_dim: Optional[int],
    transformer_layers: int,
    transformer_dropout: float,
    lr: float,
    seed: int,
    device: Optional[str],
    train_source: TrainSource,
    max_train_windows: Optional[int],
    max_eval_windows: Optional[int],
    horizon: int = 1,
) -> Dict:
    return dict(
        lookback=lookback,
        tau=tau,
        num_days=num_days,
        slots_per_day=slots_per_day,
        residual_eps=residual_eps,
        min_scale=min_scale,
        use_mad_consistency=use_mad_consistency,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        seed=seed,
        device=device,
        train_source=train_source,
        max_train_windows=max_train_windows,
        max_eval_windows=max_eval_windows,
        horizon=horizon,
    )


def anomaly_detection_score_metrics(
    ori_data,
    generated_data,
    *,
    lookback: int = 48,
    tau: float = 0.12,
    num_days: int = 30,
    slots_per_day: int = 6,
    residual_eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    iterations: int = DEFAULT_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[str] = None,
    train_source: TrainSource = "gen",
    max_train_windows: Optional[int] = DEFAULT_MAX_TRAIN_WINDOWS,
    max_eval_windows: Optional[int] = DEFAULT_MAX_EVAL_WINDOWS,
) -> float:
    """PR-AUC for contemporaneous anomaly detection (train on sim, test on real by default)."""
    kw = _speed_kwargs(
        lookback, tau, num_days, slots_per_day, residual_eps, min_scale,
        use_mad_consistency, iterations, batch_size, eval_batch_size,
        hidden_dim, transformer_layers, transformer_dropout, lr, seed, device,
        train_source, max_train_windows, max_eval_windows,
    )
    return _run_tstr_pr(ori_data, generated_data, **kw, horizon=0, desc="anomaly_detection")


def anomaly_prediction_score_metrics(
    ori_data,
    generated_data,
    *,
    lookback: int = 48,
    horizon: int = 1,
    tau: float = 0.12,
    num_days: int = 30,
    slots_per_day: int = 6,
    residual_eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    iterations: int = DEFAULT_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[str] = None,
    train_source: TrainSource = "gen",
    max_train_windows: Optional[int] = DEFAULT_MAX_TRAIN_WINDOWS,
    max_eval_windows: Optional[int] = DEFAULT_MAX_EVAL_WINDOWS,
) -> float:
    """PR-AUC for future anomaly prediction (train on sim, test on real by default)."""
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1 for anomaly prediction")
    kw = _speed_kwargs(
        lookback, tau, num_days, slots_per_day, residual_eps, min_scale,
        use_mad_consistency, iterations, batch_size, eval_batch_size,
        hidden_dim, transformer_layers, transformer_dropout, lr, seed, device,
        train_source, max_train_windows, max_eval_windows, horizon=int(horizon),
    )
    return _run_tstr_pr(
        ori_data,
        generated_data,
        **kw,
        desc=f"anomaly_prediction_h{int(horizon)}",
    )


def anomaly_downstream_fidelity_metrics(
    ori_data,
    generated_data,
    *,
    lookback: int = 48,
    horizon: int = 1,
    tau: float = 0.12,
    num_days: int = 30,
    slots_per_day: int = 6,
    residual_eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    iterations: int = DEFAULT_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[str] = None,
    train_source: TrainSource = "gen",
    max_train_windows: Optional[int] = DEFAULT_MAX_TRAIN_WINDOWS,
    max_eval_windows: Optional[int] = DEFAULT_MAX_EVAL_WINDOWS,
) -> Dict[str, float]:
    """Return ``{\"Det-PR\": ..., \"Pred-PR\": ...}`` (PR-AUC only)."""
    common = dict(
        lookback=lookback,
        horizon=horizon,
        tau=tau,
        num_days=num_days,
        slots_per_day=slots_per_day,
        residual_eps=residual_eps,
        min_scale=min_scale,
        use_mad_consistency=use_mad_consistency,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        seed=seed,
        device=device,
        train_source=train_source,
        max_train_windows=max_train_windows,
        max_eval_windows=max_eval_windows,
    )
    return {
        "Det-PR": anomaly_detection_score_metrics(ori_data, generated_data, **common),
        "Pred-PR": anomaly_prediction_score_metrics(ori_data, generated_data, **common),
    }


def anomaly_downstream_fidelity_gap_metrics(
    ori_data,
    generated_data,
    **kwargs,
) -> Dict[str, float]:
    """Return gap to oracle: ``oracle_PR - TSTR_PR`` (smaller is better)."""
    oracle = anomaly_downstream_fidelity_metrics(ori_data, ori_data, train_source="ori", **kwargs)
    tstr = anomaly_downstream_fidelity_metrics(ori_data, generated_data, train_source="gen", **kwargs)
    return {
        "Det-gap": oracle["Det-PR"] - tstr["Det-PR"],
        "Pred-gap": oracle["Pred-PR"] - tstr["Pred-PR"],
    }


def downstream_summary_table(
    ori_data,
    baselines: Mapping[str, np.ndarray],
    *,
    baseline_names: Optional[Sequence[str]] = None,
    decimals: int = 4,
    lookback: int = 48,
    horizon: int = 1,
    tau: float = 0.12,
    num_days: int = 30,
    slots_per_day: int = 6,
    residual_eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    iterations: int = DEFAULT_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[str] = None,
    max_train_windows: Optional[int] = DEFAULT_MAX_TRAIN_WINDOWS,
    max_eval_windows: Optional[int] = DEFAULT_MAX_EVAL_WINDOWS,
    detection_label: DetectionLabel = "onset",
    detection_difficulty: DetectionDifficulty = "standard",
    detection_tau: Optional[float] = None,
    detection_lookback: Optional[int] = None,
    detection_min_normal_run: Optional[int] = None,
    cond: Optional[np.ndarray] = None,
    baseline_conds: Optional[Mapping[str, np.ndarray]] = None,
    cond_group_num_days: Optional[int] = None,
    ori_baseline: Optional[np.ndarray] = None,
) -> "pd.DataFrame":
    """Per-group table: oracle + gen baselines; columns = Det-PRAUC, Pred-PRAUC.

    By default detection uses **onset** labels (first point of each anomaly run).
    Both columns are PR-AUC (higher is better). ``oracle`` is train on ori / test on ori;
    gen rows are TSTR (train on sim). Set ``detection_label=\"point\"`` for point-level detection.

    ``detection_difficulty`` presets (onset only):
      - ``standard``: τ, lookback=48, cold-start=1
      - ``hard``: τ≥0.18, lookback=24, 3 normal steps before onset
      - ``severe``: τ≥0.22, lookback=12, 6 normal steps before onset

    Override any preset with ``detection_tau`` / ``detection_lookback`` /
    ``detection_min_normal_run``. Prediction always uses ``tau`` and ``lookback``.
    """
    import pandas as pd

    names = list(baseline_names or baselines.keys())
    if not names:
        raise ValueError("baselines must contain at least one entry")

    ori_data, _ = _prepare_arrays(ori_data, next(iter(baselines.values())))
    ori_time, _ = extract_time(ori_data)

    zone_stats = None
    if cond_group_num_days is not None:
        if cond is None:
            raise ValueError("cond is required when cond_group_num_days is set")
        zone_stats = fit_cond_group_zone_stats(
            ori_data,
            cond,
            num_days=int(cond_group_num_days),
            slots_per_day=int(slots_per_day),
            eps=residual_eps,
            min_scale=min_scale,
            use_mad_consistency=use_mad_consistency,
        )

    label_cfg = _label_kwargs(
        tau,
        num_days,
        slots_per_day,
        residual_eps,
        min_scale,
        use_mad_consistency,
        zone_stats=zone_stats,
    )
    det_settings = _resolve_detection_settings(
        tau=tau,
        lookback=lookback,
        detection_label=detection_label,
        detection_difficulty=detection_difficulty,
        detection_tau=detection_tau,
        detection_lookback=detection_lookback,
        detection_min_normal_run=detection_min_normal_run,
    )
    y_ori_point = _anomaly_labels(
        ori_data, cond=cond, baseline=ori_baseline, **label_cfg
    )
    y_ori_det = _build_detection_y(
        ori_data,
        label_cfg,
        cond=cond,
        baseline=ori_baseline,
        detection_tau=det_settings.detection_tau,
        detection_label=det_settings.detection_label,
        detection_min_normal_run=det_settings.detection_min_normal_run,
    )
    eval_cache = _build_ori_eval_cache(
        ori_data,
        ori_time,
        y_ori_det,
        y_ori_point,
        det_lookback=det_settings.detection_lookback,
        pred_lookback=lookback,
        horizon=horizon,
        max_eval_windows=max_eval_windows,
        seed=seed,
    )

    run_kw = dict(
        lookback=lookback,
        det_lookback=det_settings.detection_lookback,
        horizon=horizon,
        label_cfg=label_cfg,
        det_settings=det_settings,
        max_train_windows=max_train_windows,
        iterations=iterations,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_dropout=transformer_dropout,
        lr=lr,
        seed=seed,
        device=device,
    )

    oracle_raw = _run_fidelity_with_cache(
        ori_data,
        ori_data,
        eval_cache,
        train_source="ori",
        ori_cond=cond,
        gen_cond=cond,
        ori_baseline=ori_baseline,
        gen_baseline=ori_baseline,
        **run_kw
    )

    rows: list[Dict[str, Union[str, float]]] = [
        {
            "baseline": "oracle",
            "Det-PRAUC": oracle_raw["Det-PR"],
            "Pred-PRAUC": oracle_raw["Pred-PR"],
        },
    ]
    for i, name in enumerate(names):
        gen = np.asarray(baselines[name], dtype=np.float32)
        if gen.ndim == 2:
            gen = gen[..., np.newaxis]
        gen_cond = None
        if zone_stats is not None:
            if baseline_conds is not None and name in baseline_conds:
                gen_cond = baseline_conds[name]
            elif cond is not None and gen.shape[0] == cond.shape[0]:
                gen_cond = cond
            else:
                raise ValueError(
                    f"baseline_conds['{name}'] required for cond-grouped labels "
                    f"(gen rows={gen.shape[0]})"
                )
        tstr_raw = _run_fidelity_with_cache(
            ori_data,
            gen,
            eval_cache,
            train_source="gen",
            ori_cond=cond,
            gen_cond=gen_cond,
            ori_baseline=ori_baseline,
            seed=seed + 100 * (i + 1),
            **{k: v for k, v in run_kw.items() if k != "seed"},
        )
        rows.append(
            {
                "baseline": name,
                "Det-PRAUC": tstr_raw["Det-PR"],
                "Pred-PRAUC": tstr_raw["Pred-PR"],
            }
        )

    return pd.DataFrame(rows).set_index("baseline").round(decimals)
