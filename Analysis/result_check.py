"""
Stage 1 result check — read saved pack + checkpoint, print soft/hard evaluation metrics.

Soft track:  weight-weighted reconstruction, gate calibration (p_hat vs weight)
Hard track:  mask_hard occurrence detection, tau_pred sweep, sparse hit

Run:
    cd Analysis
    python result_check.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import (
        average_precision_score,
        mean_absolute_error,
        precision_recall_fscore_support,
        r2_score,
        roc_auc_score,
    )
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from zero_aware_residual_learner import (
    CONFIG,
    ResidualDataset,
    SCRIPT_DIR,
    ZeroAwareResidualAutoencoder,
    load_latent_encoder,
    occurrence_mask_from_data_r,
    prepare_filtered_residual,
)

CHECK_CONFIG: Dict = {
    "result_dir": str(SCRIPT_DIR / "Result"),
    "output_prefix": CONFIG["output_prefix"],
    "checkpoint_name": CONFIG["checkpoint_name"],
    "batch_size": CONFIG.get("extract_batch_size", 256),
    "occurrence_threshold": CONFIG.get("occurrence_threshold", 0.12),
    "shuffle_z_seed": 42,
    "tau_list": None,
    "tau_beta_f2": 2.0,
    "use_soft_amplitude": True,
}


def merge_check_cfg(cfg: Optional[Dict] = None) -> Dict:
    """Merge training CONFIG with check overrides (filter keys live in CONFIG)."""
    merged = dict(CONFIG)
    merged.update(cfg or CHECK_CONFIG)
    return merged


def load_anomaly_pack(
    result_dir: Union[str, Path],
    prefix: str,
    cfg: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """Load npy pack; compute mask_hard from data if missing."""
    cfg = merge_check_cfg(cfg)
    result_dir = Path(result_dir)

    pack = {
        "z_a": np.load(result_dir / f"{prefix}_latent.npy"),
        "data": np.load(result_dir / f"{prefix}.npy"),
        "cond": np.load(result_dir / f"{prefix}_condition.npy"),
        "data_f": np.load(result_dir / f"{prefix}_residual.npy"),
        "weight": np.load(result_dir / f"{prefix}_mask.npy"),
    }
    for key, arr in pack.items():
        pack[key] = np.asarray(arr, dtype=np.float32)

    mask_hard_path = result_dir / f"{prefix}_mask_hard.npy"
    if mask_hard_path.is_file():
        pack["mask_hard"] = np.load(mask_hard_path).astype(np.float32)
    else:
        _, _, mask_hard, _, _ = prepare_filtered_residual(pack["data"], cfg)
        pack["mask_hard"] = mask_hard

    pack["mask"] = pack["weight"]
    pack["cond"] = pack["cond"].reshape(-1, 1, 1).astype(np.float32)
    return pack


@torch.no_grad()
def collect_model_outputs(
    model: ZeroAwareResidualAutoencoder,
    data_f: np.ndarray,
    weight: np.ndarray,
    cond: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    dataset = ResidualDataset(data_f=data_f, mask=weight, cond=cond)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    chunks: Dict[str, List[np.ndarray]] = {
        "target": [],
        "weight": [],
        "a_hat": [],
        "p_logit": [],
        "p_hat": [],
        "v_hat": [],
        "z": [],
    }

    for batch in loader:
        residual = batch["residual"].to(device)
        weight_b = batch["mask"].to(device)
        cond_b = batch["cond"].to(device)
        out = model(residual, weight_b, cond_b)

        chunks["target"].append(residual.cpu().numpy())
        chunks["weight"].append(weight_b.cpu().numpy())
        chunks["a_hat"].append(out["a_hat"].cpu().numpy())
        chunks["p_logit"].append(out["p_logit"].cpu().numpy())
        chunks["p_hat"].append(out["p_hat"].cpu().numpy())
        chunks["v_hat"].append(out["v_hat"].cpu().numpy())
        chunks["z"].append(out["z_a"].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


@torch.no_grad()
def decoder_forward_with_given_z(
    model: ZeroAwareResidualAutoencoder,
    z: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    z_t = torch.as_tensor(z, dtype=torch.float32)
    if z_t.ndim == 3 and z_t.shape[-1] == 1:
        z_t = z_t.squeeze(-1)
    loader = DataLoader(TensorDataset(z_t), batch_size=batch_size, shuffle=False)

    chunks: Dict[str, List[np.ndarray]] = {
        "a_hat": [],
        "p_logit": [],
        "p_hat": [],
        "v_hat": [],
    }
    for (z_b,) in loader:
        z_b = z_b.to(device)
        a_hat_b, p_logit_b, p_hat_b, v_hat_b = model.decode(z_b)
        chunks["a_hat"].append(a_hat_b.cpu().numpy())
        chunks["p_logit"].append(p_logit_b.cpu().numpy())
        chunks["p_hat"].append(p_hat_b.cpu().numpy())
        chunks["v_hat"].append(v_hat_b.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def evaluate_weighted_reconstruction(
    target: np.ndarray,
    a_hat: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Soft reconstruction metrics; weight is continuous gate, not 0/1."""
    target = np.asarray(target, dtype=np.float64)
    a_hat = np.asarray(a_hat, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    diff = a_hat - target
    sq_err = diff ** 2
    abs_err = np.abs(diff)

    mse_all = np.mean(sq_err)
    mse_weighted = np.sum(weight * sq_err) / (np.sum(weight) + eps)
    mse_low_weight = np.sum((1.0 - weight) * sq_err) / (np.sum(1.0 - weight) + eps)
    mae_weighted = np.sum(weight * abs_err) / (np.sum(weight) + eps)

    target_sign = np.sign(target)
    pred_sign = np.sign(a_hat)
    sign_acc = np.sum(weight * (target_sign == pred_sign)) / (np.sum(weight) + eps)

    return {
        "mse_all": float(mse_all),
        "mse_weighted": float(mse_weighted),
        "mse_low_weight": float(mse_low_weight),
        "mae_weighted": float(mae_weighted),
        "sign_acc": float(sign_acc),
    }


def evaluate_gate_calibration(
    weight: np.ndarray,
    p_hat: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """How well decoder gate p_hat matches soft filter weight."""
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    p = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    gate_mse = float(np.mean((p - w) ** 2))
    gate_mae = float(np.mean(np.abs(p - w)))
    if w.std() < eps or p.std() < eps:
        gate_corr = float("nan")
    else:
        gate_corr = float(np.corrcoef(w, p)[0, 1])
    return {
        "gate_mse": gate_mse,
        "gate_mae": gate_mae,
        "gate_corr": gate_corr,
    }


def evaluate_occurrence_at_tau(
    mask_hard: np.ndarray,
    p_hat: np.ndarray,
    tau_pred: float,
) -> Dict[str, float]:
    """Binary occurrence metrics: p_hat >= tau_pred vs mask_hard."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required: pip install scikit-learn")

    y_true = np.asarray(mask_hard, dtype=np.float64).reshape(-1).astype(int)
    y_score = np.asarray(p_hat, dtype=np.float64).reshape(-1)
    y_pred = (y_score >= tau_pred).astype(int)

    auprc = float(average_precision_score(y_true, y_score))
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "tau_pred": float(tau_pred),
        "auprc": auprc,
        "auroc": auroc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def print_metrics(metrics: Dict[str, float], title: str) -> None:
    print(f"=== {title} ===")
    for k, v in metrics.items():
        if isinstance(v, float) and np.isnan(v):
            print(f"{k}: nan")
        else:
            print(f"{k}: {v:.6f}")


def _squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] != 1:
            raise ValueError(f"expected last dim 1 for 3D array, got {arr.shape}")
        return arr[..., 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected (N, T) or (N, T, 1), got {arr.shape}")


def sweep_best_tau_pred(
    p_hat: np.ndarray,
    mask_hard: np.ndarray,
    tau_list: Optional[np.ndarray] = None,
    beta: float = 1.0,
    select_by: str = "f1",
    eps: float = 1e-8,
    verbose: bool = True,
) -> Tuple[float, Dict, List[Dict]]:
    """Sweep tau_pred on p_hat; GT is mask_hard (binary occurrence)."""
    if select_by not in ("f1", "fbeta"):
        raise ValueError(f"select_by must be 'f1' or 'fbeta', got {select_by!r}")

    p_2d = _squeeze_to_2d(p_hat).astype(np.float64)
    m_2d = _squeeze_to_2d(mask_hard).astype(bool)

    if tau_list is None:
        tau_list = np.linspace(0.05, 0.95, 50)
    tau_list = np.asarray(tau_list, dtype=np.float64)

    num_true = int(m_2d.sum())
    rows: List[Dict] = []

    for tau in tau_list:
        pred = p_2d >= tau
        tp = int(np.logical_and(pred, m_2d).sum())
        fp = int(np.logical_and(pred, ~m_2d).sum())
        fn = int(np.logical_and(~pred, m_2d).sum())
        num_pred = int(pred.sum())

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        fbeta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)

        rows.append({
            "tau_pred": float(tau),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fbeta": float(fbeta),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_pred": num_pred,
            "num_true": num_true,
        })

    score_key = "f1" if select_by == "f1" else "fbeta"
    best_idx = int(np.argmax([r[score_key] for r in rows]))
    best_metrics = dict(rows[best_idx])
    tau_best = float(best_metrics["tau_pred"])

    if verbose:
        label = f"F{beta:g}" if select_by == "fbeta" else "F1"
        print(f"=== tau_pred sweep on p_hat (GT=mask_hard, select by max {label}) ===")
        print(f"num_true anomalies: {num_true}")
        print(f"Best tau_pred: {tau_best:.4f}")
        print(
            f"  precision={best_metrics['precision']:.4f} "
            f"recall={best_metrics['recall']:.4f} "
            f"f1={best_metrics['f1']:.4f} "
            f"fbeta={best_metrics['fbeta']:.4f} "
            f"num_pred={int(best_metrics['num_pred'])}"
        )
        print(f"\nTop tau_pred by {label}:")
        print(f"{'tau_pred':>8s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'fbeta':>8s} {'num_pred':>10s}")
        for r in sorted(rows, key=lambda x: x[score_key], reverse=True)[:5]:
            print(
                f"{r['tau_pred']:8.3f} {r['precision']:10.4f} {r['recall']:8.4f} "
                f"{r['f1']:8.4f} {r['fbeta']:8.4f} {int(r['num_pred']):10d}"
            )

    return tau_best, best_metrics, rows


def make_sparse_reconstruction(
    p_hat: np.ndarray,
    v_hat: np.ndarray,
    tau_pred: float,
    use_soft_amplitude: bool = True,
) -> np.ndarray:
    p_hat = np.asarray(p_hat, dtype=np.float32)
    v_hat = np.asarray(v_hat, dtype=np.float32)
    hard_gate = (p_hat >= tau_pred).astype(np.float32)
    if use_soft_amplitude:
        return (hard_gate * (p_hat * v_hat)).astype(np.float32)
    return (hard_gate * v_hat).astype(np.float32)


def compute_sparse_hit_metrics(
    target: np.ndarray,
    p_hat: np.ndarray,
    v_hat: np.ndarray,
    mask_hard: np.ndarray,
    tau_pred: float,
    use_soft_amplitude: bool = True,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Sparse hit metrics using binary mask_hard as GT."""
    target = np.asarray(target, dtype=np.float64)
    p_hat = np.asarray(p_hat, dtype=np.float64)
    v_hat = np.asarray(v_hat, dtype=np.float64)

    a_hat_sparse = make_sparse_reconstruction(
        p_hat, v_hat, tau_pred, use_soft_amplitude=use_soft_amplitude
    ).astype(np.float64)

    hard_gate = p_hat >= tau_pred
    mask_bool = np.asarray(mask_hard, dtype=bool)

    hit = hard_gate & mask_bool
    miss = (~hard_gate) & mask_bool
    false_pos = hard_gate & (~mask_bool)

    num_true_anom = int(mask_bool.sum())
    num_pred_anom = int(hard_gate.sum())
    num_hit = int(hit.sum())
    num_miss = int(miss.sum())
    num_false_pos = int(false_pos.sum())

    recall = num_hit / (num_true_anom + eps)
    precision = num_hit / (num_pred_anom + eps)
    miss_rate = num_miss / (num_true_anom + eps)

    if num_hit > 0:
        mse_hit = float(np.mean((a_hat_sparse[hit] - target[hit]) ** 2))
        mae_hit = float(np.mean(np.abs(a_hat_sparse[hit] - target[hit])))
        sign_acc_hit = float(np.mean(np.sign(a_hat_sparse[hit]) == np.sign(target[hit])))
    else:
        mse_hit = float("nan")
        mae_hit = float("nan")
        sign_acc_hit = float("nan")

    mae_missed = float(np.mean(np.abs(target[miss]))) if num_miss > 0 else 0.0
    mse_zero = float(np.mean((a_hat_sparse[~mask_bool]) ** 2)) if (~mask_bool).any() else float("nan")
    mse_false_pos = (
        float(np.mean((a_hat_sparse[false_pos]) ** 2)) if num_false_pos > 0 else 0.0
    )

    return {
        "tau_pred": float(tau_pred),
        "num_true_anom": num_true_anom,
        "num_pred_anom": num_pred_anom,
        "num_hit": num_hit,
        "num_miss": num_miss,
        "num_false_pos": num_false_pos,
        "recall": float(recall),
        "precision": float(precision),
        "miss_rate": float(miss_rate),
        "mse_hit": mse_hit,
        "mae_hit": mae_hit,
        "sign_acc_hit": sign_acc_hit,
        "mae_missed": mae_missed,
        "mse_zero": mse_zero,
        "mse_false_pos": mse_false_pos,
    }


def print_sparse_hit_metrics(metrics: Dict, title: str = "Sparse hit metrics") -> None:
    print(f"=== {title} ===")
    keys = [
        "tau_pred", "recall", "precision", "miss_rate",
        "mse_zero", "mse_hit", "mae_hit", "sign_acc_hit",
        "mae_missed", "mse_false_pos",
        "num_hit", "num_miss", "num_false_pos", "num_pred_anom", "num_true_anom",
    ]
    for k in keys:
        v = metrics.get(k)
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            print(f"{k}: nan")
        elif isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


def run_shuffle_z_test(
    model: ZeroAwareResidualAutoencoder,
    z: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    mask_hard: np.ndarray,
    tau_pred: float,
    seed: int = 42,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Dict:
    rng = np.random.default_rng(seed)
    z = np.asarray(z)
    if z.ndim == 3 and z.shape[-1] == 1:
        z = z.squeeze(-1)
    perm = rng.permutation(z.shape[0])
    z_shuffle = z[perm]

    shuffle_out = decoder_forward_with_given_z(
        model=model,
        z=z_shuffle,
        batch_size=batch_size,
        device=device,
    )

    rec_metrics = evaluate_weighted_reconstruction(
        target=target,
        a_hat=shuffle_out["a_hat"],
        weight=weight,
    )
    occ_metrics = evaluate_occurrence_at_tau(
        mask_hard=mask_hard,
        p_hat=shuffle_out["p_hat"],
        tau_pred=tau_pred,
    )

    return {
        "reconstruction": rec_metrics,
        "occurrence": occ_metrics,
        "perm": perm,
    }


RESIDUAL_PROBE_KEYS = [
    "anomaly_count",
    "positive_count",
    "negative_count",
    "positive_ratio",
    "negative_ratio",
    "max_abs_residual",
    "mean_abs_anom_residual",
    "longest_duration",
]


def _to_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def longest_consecutive_ones(binary_1d: np.ndarray) -> int:
    arr = np.asarray(binary_1d).astype(int).ravel()
    if arr.size == 0:
        return 0
    max_len = cur = 0
    for v in arr:
        if v == 1:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len


def extract_anomaly_statistics(
    data_r: np.ndarray,
    mask_hard: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    data_r = np.asarray(data_r, dtype=np.float64)
    mask_hard = np.asarray(mask_hard, dtype=np.float64)
    n_samples = data_r.shape[0]

    anomaly_count = _to_1d(mask_hard.sum(axis=(1, 2)))
    positive_count = _to_1d(((data_r > 0) & (mask_hard > 0)).sum(axis=(1, 2)))
    negative_count = _to_1d(((data_r < 0) & (mask_hard > 0)).sum(axis=(1, 2)))
    positive_ratio = positive_count / (anomaly_count + eps)
    negative_ratio = negative_count / (anomaly_count + eps)
    max_abs_residual = _to_1d(np.max(np.abs(data_r), axis=(1, 2)))
    sum_abs_anom = _to_1d(np.sum(mask_hard * np.abs(data_r), axis=(1, 2)))
    mean_abs_anom_residual = sum_abs_anom / (anomaly_count + eps)

    longest_duration = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        longest_duration[i] = longest_consecutive_ones(mask_hard[i, :, 0])

    return {
        "anomaly_count": anomaly_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "max_abs_residual": max_abs_residual,
        "mean_abs_anom_residual": mean_abs_anom_residual,
        "longest_duration": longest_duration,
    }


def extract_load_behavior_statistics(data: np.ndarray, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    x = np.asarray(data, dtype=np.float64)[..., 0]
    peak_load = x.max(axis=1)
    mean_load = x.mean(axis=1)
    std_load = x.std(axis=1)
    median_load = np.median(x, axis=1)
    min_load = x.min(axis=1)
    peak_to_mean = peak_load / (mean_load + eps)
    peak_to_median = peak_load / (median_load + eps)
    diff = np.diff(x, axis=1)
    max_ramp_up = diff.max(axis=1)
    max_ramp_down = diff.min(axis=1)
    mean_abs_ramp = np.mean(np.abs(diff), axis=1)
    q90 = np.quantile(x, 0.90, axis=1, keepdims=True)
    q10 = np.quantile(x, 0.10, axis=1, keepdims=True)
    high_load_count = (x >= q90).sum(axis=1)
    low_load_count = (x <= q10).sum(axis=1)
    load_range = peak_load - min_load
    load_cv = std_load / (mean_load + eps)
    return {
        "x_peak_load": peak_load,
        "x_mean_load": mean_load,
        "x_std_load": std_load,
        "x_median_load": median_load,
        "x_min_load": min_load,
        "x_peak_to_mean": peak_to_mean,
        "x_peak_to_median": peak_to_median,
        "x_max_ramp_up": max_ramp_up,
        "x_max_ramp_down": max_ramp_down,
        "x_mean_abs_ramp": mean_abs_ramp,
        "x_high_load_count": high_load_count,
        "x_low_load_count": low_load_count,
        "x_load_range": load_range,
        "x_load_cv": load_cv,
    }


def latent_regression_probe(
    z: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for latent probing")
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z_train, z_test, y_train, y_test = train_test_split(
        z, y, test_size=test_size, random_state=random_state
    )
    model = Ridge(alpha=1.0)
    model.fit(z_train, y_train)
    pred = model.predict(z_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def _regression_dict(probe_results: Optional[Dict]) -> Dict:
    if probe_results is None:
        return {}
    if "regression" in probe_results:
        return probe_results["regression"]
    return probe_results


def average_probe_r2(probe_results: Optional[Dict], keys: Optional[List[str]] = None) -> float:
    reg = _regression_dict(probe_results)
    if keys is None:
        keys = list(reg.keys())
    values = []
    for key in keys:
        if key not in reg:
            continue
        r2 = reg[key].get("r2", np.nan)
        if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
            continue
        values.append(float(r2))
    return float(np.mean(values)) if values else float("nan")


def run_latent_probing(
    z: np.ndarray,
    stats: Dict[str, np.ndarray],
    regression_keys: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    if regression_keys is None:
        regression_keys = RESIDUAL_PROBE_KEYS
    regression_results = {}
    for name in regression_keys:
        if name not in stats:
            continue
        regression_results[name] = latent_regression_probe(z, stats[name])
    return {"regression": regression_results}


FINAL_SUMMARY_KEYS = [
    ("preprocess_threshold", "filter threshold"),
    ("weight_mean", "soft weight mean"),
    ("mask_hard_ratio", "mask_hard rate"),
    ("soft_sign_acc", "soft sign accuracy"),
    ("gate_corr", "gate corr (p_hat vs weight)"),
    ("best_tau_f1", "tau_pred @ F1"),
    ("best_f1", "occurrence F1"),
    ("mask_auprc", "occurrence AUPRC"),
    ("mask_auroc", "occurrence AUROC"),
    ("best_tau_f2", "tau_pred @ F2"),
    ("best_f2", "occurrence F2"),
    ("best_recall_f2", "occurrence recall @ F2"),
    ("sparse_f1_recall", "sparse recall @ F1 tau"),
    ("sparse_f1_precision", "sparse precision @ F1 tau"),
    ("sparse_f1_sign_acc_hit", "sparse sign acc @ hit"),
    ("shuffle_gap_mse_weighted", "shuffle-z gap (mse_weighted)"),
    ("shuffle_gap_auprc", "shuffle-z gap (AUPRC)"),
    ("probe_avg_r2", "probe R2 (anomaly stats)"),
    ("x_probe_avg_r2", "probe R2 (load behavior)"),
    ("z_vs_c_gain", "z vs cond gain"),
]


def build_result_check_summary(
    cfg: Dict,
    weight_eval: np.ndarray,
    mask_hard: np.ndarray,
    rec_metrics: Dict[str, float],
    gate_metrics: Dict[str, float],
    occ_f1: Dict[str, float],
    best_f1: Dict,
    best_f2: Dict,
    metrics_sparse_f1: Dict[str, float],
    shuffle_results: Dict,
    probe_results: Dict,
    x_probe_results: Dict,
    probe_c_results: Dict,
) -> Dict[str, float]:
    """Flatten key metrics into one row (aligned with ExperimentFirst.ipynb)."""
    weight_eval = np.asarray(weight_eval)
    mask_hard = np.asarray(mask_hard)
    shuffle_rec = shuffle_results["reconstruction"]
    shuffle_occ = shuffle_results["occurrence"]

    probe_z_avg_r2 = average_probe_r2(probe_results, keys=RESIDUAL_PROBE_KEYS)
    probe_c_avg_r2 = average_probe_r2(probe_c_results, keys=RESIDUAL_PROBE_KEYS)

    summary = {
        "preprocess_threshold": float(cfg.get("occurrence_threshold", cfg["filter_threshold"])),
        "weight_mean": float(weight_eval.mean()),
        "mask_hard_ratio": float(mask_hard.mean()),
        "soft_sign_acc": rec_metrics["sign_acc"],
        "gate_corr": gate_metrics["gate_corr"],
        "best_tau_f1": float(best_f1["tau_pred"]),
        "best_f1": float(best_f1["f1"]),
        "mask_auprc": occ_f1["auprc"],
        "mask_auroc": occ_f1["auroc"],
        "best_tau_f2": float(best_f2["tau_pred"]),
        "best_f2": float(best_f2["fbeta"]),
        "best_recall_f2": float(best_f2["recall"]),
        "sparse_f1_recall": metrics_sparse_f1["recall"],
        "sparse_f1_precision": metrics_sparse_f1["precision"],
        "sparse_f1_sign_acc_hit": metrics_sparse_f1["sign_acc_hit"],
        "shuffle_gap_mse_weighted": shuffle_rec["mse_weighted"] - rec_metrics["mse_weighted"],
        "shuffle_gap_auprc": occ_f1["auprc"] - shuffle_occ["auprc"],
        "probe_avg_r2": probe_z_avg_r2,
        "x_probe_avg_r2": average_probe_r2(x_probe_results),
        "z_vs_c_gain": probe_z_avg_r2 - probe_c_avg_r2,
    }
    return summary


def print_stage1_final_summary(
    summary: Dict[str, float],
    title: str = "Stage 1 Final Result",
) -> None:
    """Print compact final metrics only (aligned with ExperimentFirst.ipynb)."""
    print("=" * 58)
    print(title)
    print("=" * 58)
    for key, label in FINAL_SUMMARY_KEYS:
        val = summary.get(key, np.nan)
        if isinstance(val, float):
            if np.isnan(val):
                print(f"  {label:32s}: nan")
            else:
                print(f"  {label:32s}: {val:.4f}")
        else:
            print(f"  {label:32s}: {val}")

    passed = (
        summary.get("soft_sign_acc", 0.0) >= 0.90
        and summary.get("gate_corr", 0.0) >= 0.40
        and summary.get("best_f1", 0.0) >= 0.45
        and summary.get("shuffle_gap_auprc", 0.0) >= 0.20
        and summary.get("z_vs_c_gain", 0.0) >= 0.30
    )
    verdict = "PASS — ready for Stage 2" if passed else "REVIEW — check weak metrics above"
    print("-" * 58)
    print(f"  {'verdict':32s}: {verdict}")
    print("=" * 58)


def run_result_check(cfg: Optional[Dict] = None) -> Dict:
    """Load pack + checkpoint; compute metrics and print final summary only."""
    cfg = merge_check_cfg(cfg)
    result_dir = Path(cfg["result_dir"])
    if not result_dir.is_absolute():
        result_dir = SCRIPT_DIR / result_dir
    prefix = cfg["output_prefix"]
    ckpt_path = result_dir / cfg["checkpoint_name"]

    pack = load_anomaly_pack(result_dir, prefix, cfg)
    data = pack["data"]
    cond = pack["cond"]
    data_f = pack["data_f"]
    weight = pack["weight"]
    mask_hard = pack["mask_hard"]
    z_a_saved = pack["z_a"]

    model = load_latent_encoder(ckpt_path)
    device = next(model.parameters()).device
    batch_size = cfg["batch_size"]

    outputs = collect_model_outputs(
        model=model,
        data_f=data_f,
        weight=weight,
        cond=cond,
        batch_size=batch_size,
        device=device,
    )

    target = outputs["target"]
    weight_eval = outputs["weight"]
    a_hat = outputs["a_hat"]
    p_hat = outputs["p_hat"]
    v_hat = outputs["v_hat"]
    z_from_model = outputs["z"]

    z_a_flat = z_a_saved.squeeze(-1) if z_a_saved.ndim == 3 else z_a_saved
    z_diff = float(np.max(np.abs(z_from_model - z_a_flat)))

    rec_metrics = evaluate_weighted_reconstruction(
        target=target, a_hat=a_hat, weight=weight_eval
    )
    gate_metrics = evaluate_gate_calibration(weight=weight_eval, p_hat=p_hat)

    tau_list = cfg.get("tau_list")
    if tau_list is None:
        tau_list = np.linspace(0.05, 0.95, 50)

    tau_f1, best_f1, _ = sweep_best_tau_pred(
        p_hat=p_hat,
        mask_hard=mask_hard,
        tau_list=tau_list,
        beta=1.0,
        select_by="f1",
        verbose=False,
    )

    tau_f2, best_f2, _ = sweep_best_tau_pred(
        p_hat=p_hat,
        mask_hard=mask_hard,
        tau_list=tau_list,
        beta=cfg["tau_beta_f2"],
        select_by="fbeta",
        verbose=False,
    )

    occ_f1 = evaluate_occurrence_at_tau(mask_hard=mask_hard, p_hat=p_hat, tau_pred=tau_f1)
    occ_f2 = evaluate_occurrence_at_tau(mask_hard=mask_hard, p_hat=p_hat, tau_pred=tau_f2)

    sparse_f1 = compute_sparse_hit_metrics(
        target=target,
        p_hat=p_hat,
        v_hat=v_hat,
        mask_hard=mask_hard,
        tau_pred=tau_f1,
        use_soft_amplitude=cfg["use_soft_amplitude"],
    )
    sparse_f2 = compute_sparse_hit_metrics(
        target=target,
        p_hat=p_hat,
        v_hat=v_hat,
        mask_hard=mask_hard,
        tau_pred=tau_f2,
        use_soft_amplitude=cfg["use_soft_amplitude"],
    )

    shuffle_results = run_shuffle_z_test(
        model=model,
        z=z_from_model,
        target=target,
        weight=weight_eval,
        mask_hard=mask_hard,
        tau_pred=tau_f1,
        seed=cfg["shuffle_z_seed"],
        batch_size=batch_size,
        device=device,
    )

    stats = extract_anomaly_statistics(data_r=target, mask_hard=mask_hard)
    probe_results = run_latent_probing(z=z_from_model, stats=stats)
    x_stats = extract_load_behavior_statistics(data)
    x_probe_results = run_latent_probing(z=z_from_model, stats=x_stats, regression_keys=sorted(x_stats.keys()))
    cond_2d = cond.reshape(cond.shape[0], -1)
    probe_c_results = run_latent_probing(z=cond_2d, stats=stats)

    summary = build_result_check_summary(
        cfg=cfg,
        weight_eval=weight_eval,
        mask_hard=mask_hard,
        rec_metrics=rec_metrics,
        gate_metrics=gate_metrics,
        occ_f1=occ_f1,
        best_f1=best_f1,
        best_f2=best_f2,
        metrics_sparse_f1=sparse_f1,
        shuffle_results=shuffle_results,
        probe_results=probe_results,
        x_probe_results=x_probe_results,
        probe_c_results=probe_c_results,
    )
    print_stage1_final_summary(summary)

    return {
        "pack": pack,
        "outputs": outputs,
        "summary": summary,
        "rec_metrics": rec_metrics,
        "gate_metrics": gate_metrics,
        "tau_pred_f1": tau_f1,
        "tau_pred_f2": tau_f2,
        "best_f1": best_f1,
        "best_f2": best_f2,
        "occ_f1": occ_f1,
        "occ_f2": occ_f2,
        "sparse_f1": sparse_f1,
        "sparse_f2": sparse_f2,
        "shuffle_results": shuffle_results,
        "probe_results": probe_results,
        "x_probe_results": x_probe_results,
        "probe_c_results": probe_c_results,
        "z_diff": z_diff,
    }


if __name__ == "__main__":
    run_result_check(CHECK_CONFIG)
