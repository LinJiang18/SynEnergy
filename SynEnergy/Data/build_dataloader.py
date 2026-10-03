import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import joblib
from typing import Any, Optional, Tuple


class Normalizer:
    def __init__(self, scaler, auto_norm=True):
        self.scaler = scaler
        self.auto_norm = auto_norm

    def normalize(self, x):
        # x: (M, C)
        x = self.scaler.transform(x)
        if self.auto_norm:
            x = x * 2.0 - 1.0
        return x

    def unnormalize(self, x):
        # x: (M, C)
        if self.auto_norm:
            x = (x + 1.0) / 2.0
        return self.scaler.inverse_transform(x)


def normalize_threshold(normalizer, threshold, feature_size=1):
    """
    Convert threshold from original (raw) space to normalized space using the same
    normalizer as the data (MinMaxScaler + *2-1 to [-1, 1]).

    Args:
        normalizer: Normalizer instance with .normalize(x) where x is (M, C).
        threshold: scalar or array of shape (C,) in original space.
        feature_size: number of channels C (used when threshold is scalar).

    Returns:
        float if feature_size==1 and threshold is scalar, else np.ndarray of shape (C,).
    """
    if np.isscalar(threshold) or isinstance(threshold, (int, float)):
        threshold = np.array([float(threshold)] * feature_size, dtype=np.float64)
    else:
        threshold = np.asarray(threshold, dtype=np.float64).flatten()
        if threshold.size != feature_size:
            raise ValueError(
                "threshold size {} must match feature_size {}.".format(threshold.size, feature_size)
            )
    x = threshold.reshape(1, -1)  # (1, C)
    out = normalizer.normalize(x)  # (1, C)
    if out.size == 1:
        return float(out.flat[0])
    return out.flatten()


class SplitWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length):
        """
        data: np.ndarray of shape (M, C) — long series, sliced into windows of length seq_length.
        seq_length: int
        """
        self.data = torch.from_numpy(data).float()
        self.seq_length = seq_length

        # number of full chunks
        self.num_chunks = self.data.shape[0] // seq_length

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        return self.data[start:end]


class PrechunkedDataset(torch.utils.data.Dataset):
    """Normalized samples already shaped (N, L, C); one batch item per row."""

    def __init__(self, data_nlc: np.ndarray):
        x = np.asarray(data_nlc)
        if x.ndim != 3:
            raise ValueError("PrechunkedDataset expects (N, L, C), got shape={}".format(x.shape))
        self.data = torch.from_numpy(x.astype(np.float32, copy=False)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def load_latent_npy(path: str, expected_n: int = None) -> np.ndarray:
    """Load z_a latent table; accepts (N, D, 1) or (N, D)."""
    z = np.load(path, allow_pickle=False)
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 3 and z.shape[-1] == 1:
        z = z[:, :, 0]
    if z.ndim != 2:
        raise ValueError("latent must be (N, D) or (N, D, 1), got shape={}".format(z.shape))
    if expected_n is not None and z.shape[0] != expected_n:
        raise ValueError(
            "latent rows {} must match dataset N {}.".format(z.shape[0], expected_n)
        )
    return z


class ZAwareConditionalDataset(torch.utils.data.Dataset):
    """(N, L, C) series + cond (N, C1) + z_a (N, D) aligned by index."""

    def __init__(
        self,
        data_nlc: np.ndarray,
        cond_nc: np.ndarray,
        z_a_nd: np.ndarray,
        baseline_nlc: Optional[np.ndarray] = None,
    ):
        x = np.asarray(data_nlc)
        cond = np.asarray(cond_nc)
        z_a = np.asarray(z_a_nd)
        if x.ndim != 3:
            raise ValueError("ZAwareConditionalDataset expects data (N, L, C), got {}".format(x.shape))
        if cond.ndim != 2:
            raise ValueError("ZAwareConditionalDataset expects cond (N, C1), got {}".format(cond.shape))
        if z_a.ndim != 2:
            raise ValueError("ZAwareConditionalDataset expects z_a (N, D), got {}".format(z_a.shape))
        n = x.shape[0]
        if cond.shape[0] != n or z_a.shape[0] != n:
            raise ValueError(
                "N mismatch: data={}, cond={}, z_a={}".format(n, cond.shape[0], z_a.shape[0])
            )
        self.data = torch.from_numpy(x.astype(np.float32, copy=False)).float()
        self.cond = torch.from_numpy(cond.astype(np.int64, copy=False)).long()
        self.z_a = torch.from_numpy(z_a.astype(np.float32, copy=False)).float()
        self.baseline = None
        if baseline_nlc is not None:
            baseline = np.asarray(baseline_nlc)
            if baseline.shape != x.shape:
                raise ValueError(
                    "baseline shape {} must match data shape {}".format(baseline.shape, x.shape)
                )
            self.baseline = torch.from_numpy(baseline.astype(np.float32, copy=False)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.baseline is not None:
            return self.data[idx], self.cond[idx], self.z_a[idx], self.baseline[idx]
        return self.data[idx], self.cond[idx], self.z_a[idx]


class ConditionalPrechunkedDataset(torch.utils.data.Dataset):
    """(N, L, C) series with per-sample condition (N, C1)."""

    def __init__(self, data_nlc: np.ndarray, cond_nc: np.ndarray):
        x = np.asarray(data_nlc)
        cond = np.asarray(cond_nc)
        if x.ndim != 3:
            raise ValueError("ConditionalPrechunkedDataset expects data (N, L, C), got {}".format(x.shape))
        if cond.ndim != 2:
            raise ValueError("ConditionalPrechunkedDataset expects cond (N, C1), got {}".format(cond.shape))
        if x.shape[0] != cond.shape[0]:
            raise ValueError(
                "data N={} and cond N={} must match.".format(x.shape[0], cond.shape[0])
            )
        self.data = torch.from_numpy(x.astype(np.float32, copy=False)).float()
        self.cond = torch.from_numpy(cond.astype(np.int64, copy=False)).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.cond[idx]

def split_to_chunks(data, seq_length):
    """
    If data is 2D: (M, C) -> split into (N, seq_length, C) by trimming.
    If data is 3D: (N, L, C) -> require L == seq_length, then return as-is.
    """
    x = np.asarray(data)

    if x.ndim == 2:
        M, C = x.shape
        num_chunks = M // seq_length
        trimmed = x[: num_chunks * seq_length]
        chunks = trimmed.reshape(num_chunks, seq_length, C)
        return chunks

    if x.ndim == 3:
        N, L, C = x.shape
        if L != seq_length:
            raise ValueError(
                "data is (N, L, C) but L != seq_length: L={}, seq_length={}".format(L, seq_length)
            )
        return x

    raise ValueError(f"data must be 2D (M, C) or 3D (N, L, C), got shape={x.shape}")

def load_npy_or_npz_X(path):
    """
    Load data from .npy or .npz.
    - .npz: return arr["X"]
    Post:
      - if 1D: (M,) -> (M,1)
      - if 2D: (M,C) keep (contiguous stream; windows are formed by chunking)
      - if 3D: (N,L,C) keep — each of N rows is one sample of length L
    """
    obj = np.load(path, allow_pickle=False)

    if isinstance(obj, np.lib.npyio.NpzFile):
        if "X" not in obj.files:
            raise KeyError(f"NPZ file has no 'X'. available keys: {obj.files}")
        x = obj["X"]
        obj.close()
    else:
        x = obj

    x = np.asarray(x)

    if x.ndim == 1:
        x = x[:, None]

    return x


def load_cond_npy(
    path: str,
    cond_dim: int,
    expected_n: int = None,
    num_classes_per_cond=None,
):
    """
    Load integer condition indices from .npy.
    Accepts (N, C1, 1), (N, C1), or (N,) when cond_dim == 1.
    No normalization is applied.
    """
    cond = np.load(path, allow_pickle=False)
    cond = np.asarray(cond)

    if cond.ndim == 3 and cond.shape[-1] == 1:
        cond = cond[:, :, 0]
    elif cond.ndim == 2:
        pass
    elif cond.ndim == 1 and cond_dim == 1:
        cond = cond[:, None]
    else:
        raise ValueError(
            "cond must be (N, C1, 1), (N, C1), or (N,) when C1=1; got shape={}".format(cond.shape)
        )

    if cond.shape[1] != cond_dim:
        raise ValueError(
            "cond second dim {} must match cond_dim {}.".format(cond.shape[1], cond_dim)
        )

    if expected_n is not None and cond.shape[0] < expected_n:
        raise ValueError(
            "cond has {} rows but {} required.".format(cond.shape[0], expected_n)
        )

    cond = cond.astype(np.int64)

    if num_classes_per_cond is not None:
        for j, k in enumerate(num_classes_per_cond):
            cmin = int(cond[:, j].min())
            cmax = int(cond[:, j].max())
            if cmin < 0 or cmax >= int(k):
                raise ValueError(
                    "cond channel {} values must be in [0, {}), got [{}, {}].".format(
                        j, k, cmin, cmax
                    )
                )

    return cond


def _cond_labels_from_array(cond: np.ndarray) -> np.ndarray:
    """Extract per-row condition labels for stratified sampling."""
    cond = np.asarray(cond)
    if cond.ndim == 3 and cond.shape[-1] == 1:
        cond = cond[:, :, 0]
    if cond.ndim == 2:
        labels = cond[:, 0]
    elif cond.ndim == 1:
        labels = cond
    else:
        raise ValueError("cond must be (N, C1), (N, C1, 1), or (N,) for stratified sampling.")
    return labels.astype(np.int64)


def _stratified_sample_row_indices(
    cond: np.ndarray,
    num_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample row indices without replacement, preserving condition proportions.

    Uses largest-remainder allocation per condition class, then random draw
    within each class.
    """
    n = int(cond.shape[0])
    if num_samples > n:
        raise ValueError(
            "Requested num_samples {} exceeds available rows {}.".format(num_samples, n)
        )
    if num_samples == n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    labels = _cond_labels_from_array(cond)
    unique, counts = np.unique(labels, return_counts=True)

    raw = num_samples * counts.astype(np.float64) / float(n)
    alloc = np.floor(raw).astype(np.int64)
    remainder = int(num_samples - alloc.sum())
    if remainder > 0:
        frac = raw - alloc
        for idx in np.argsort(-frac)[:remainder]:
            alloc[idx] += 1

    alloc = np.minimum(alloc, counts)
    while int(alloc.sum()) < num_samples:
        spare = counts - alloc
        candidates = np.where(spare > 0)[0]
        if candidates.size == 0:
            break
        pick = candidates[np.argmax(spare[candidates])]
        alloc[pick] += 1

    selected = []
    for cls, k in zip(unique, alloc):
        if k <= 0:
            continue
        pool = np.flatnonzero(labels == cls)
        k = min(int(k), pool.size)
        selected.append(rng.choice(pool, size=k, replace=False))

    indices = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    if indices.size < num_samples:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), indices)
        if remaining.size == 0:
            raise ValueError("Unable to draw {} stratified samples from N={}.".format(num_samples, n))
        extra = rng.choice(remaining, size=num_samples - indices.size, replace=False)
        indices = np.concatenate([indices, extra])
    elif indices.size > num_samples:
        indices = rng.choice(indices, size=num_samples, replace=False)

    rng.shuffle(indices)
    return indices.astype(np.int64)


def _sample_row_indices(n: int, num_samples: int) -> np.ndarray:
    if num_samples > n:
        raise ValueError(
            "Requested num_samples {} exceeds available rows {}.".format(num_samples, n)
        )
    if num_samples == n:
        return np.arange(n, dtype=np.int64)
    return np.random.choice(n, size=num_samples, replace=False)


def _sample_indices_from_config(
    cond: np.ndarray,
    num_samples: int,
    config: dict,
) -> np.ndarray:
    """Stratified by condition proportions; seed from sample.params.sample_seed."""
    sample_params = config.get("sample", {}).get("params", {})
    seed = sample_params.get("sample_seed", 42)
    return _stratified_sample_row_indices(cond, num_samples, seed=seed)


def load_sample_cond(
    config: dict,
    num_samples: int,
    return_indices: bool = False,
):
    """Load sample-time conditions from sample.params.cond_path (required)."""
    sample_params = config.get("sample", {}).get("params", {})
    cond_path = sample_params.get("cond_path")
    if not cond_path:
        raise ValueError(
            "sample.params.cond_path is required when using --condition for sampling."
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
    indices = _sample_indices_from_config(cond, num_samples, config)
    sampled = cond[indices]
    if return_indices:
        return sampled, indices
    return sampled


def load_stage2_sample_pairs(
    config: dict, num_samples: int, indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load index-aligned (cond, z_a) for Stage 2 sampling.

    cond and z_a always come from the same row indices. When num_samples is less
    than the table size, rows are drawn without replacement (unless indices is given).
    """
    sample_params = config.get("sample", {}).get("params", {})
    stage2 = config.get("stage2", {})
    cond_path = sample_params.get("cond_path")
    latent_path = stage2.get("latent_path")
    if not cond_path:
        raise ValueError(
            "sample.params.cond_path is required for Stage 2 sampling."
        )
    if not latent_path:
        raise ValueError("stage2.latent_path is required for Stage 2 sampling.")

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
            "cond rows {} and latent rows {} must match for aligned sampling.".format(
                n, z.shape[0]
            )
        )

    if indices is None:
        indices = _sample_indices_from_config(cond, num_samples, config)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape[0] != num_samples:
            raise ValueError(
                "indices length {} must match num_samples {}.".format(
                    indices.shape[0], num_samples
                )
            )
        if indices.min() < 0 or indices.max() >= n:
            raise ValueError("indices out of range for N={}.".format(n))

    return cond[indices], z[indices], indices


def load_sample_z_a(config: dict, num_samples: int) -> np.ndarray:
    """Load z_a for sampling; prefer load_stage2_sample_pairs for aligned cond+z_a."""
    _, z, _ = load_stage2_sample_pairs(config, num_samples)
    return z


def build_dataloader(
    config: dict,
    args: Any,
):

    # -------- config --------
    train_params = config["train"]["params"]
    npy_path = train_params["data_path"]
    seq_length = train_params["seq_length"]
    batch_size = train_params["batch_size"]

    # -------- load raw data --------
    # data = np.load(npy_path)
    # if data.ndim == 1:
    #     data = data[:, None]
    # # (M, C)
    data = load_npy_or_npz_X(npy_path)

    if data.ndim == 3:
        N, L, C = data.shape
        if L != seq_length:
            raise ValueError(
                "3D data_path has L={} but train.params.seq_length={}; align config or reshape data.".format(
                    L, seq_length
                )
            )
    elif data.ndim == 2:
        M, _C = data.shape
        n_keep = (M // seq_length) * seq_length
        if n_keep < seq_length:
            raise ValueError(
                "2D data_path has fewer than one full window: M={}, seq_length={}, path={}.".format(
                    M, seq_length, npy_path
                )
            )
        data = data[:n_keep]

    # Full aligned stream / samples: scaler + origin_*.npy always match this (same size as chunked dataset).
    scaler = MinMaxScaler()
    if data.ndim == 2:
        scaler.fit(data)
    else:
        scaler.fit(data.reshape(-1, data.shape[-1]))

    normalizer = Normalizer(scaler, auto_norm=True)
    origin_unnorm = split_to_chunks(data, seq_length)  # (N, L, C)
    if data.ndim == 2:
        data_norm_full = normalizer.normalize(data)
        origin_norm = split_to_chunks(data_norm_full, seq_length)
    else:
        flat = data.reshape(-1, data.shape[-1])
        data_norm_full = normalizer.normalize(flat).reshape(data.shape)
        origin_norm = data_norm_full

    # -------- save both (full aligned dataset; N matches len(dataset) when not restricted) --------
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"origin_data_unnorm_{args.name}.npy"), origin_unnorm)
    np.save(os.path.join(args.output_dir, f"origin_data_norm_{args.name}.npy"), origin_norm)
    joblib.dump({"scaler": scaler, "auto_norm": True}, os.path.join(args.output_dir, "normalizer.joblib"))

    data_norm = data_norm_full

    use_condition = bool(getattr(args, "condition", False))
    cond_array = None

    if use_condition and getattr(args, "train", False):
        model_params = config.get("model", {}).get("params", {})
        cond_dim = int(model_params.get("cond_dim", 1))
        num_classes_per_cond = model_params.get("num_classes_per_cond")
        cond_path = train_params.get("cond_path")
        if not cond_path:
            raise ValueError(
                "train.params.cond_path is required when using --condition for training."
            )
        n_samples = origin_unnorm.shape[0]
        cond_array = load_cond_npy(
            cond_path,
            cond_dim=cond_dim,
            expected_n=n_samples,
            num_classes_per_cond=num_classes_per_cond,
        )
        if cond_array.shape[0] != n_samples:
            raise ValueError(
                "cond rows {} must match dataset samples {}.".format(
                    cond_array.shape[0], n_samples
                )
            )

    if data_norm.ndim == 3:
        if cond_array is not None:
            dataset = ConditionalPrechunkedDataset(data_norm, cond_array)
        else:
            dataset = PrechunkedDataset(data_norm)
    else:
        if cond_array is not None:
            raise ValueError(
                "Conditional training requires 3D data_path (N, L, C) aligned with cond."
            )
        dataset = SplitWindowDataset(data_norm, seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return {"dataloader": dataloader, "normalizer": normalizer, "scaler": scaler}


def build_stage2_dataloader(config: dict, args: Any):
    """Build (x_norm, cond, z_a) loader for Stage 2 ControlNet training."""
    train_params = config["train"]["params"]
    stage2 = config.get("stage2", {})
    npy_path = train_params["data_path"]
    seq_length = train_params["seq_length"]
    batch_size = train_params["batch_size"]
    latent_path = stage2.get("latent_path")
    if not latent_path:
        raise ValueError("stage2.latent_path is required for Stage 2 training.")

    data = load_npy_or_npz_X(npy_path)
    if data.ndim == 3:
        if data.shape[1] != seq_length:
            raise ValueError(
                "3D data L={} != seq_length={}".format(data.shape[1], seq_length)
            )
    elif data.ndim == 2:
        n_keep = (data.shape[0] // seq_length) * seq_length
        data = data[:n_keep]

    backbone_output = stage2.get(
        "backbone_output_dir",
        os.path.join("OUTPUT", stage2.get("backbone_run_name", args.name)),
    )
    normalizer = load_normalizer_from_disk(backbone_output)
    if normalizer is None:
        scaler = MinMaxScaler()
        if data.ndim == 2:
            scaler.fit(data)
        else:
            scaler.fit(data.reshape(-1, data.shape[-1]))
        normalizer = Normalizer(scaler, auto_norm=True)

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
        raise ValueError("train.params.cond_path is required for Stage 2.")
    cond_array = load_cond_npy(
        cond_path,
        cond_dim=cond_dim,
        expected_n=n_samples,
        num_classes_per_cond=num_classes_per_cond,
    )

    z_array = load_latent_npy(latent_path, expected_n=n_samples)

    dataset = ZAwareConditionalDataset(origin_norm, cond_array, z_array)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return {
        "dataloader": dataloader,
        "normalizer": normalizer,
        "origin_unnorm": origin_unnorm,
    }


def load_normalizer_from_disk(output_dir):
    """Load normalizer from normalizer.joblib; returns None if file not found."""
    path = os.path.join(output_dir, "normalizer.joblib")
    if not os.path.isfile(path):
        return None
    d = joblib.load(path)
    return Normalizer(d["scaler"], d.get("auto_norm", True))


def unnormalize_sample(samples: np.ndarray, normalizer) -> np.ndarray:
    """
    samples: np.ndarray, shape (B, T, 1)
    normalizer: instance of Normalizer(scaler, auto_norm=...)
    return: np.ndarray, shape (B, T, 1) in original scale
    """
    samples = np.asarray(samples)
    assert samples.ndim == 3 , f"expected dim = 3"

    B, T, C = samples.shape
    x2d = samples.reshape(B * T, C)
    x2d = normalizer.unnormalize(x2d)
    return x2d.reshape(B, T, C)


def unnormalize_torch(x: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    """Differentiable unnormalize for Stage 2 anomaly loss (B, T, C)."""
    if normalizer.auto_norm:
        x = (x + 1.0) / 2.0
    scale = torch.as_tensor(
        normalizer.scaler.scale_, device=x.device, dtype=x.dtype
    )
    min_ = torch.as_tensor(
        normalizer.scaler.min_, device=x.device, dtype=x.dtype
    )
    b, t, c = x.shape
    x2d = x.reshape(b * t, c)
    x2d = (x2d - min_) / scale
    return x2d.reshape(b, t, c)

