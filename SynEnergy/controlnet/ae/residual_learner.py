"""
Zero-aware residual AE for the controlnet pipeline.

All preprocessing and training run in normalized data space (backbone MinMax scaler).
Input: normalized ``data`` + categorical ``cond`` (cast to float for the AE encoder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from controlnet.graph.Graph_Aggregation import (
    apply_dual_graph_encoder_input,
    apply_graph_encoder_input,
    load_affinity_matrix,
)
from controlnet.graph.graph_trainer import (
    GRAPH_CKPT_NAME,
    TrainableDualRegionGraph,
    TrainableRegionGraph,
    apply_graph_encoder_input_torch,
    load_graph_checkpoint,
    save_graph_checkpoint,
    train_region_graph,
)

# Stage 1 defaults (ExperimentFirst ablation; synced with power_controlnet.yaml).
STAGE1_FILTER_THRESHOLD = 0.12
STAGE1_OCCURRENCE_THRESHOLD = 0.12
STAGE1_FILTER_GAMMA = 50.0
FIXED_FILTER_THRESHOLD = STAGE1_FILTER_THRESHOLD
FIXED_POS_WEIGHT_VALUE = 5.0

# A4 loss (gate_zero on w<0.1 + gate_anom on w≥0.5).
DEFAULT_LAMBDA_GATE_ZERO = 0.2
DEFAULT_LAMBDA_GATE_ANOM = 0.2
DEFAULT_GATE_NORMAL_THRESHOLD = 0.1
DEFAULT_GATE_ANOM_THRESHOLD = 0.3
DEFAULT_ANOM_WEIGHT_THRESHOLD = 0.5

DEFAULT_AE_CFG: Dict = {
    "num_days": 30,
    "slots_per_day": 6,
    "residual_eps": 1e-6,
    "min_scale": 1.0,
    "use_mad_consistency": True,

    "filter_threshold": STAGE1_FILTER_THRESHOLD,
    "filter_mode": "soft",
    "filter_gamma": STAGE1_FILTER_GAMMA,

    "seq_len": 180,
    "hidden_dim": 64,
    "latent_dim": 64,
    "num_layers": 3,
    "dropout": 0.1,
    "output_scale": 1.0,
    "model_eps": 1e-6,

    "batch_size": 128,
    "extract_batch_size": 256,
    "lr": 1e-3,
    "num_epochs": 200,
    "device": None,

    "lambda_mask": 1.0,
    "lambda_anom": 2.0,
    "lambda_zero": 1.0,
    "lambda_gate_zero": DEFAULT_LAMBDA_GATE_ZERO,
    "lambda_gate_anom": DEFAULT_LAMBDA_GATE_ANOM,
    "gate_normal_threshold": DEFAULT_GATE_NORMAL_THRESHOLD,
    "gate_anom_threshold": DEFAULT_GATE_ANOM_THRESHOLD,
    "anom_weight_threshold": DEFAULT_ANOM_WEIGHT_THRESHOLD,
    "lambda_sign": 0.1,
    "pos_weight_value": 1.0,
    "occurrence_threshold": STAGE1_OCCURRENCE_THRESHOLD,
    "loss_eps": 1e-6,
    "graph": {
        "enabled": False,
        "train": False,
        "dual_graph": False,
        "inject_scale": 0.0,
        "lambda_pre": 0.1,
        "graph_lr": 1.0e-3,
        "graph_epochs": 50,
        "graph_ckpt": GRAPH_CKPT_NAME,
        "affinity_path": None,
        "spatial_affinity_path": None,
        "attribute_affinity_path": None,
        "max_hop": 2,
        "edge_eps": 0.0,
        "eps": 1e-8,
        "gate_bias_init": -10.0,
        "init_identity": True,
    },
}


# =============================================================================
# NumPy preprocessing (offline / saving)
# =============================================================================

def extract_standardized_residual(
    data: np.ndarray,
    num_days: int = 30,
    slots_per_day: int = 6,
    eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stable per-user standardized residual."""
    expected_len = num_days * slots_per_day

    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")
    if data.ndim != 3:
        raise ValueError(f"data must have shape (N, {expected_len}, 1), got {data.shape}")

    n_samples, time_len, n_features = data.shape
    if time_len != expected_len:
        raise ValueError(f"time dimension must be {expected_len}, got {time_len}")
    if n_features != 1:
        raise ValueError(f"last dimension must be 1, got {n_features}")

    x = data.reshape(n_samples, num_days, slots_per_day)
    if baseline is None:
        baseline = np.median(x, axis=1, keepdims=True)
    else:
        baseline = np.asarray(baseline, dtype=data.dtype)
        if baseline.size != data.size:
            raise ValueError(
                f"baseline must match data shape {data.shape}, got {baseline.shape}"
            )
        baseline = baseline.reshape(n_samples, num_days, slots_per_day)
    raw_residual = x - baseline
    mad = np.median(np.abs(raw_residual), axis=1, keepdims=True)

    scale = (1.4826 * mad) if use_mad_consistency else mad.copy()
    scale = np.maximum(scale, min_scale)

    standardized_residual = raw_residual / (scale + eps)
    standardized_residual = standardized_residual.reshape(n_samples, expected_len, 1)
    raw_residual_out = raw_residual.reshape(n_samples, expected_len, 1)

    return standardized_residual, baseline, scale, raw_residual_out, mad


def filter_residual_deviation(
    data_r: np.ndarray,
    threshold: float = 2.0,
    mode: str = "soft",
    gamma: float = 5.0,
    eps: float = 1e-6,
    return_mask: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Zero-aware filtering on standardized residual."""
    del eps
    if not isinstance(data_r, np.ndarray):
        raise TypeError(f"data_r must be np.ndarray, got {type(data_r)}")
    if data_r.ndim != 3 or data_r.shape[-1] != 1:
        raise ValueError(f"data_r must have shape (N, T, 1), got {data_r.shape}")
    if mode not in ("hard", "soft"):
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")

    abs_r = np.abs(data_r)

    if mode == "hard":
        mask = abs_r > threshold
        filtered_residual = data_r * mask
        if return_mask:
            return filtered_residual, mask.astype(np.float32)
        return filtered_residual

    weight = 1.0 / (1.0 + np.exp(-gamma * (abs_r - threshold)))
    filtered_residual = data_r * weight
    if return_mask:
        return filtered_residual, weight.astype(np.float32)
    return filtered_residual


def occurrence_mask_from_data_r(
    data_r: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """Hard 0/1 occurrence label: |standardized residual| > threshold."""
    return (np.abs(data_r) > threshold).astype(np.float32)


def prepare_filtered_residual(
    data: np.ndarray,
    cfg: Optional[Dict] = None,
    cond: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
    """
    data -> data_r -> data_f, weight (soft gate), mask_hard (occurrence for eval).

    When ``num_days * slots_per_day != seq_len``, rows are grouped by ``cond``,
    reshaped to (1, num_days, slots_per_day) per zone, and mapped back to daily rows.

    Returns:
        data_f, weight, mask_hard, data_r, cond_group_stats (or None for FL path)
    """
    cfg = cfg or DEFAULT_AE_CFG
    from controlnet.ae.cond_grouped_residual import (
        extract_standardized_residual_by_cond,
        uses_cond_grouped_residual,
    )

    if uses_cond_grouped_residual(cfg):
        if cond is None:
            raise ValueError(
                "cond is required when num_days * slots_per_day != seq_len"
            )
        data_r, cond_group_stats = extract_standardized_residual_by_cond(
            data,
            cond,
            num_days=int(cfg["num_days"]),
            slots_per_day=int(cfg["slots_per_day"]),
            eps=float(cfg["residual_eps"]),
            min_scale=float(cfg["min_scale"]),
            use_mad_consistency=bool(cfg["use_mad_consistency"]),
        )
    else:
        data_r, _, _, _, _ = extract_standardized_residual(
            data,
            num_days=cfg["num_days"],
            slots_per_day=cfg["slots_per_day"],
            eps=cfg["residual_eps"],
            min_scale=cfg["min_scale"],
            use_mad_consistency=cfg["use_mad_consistency"],
            baseline=baseline,
        )
        cond_group_stats = None

    occ_threshold = float(cfg.get("occurrence_threshold", cfg["filter_threshold"]))
    mask_hard = occurrence_mask_from_data_r(data_r, threshold=occ_threshold)

    data_f, weight = filter_residual_deviation(
        data_r,
        threshold=cfg["filter_threshold"],
        mode=cfg["filter_mode"],
        gamma=cfg["filter_gamma"],
        return_mask=True,
    )
    return (
        data_f.astype(np.float32),
        weight.astype(np.float32),
        mask_hard.astype(np.float32),
        data_r.astype(np.float32),
        cond_group_stats,
    )



# =============================================================================
# Torch preprocessing (differentiable; controlnet anomaly loss)
# =============================================================================

def extract_standardized_residual_torch(
    data: torch.Tensor,
    num_days: int = 30,
    slots_per_day: int = 6,
    eps: float = 1e-6,
    min_scale: float = 1.0,
    use_mad_consistency: bool = True,
    detach_stats: bool = True,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable standardized residual; median/MAD detached when detach_stats=True."""
    expected_len = num_days * slots_per_day
    if data.ndim != 3 or data.shape[-1] != 1:
        raise ValueError(f"data must be (N, {expected_len}, 1), got {tuple(data.shape)}")
    if data.shape[1] != expected_len:
        raise ValueError(f"time dimension must be {expected_len}, got {data.shape[1]}")

    n_samples = data.shape[0]
    x = data.reshape(n_samples, num_days, slots_per_day)
    if baseline is None:
        baseline = torch.median(x, dim=1, keepdim=True).values
    else:
        baseline = baseline.to(device=x.device, dtype=x.dtype).reshape(
            n_samples, num_days, slots_per_day
        )
    if detach_stats:
        baseline = baseline.detach()

    raw_residual = x - baseline
    mad = torch.median(torch.abs(raw_residual), dim=1, keepdim=True).values
    if detach_stats:
        mad = mad.detach()

    scale = (1.4826 * mad) if use_mad_consistency else mad
    scale = torch.clamp(scale, min=min_scale)
    standardized = raw_residual / (scale + eps)
    return standardized.reshape(n_samples, expected_len, 1)


def filter_residual_deviation_torch(
    data_r: torch.Tensor,
    threshold: float = 2.0,
    mode: str = "soft",
    gamma: float = 5.0,
    return_mask: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Differentiable filter; default soft sigmoid gate."""
    if mode not in ("hard", "soft"):
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")

    abs_r = torch.abs(data_r)
    if mode == "hard":
        mask = (abs_r > threshold).to(data_r.dtype)
        filtered = data_r * mask
    else:
        mask = torch.sigmoid(gamma * (abs_r - threshold))
        filtered = data_r * mask

    if return_mask:
        return filtered, mask
    return filtered


def filtered_residual_from_data_torch(
    data: torch.Tensor,
    cfg: Optional[Dict] = None,
    detach_stats: bool = True,
    cond: Optional[torch.Tensor] = None,
    zone_baseline: Optional[torch.Tensor] = None,
    zone_scale: Optional[torch.Tensor] = None,
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full torch pipeline: data -> data_f, weight, mask_hard."""
    cfg = cfg or DEFAULT_AE_CFG
    from controlnet.ae.cond_grouped_residual import (
        filtered_residual_from_cond_grouped_torch,
        uses_cond_grouped_residual,
    )

    if uses_cond_grouped_residual(cfg):
        if cond is None or zone_baseline is None or zone_scale is None:
            raise ValueError(
                "cond-grouped residual requires cond, zone_baseline, and zone_scale"
            )
        return filtered_residual_from_cond_grouped_torch(
            data, cond, zone_baseline, zone_scale, cfg
        )

    data_r = extract_standardized_residual_torch(
        data,
        num_days=cfg["num_days"],
        slots_per_day=cfg["slots_per_day"],
        eps=cfg["residual_eps"],
        min_scale=cfg["min_scale"],
        use_mad_consistency=cfg["use_mad_consistency"],
        detach_stats=detach_stats,
        baseline=baseline,
    )
    data_f, weight = filter_residual_deviation_torch(
        data_r,
        threshold=cfg["filter_threshold"],
        mode=cfg["filter_mode"],
        gamma=cfg["filter_gamma"],
        return_mask=True,
    )
    occ_threshold = float(cfg.get("occurrence_threshold", cfg["filter_threshold"]))
    mask_hard = (torch.abs(data_r) > occ_threshold).to(data_r.dtype)
    return data_f, weight, mask_hard


# =============================================================================
# Dataset + Autoencoder
# =============================================================================

def _prepare_cond_array(cond: np.ndarray) -> np.ndarray:
    cond = np.asarray(cond, dtype=np.float32)
    if cond.ndim == 3 and cond.shape[-1] == 1:
        cond = cond.squeeze(-1)
    if cond.ndim != 2:
        raise ValueError(f"cond must be (N, M) or (N, M, 1), got {cond.shape}")
    return cond


class ResidualDataset(Dataset):
    """Each item: residual=encoder input, target_r=reconstruction target, mask=soft weight."""

    def __init__(
        self,
        data_f: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
        cond: Union[np.ndarray, torch.Tensor],
        target_r: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        if not isinstance(data_f, torch.Tensor):
            data_f = torch.as_tensor(data_f, dtype=torch.float32)
        else:
            data_f = data_f.float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()

        if data_f.ndim != 3 or data_f.shape[-1] != 1:
            raise ValueError(f"data_f must be (N, T, 1), got {tuple(data_f.shape)}")
        if mask.shape != data_f.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != data_f {tuple(data_f.shape)}")

        if target_r is None:
            target_r = data_f
        elif not isinstance(target_r, torch.Tensor):
            target_r = torch.as_tensor(target_r, dtype=torch.float32)
        else:
            target_r = target_r.float()
        if target_r.shape != data_f.shape:
            raise ValueError(
                f"target_r shape {tuple(target_r.shape)} != data_f {tuple(data_f.shape)}"
            )

        self.data_f = data_f
        self.target_r = target_r
        self.mask = mask
        self.n_samples, self.seq_len, _ = data_f.shape

        cond_np = _prepare_cond_array(
            cond.cpu().numpy() if isinstance(cond, torch.Tensor) else cond
        )
        if cond_np.shape[0] != self.n_samples:
            raise ValueError(f"cond N={cond_np.shape[0]} != data_f N={self.n_samples}")
        self.cond = torch.as_tensor(cond_np, dtype=torch.float32)
        self.cond_dim = self.cond.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "residual": self.data_f[idx],
            "target_r": self.target_r[idx],
            "mask": self.mask[idx],
            "cond": self.cond[idx],
        }


class ZeroAwareResidualAutoencoder(nn.Module):
    """
    Encoder: [residual, weight, cond_broadcast] -> Conv1d -> pool -> z_a
    Decoder: z_a -> p_hat (soft gate) * v_hat -> a_hat
    """

    def __init__(
        self,
        cond_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        seq_len: int = 180,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_scale: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.output_scale = output_scale
        self.eps = eps

        encoder_in_ch = 2 + cond_dim
        self.encoder_convs = self._build_conv_stack(
            encoder_in_ch, hidden_dim, num_layers, dropout
        )
        self.to_latent = nn.Linear(hidden_dim * 3, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, seq_len * hidden_dim)
        self.decoder_convs = self._build_conv_stack(
            hidden_dim, hidden_dim, num_layers, dropout
        )
        self.mask_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.value_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    @staticmethod
    def _build_conv_stack(
        in_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        kernel_size: int = 5,
        padding: int = 2,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv1d(ch_in, out_channels, kernel_size=kernel_size, padding=padding)
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def encode(
        self,
        residual: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = residual.shape
        if t != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got T={t}")
        if cond is None:
            cond = torch.zeros(b, self.cond_dim, device=residual.device, dtype=residual.dtype)

        cond_seq = cond.unsqueeze(1).expand(b, t, self.cond_dim)
        encoder_input = torch.cat([residual, mask, cond_seq], dim=-1)

        h = encoder_input.transpose(1, 2)
        h = self.encoder_convs(h)

        mask_t = mask.transpose(1, 2)
        h_mean = h.mean(dim=-1)
        h_max = h.max(dim=-1).values
        h_masked = (h * mask_t).sum(dim=-1) / (mask_t.sum(dim=-1) + self.eps)
        h_pool = torch.cat([h_mean, h_max, h_masked], dim=-1)

        return self.to_latent(h_pool)

    def decode(
        self, z_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = z_a.shape[0]
        h0 = self.decoder_input(z_a).view(b, self.hidden_dim, self.seq_len)
        h_dec = self.decoder_convs(h0)

        p_logit = self.mask_head(h_dec).transpose(1, 2)
        v_raw = self.value_head(h_dec).transpose(1, 2)

        p_hat = torch.sigmoid(p_logit)
        v_hat = self.output_scale * torch.tanh(v_raw)
        a_hat = p_hat * v_hat

        return a_hat, p_logit, p_hat, v_hat

    def forward(
        self,
        residual: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_a = self.encode(residual, mask, cond)
        a_hat, p_logit, p_hat, v_hat = self.decode(z_a)
        return {
            "z_a": z_a,
            "a_hat": a_hat,
            "p_logit": p_logit,
            "p_hat": p_hat,
            "v_hat": v_hat,
        }


# =============================================================================
# Loss + Stage 1 training
# =============================================================================

def zero_aware_loss(
    a_hat: torch.Tensor,
    p_logit: torch.Tensor,
    target_r: torch.Tensor,
    mask: torch.Tensor,
    lambda_mask: float = 1.0,
    lambda_anom: float = 2.0,
    lambda_zero: float = 1.0,
    lambda_gate_zero: float = DEFAULT_LAMBDA_GATE_ZERO,
    lambda_gate_anom: float = DEFAULT_LAMBDA_GATE_ANOM,
    gate_normal_threshold: float = DEFAULT_GATE_NORMAL_THRESHOLD,
    gate_anom_threshold: float = DEFAULT_GATE_ANOM_THRESHOLD,
    anom_weight_threshold: float = DEFAULT_ANOM_WEIGHT_THRESHOLD,
    lambda_sign: float = 0.1,
    pos_weight_value: float = 1.0,
    eps: float = 1e-6,
    soft_gate: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    A4 loss (ExperimentFirst ablation winner):
      - loss_anom: weighted MSE on â vs data_f
      - loss_zero: push â → 0 in low-weight region
      - loss_gate_zero: push p_hat → 0 where weight < gate_normal_threshold
      - loss_gate_anom: push p_hat ≥ gate_anom_threshold where weight ≥ anom_weight_threshold
    """
    del lambda_mask, lambda_sign, pos_weight_value, soft_gate
    weight = mask
    p_hat = torch.sigmoid(p_logit)
    zero = torch.zeros((), device=a_hat.device, dtype=a_hat.dtype)

    loss_anom = (weight * (a_hat - target_r) ** 2).sum() / (weight.sum() + eps)
    loss_zero = ((1.0 - weight) * (a_hat ** 2)).sum() / ((1.0 - weight).sum() + eps)

    normal_mask = (weight < gate_normal_threshold).to(weight.dtype)
    loss_gate_zero = (normal_mask * (p_hat ** 2)).sum() / (normal_mask.sum() + eps)

    anom_mask = (weight >= anom_weight_threshold).to(weight.dtype)
    loss_gate_anom = (
        anom_mask * torch.relu(gate_anom_threshold - p_hat) ** 2
    ).sum() / (anom_mask.sum() + eps)

    loss = (
        lambda_anom * loss_anom
        + lambda_zero * loss_zero
        + lambda_gate_zero * loss_gate_zero
        + lambda_gate_anom * loss_gate_anom
    )

    return {
        "loss": loss,
        "loss_mask": zero,
        "loss_anom": loss_anom,
        "loss_zero": loss_zero,
        "loss_gate_zero": loss_gate_zero,
        "loss_gate_anom": loss_gate_anom,
        "loss_sign": zero,
    }


def _resolve_device(cfg: Dict) -> torch.device:
    device = cfg.get("device")
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        return torch.device(device)
    return device


def train_zero_aware_autoencoder(
    data_f: np.ndarray,
    mask: np.ndarray,
    cond: np.ndarray,
    cfg: Optional[Dict] = None,
    target_r: Optional[np.ndarray] = None,
) -> Tuple[ZeroAwareResidualAutoencoder, Dict[str, List[float]]]:
    """mask argument is soft weight. ``target_r`` defaults to ``data_f`` (encoder input)."""
    cfg = dict(cfg or DEFAULT_AE_CFG)
    cfg.setdefault("filter_threshold", FIXED_FILTER_THRESHOLD)
    cfg.setdefault("occurrence_threshold", FIXED_FILTER_THRESHOLD)
    cfg.setdefault("pos_weight_value", 1.0)
    cfg.setdefault("lambda_gate_zero", DEFAULT_LAMBDA_GATE_ZERO)
    cfg.setdefault("lambda_gate_anom", DEFAULT_LAMBDA_GATE_ANOM)
    cfg.setdefault("gate_normal_threshold", DEFAULT_GATE_NORMAL_THRESHOLD)
    cfg.setdefault("gate_anom_threshold", DEFAULT_GATE_ANOM_THRESHOLD)
    cfg.setdefault("anom_weight_threshold", DEFAULT_ANOM_WEIGHT_THRESHOLD)

    device = _resolve_device(cfg)
    dataset = ResidualDataset(
        data_f=data_f, mask=mask, cond=cond, target_r=target_r
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = ZeroAwareResidualAutoencoder(
        cond_dim=dataset.cond_dim,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["seq_len"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_scale=cfg["output_scale"],
        eps=cfg["model_eps"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    history: Dict[str, List[float]] = {
        "loss": [],
        "loss_mask": [],
        "loss_anom": [],
        "loss_zero": [],
        "loss_gate_zero": [],
        "loss_gate_anom": [],
        "loss_sign": [],
    }

    print("=== Training ZeroAwareResidualAutoencoder (soft gate) ===")
    model.train()
    for epoch in range(1, cfg["num_epochs"] + 1):
        epoch_losses = {k: 0.0 for k in history}
        n_batches = 0

        for batch in loader:
            residual = batch["residual"].to(device)
            target_r = batch["target_r"].to(device)
            weight_b = batch["mask"].to(device)
            cond_b = batch["cond"].to(device)

            out = model(residual, weight_b, cond_b)
            losses = zero_aware_loss(
                a_hat=out["a_hat"],
                p_logit=out["p_logit"],
                target_r=target_r,
                mask=weight_b,
                lambda_anom=cfg["lambda_anom"],
                lambda_zero=cfg["lambda_zero"],
                lambda_gate_zero=cfg["lambda_gate_zero"],
                lambda_gate_anom=cfg["lambda_gate_anom"],
                gate_normal_threshold=cfg["gate_normal_threshold"],
                gate_anom_threshold=cfg["gate_anom_threshold"],
                anom_weight_threshold=cfg["anom_weight_threshold"],
                eps=cfg["loss_eps"],
            )

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            for k in history:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        for k in history:
            epoch_losses[k] /= max(n_batches, 1)
            history[k].append(epoch_losses[k])

        print(
            f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
            f"loss={epoch_losses['loss']:.6f} | "
            f"anom={epoch_losses['loss_anom']:.6f} | "
            f"zero={epoch_losses['loss_zero']:.6f} | "
            f"gate0={epoch_losses['loss_gate_zero']:.6f} | "
            f"gate1={epoch_losses['loss_gate_anom']:.6f}"
        )

    print()
    return model, history


@torch.no_grad()
def extract_anomaly_latent(
    model: ZeroAwareResidualAutoencoder,
    data_f: np.ndarray,
    mask: np.ndarray,
    cond: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """z_a = f_theta(data_f, weight, cond). mask is soft weight."""
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    dataset = ResidualDataset(data_f=data_f, mask=mask, cond=cond)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    z_chunks: List[np.ndarray] = []
    for batch in loader:
        z_b = model.encode(
            batch["residual"].to(device),
            batch["mask"].to(device),
            batch["cond"].to(device),
        )
        z_chunks.append(z_b.cpu().numpy())
    return np.concatenate(z_chunks, axis=0)


def save_ae_checkpoint(
    model: ZeroAwareResidualAutoencoder,
    path: Union[str, Path],
    cfg: Optional[Dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": dict(cfg or DEFAULT_AE_CFG),
    }
    torch.save(payload, path)
    return path


def load_ae_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
    cfg: Optional[Dict] = None,
) -> ZeroAwareResidualAutoencoder:
    """Load AE checkpoint trained in the controlnet pipeline."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    merged_cfg = dict(DEFAULT_AE_CFG)
    merged_cfg.update(payload.get("config", {}))
    if cfg:
        merged_cfg.update(cfg)

    model = ZeroAwareResidualAutoencoder(
        cond_dim=1,
        hidden_dim=merged_cfg["hidden_dim"],
        latent_dim=merged_cfg["latent_dim"],
        seq_len=merged_cfg["seq_len"],
        num_layers=merged_cfg["num_layers"],
        dropout=merged_cfg["dropout"],
        output_scale=merged_cfg["output_scale"],
        eps=merged_cfg["model_eps"],
    )
    model.load_state_dict(payload["model_state_dict"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def build_ae_cfg(config: dict) -> dict:
    """Merge ``ae`` and shared filter keys from ``controlnet`` config."""
    ae = dict(config.get("ae", {}))
    cn = config.get("controlnet", config.get("stage2", {}))
    for key in (
        "num_days",
        "slots_per_day",
        "residual_eps",
        "min_scale",
        "use_mad_consistency",
        "filter_threshold",
        "filter_mode",
        "filter_gamma",
        "occurrence_threshold",
        "loss_eps",
        "lambda_anom",
        "lambda_zero",
        "lambda_gate_zero",
        "lambda_gate_anom",
        "gate_normal_threshold",
        "gate_anom_threshold",
        "anom_weight_threshold",
        "output_scale",
        "hidden_dim",
        "latent_dim",
        "num_layers",
        "dropout",
        "batch_size",
        "extract_batch_size",
        "lr",
        "num_epochs",
    ):
        if key in cn and key not in ae:
            ae[key] = cn[key]
    seq_len = config.get("model", {}).get("params", {}).get("seq_length")
    if seq_len is not None:
        ae.setdefault("seq_len", int(seq_len))
    ae.setdefault("latent_dim", config.get("model", {}).get("params", {}).get("z_a_dim", 64))
    default_graph = dict(DEFAULT_AE_CFG.get("graph", {}))
    default_graph.update(ae.get("graph", {}))
    ae["graph"] = default_graph
    return ae


def maybe_apply_graph_encoder_input(
    data_f: np.ndarray,
    cond: np.ndarray,
    cfg: dict,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    logger=None,
) -> Tuple[np.ndarray, Optional[dict], Optional[Union[TrainableRegionGraph, TrainableDualRegionGraph]]]:
    """
    Optionally train / load region graph and produce X' for the AE encoder.

    Training uses L = L_reg + lambda_pre * L_pre on per-sample embeddings (data_f).
  """
    graph_cfg = dict(cfg.get("graph", {}))
    if not graph_cfg.get("enabled", False):
        return data_f, None, None

    dual_graph = bool(graph_cfg.get("dual_graph", False))
    affinity_path = graph_cfg.get("affinity_path")
    if not affinity_path:
        raise ValueError("ae.graph.affinity_path is required when graph is enabled")

    affinity = load_affinity_matrix(affinity_path)
    affinity_spatial = None
    affinity_attribute = None
    if dual_graph:
        spatial_path = graph_cfg.get("spatial_affinity_path")
        attribute_path = graph_cfg.get("attribute_affinity_path")
        if not spatial_path or not attribute_path:
            raise ValueError(
                "ae.graph.spatial_affinity_path and attribute_affinity_path "
                "are required when dual_graph is enabled"
            )
        affinity_spatial = load_affinity_matrix(spatial_path)
        affinity_attribute = load_affinity_matrix(attribute_path)

    if device is None:
        device = _resolve_device(cfg)

    graph_module: Optional[Union[TrainableRegionGraph, TrainableDualRegionGraph]] = None
    graph_meta: Dict = {}
    ckpt_name = str(graph_cfg.get("graph_ckpt", GRAPH_CKPT_NAME))
    ckpt_path = None
    if output_dir is not None:
        ckpt_path = Path(output_dir) / "intermediate" / ckpt_name

    if graph_cfg.get("train", False):
        if logger is not None:
            graph_mode = "dual" if dual_graph else "single"
            logger.log_info(
                f"AE graph: training {graph_mode} region graph with "
                f"L_reg + lambda_pre * L_pre"
            )
        graph_module, history = train_region_graph(
            data_f,
            cond,
            affinity,
            cfg=graph_cfg,
            device=device,
        )
        graph_meta["graph_history"] = history
        graph_meta["dual_graph"] = dual_graph
        if ckpt_path is not None:
            save_graph_checkpoint(graph_module, ckpt_path, graph_cfg)
            graph_meta["graph_ckpt"] = str(ckpt_path)
            if logger is not None:
                logger.log_info(f"AE graph: saved checkpoint to {ckpt_path}")
    elif ckpt_path is not None and ckpt_path.is_file():
        graph_module = load_graph_checkpoint(ckpt_path, device=device, cfg=graph_cfg)
        graph_meta["graph_ckpt"] = str(ckpt_path)
        graph_meta["dual_graph"] = dual_graph
        if logger is not None:
            logger.log_info(f"AE graph: loaded checkpoint from {ckpt_path}")

    inject_scale = float(graph_cfg.get("inject_scale", 0.0))
    if inject_scale == 0.0:
        if logger is not None:
            logger.log_info("AE graph: inject_scale=0, encoder uses original data_f")
        return data_f, graph_meta or None, graph_module

    if graph_module is not None:
        data_f_enc, inj_meta = apply_graph_encoder_input_torch(
            graph_module,
            data_f,
            cond,
            affinity,
            inject_scale=inject_scale,
            max_hop=int(graph_cfg.get("max_hop", 2)),
            edge_eps=float(graph_cfg.get("edge_eps", 0.0)),
            eps=float(graph_cfg.get("eps", 1e-8)),
            device=device,
            dual_graph=dual_graph,
            affinity_spatial=affinity_spatial,
            affinity_attribute=affinity_attribute,
        )
    elif dual_graph:
        data_f_enc, inj_meta = apply_dual_graph_encoder_input(
            data_f,
            cond,
            affinity,
            affinity_spatial,
            affinity_attribute,
            inject_scale=inject_scale,
            max_hop=int(graph_cfg.get("max_hop", 2)),
            edge_eps=float(graph_cfg.get("edge_eps", 0.0)),
            eps=float(graph_cfg.get("eps", 1e-8)),
            gate_bias_init=float(graph_cfg.get("gate_bias_init", -10.0)),
            num_regions=int(affinity.shape[0]),
        )
    else:
        data_f_enc, inj_meta = apply_graph_encoder_input(
            data_f,
            cond,
            affinity,
            inject_scale=inject_scale,
            max_hop=int(graph_cfg.get("max_hop", 2)),
            edge_eps=float(graph_cfg.get("edge_eps", 0.0)),
            eps=float(graph_cfg.get("eps", 1e-8)),
            gate_bias_init=float(graph_cfg.get("gate_bias_init", -10.0)),
            num_regions=int(affinity.shape[0]),
        )

    graph_meta.update(inj_meta)
    if logger is not None:
        inj_norm = float(np.linalg.norm(graph_meta.get("injection", np.zeros(1))))
        logger.log_info(
            f"AE graph: inject_scale={inject_scale}, "
            f"||injection||_F={inj_norm:.6e}, trained={graph_module is not None}"
        )
    return data_f_enc, graph_meta, graph_module


def run_ae_phase(
    data_norm: np.ndarray,
    cond: np.ndarray,
    config: dict,
    device: Optional[torch.device] = None,
    logger=None,
    output_dir: Optional[Union[str, Path]] = None,
    baseline_norm: Optional[np.ndarray] = None,
) -> Tuple[ZeroAwareResidualAutoencoder, np.ndarray, dict]:
    """
    Train AE on normalized data and extract z_a.

    Returns model, z_a (N, D), and intermediate arrays dict.
    """
    cfg = build_ae_cfg(config)
    if device is not None:
        cfg["device"] = device

    if logger is not None:
        logger.log_info("ControlNet Phase AE: prepare filtered residual (norm space)")

    data_f, weight, mask_hard, _data_r, cond_group_stats = prepare_filtered_residual(
        data_norm, cfg, cond=cond, baseline=baseline_norm
    )
    cond_f = np.asarray(cond, dtype=np.float32)
    if cond_f.ndim == 1:
        cond_f = cond_f[:, None]

    if logger is not None:
        logger.log_info(
            f"AE inputs: data_f={data_f.shape}, weight mean={weight.mean():.6f}, "
            f"mask_hard rate={mask_hard.mean():.6f}"
        )

    data_f_enc, graph_meta, graph_module = maybe_apply_graph_encoder_input(
        data_f,
        cond,
        cfg,
        output_dir=output_dir,
        device=_resolve_device(cfg),
        logger=logger,
    )

    model, _history = train_zero_aware_autoencoder(
        data_f_enc, weight, cond_f, cfg, target_r=data_f
    )
    z_a = extract_anomaly_latent(
        model,
        data_f_enc,
        weight,
        cond_f,
        batch_size=int(cfg.get("extract_batch_size", 256)),
        device=_resolve_device(cfg),
    )

    intermediates = {
        "data_f": data_f,
        "weight": weight,
        "mask_hard": mask_hard,
        "cond": cond,
        "z_a": z_a,
    }
    if graph_meta is not None:
        intermediates["data_f_enc"] = data_f_enc
        intermediates["graph_meta"] = graph_meta
    if graph_module is not None:
        intermediates["graph_trained"] = True
    if cond_group_stats is not None:
        intermediates["cond_group_stats"] = cond_group_stats
    return model, z_a, intermediates
