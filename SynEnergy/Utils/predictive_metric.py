"""Per-dimension multi-horizon predictive score (PyTorch).

For each channel ``d`` and horizon ``h`` in ``horizons`` (default 1..5), train a **Transformer
encoder** on **generated** data: from ``x[t-lookback:t, d]`` predict ``x[t:t+h, d]`` via a
linear readout on the last time step.

**Insufficient history:** series with ``L < lookback + h`` contribute no windows for that ``h``.

Also exports ``_ScalarAutoregressiveForecaster`` (Transformer context + residual step MLP
rollout) for extreme-segment metrics.

The returned value is the **mean** of test errors over all (dimension, horizon) pairs,
unless ``eval_extreme_threshold`` is set and **no** synthetic extreme windows exist — then
the literal string ``insufficient_extreme_samples`` is returned.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.metric_utils import extract_time


def _transformer_encode(encoder: nn.Module, h: torch.Tensor) -> torch.Tensor:
    """Run TransformerEncoder; avoid Flash/mem-efficient SDPA that can break on some GPUs."""
    if h.device.type != "cuda":
        return encoder(h)
    try:
        cm = torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    except Exception:
        return encoder(h)
    with cm:
        return encoder(h)


def _d_model_nhead_dimff(hidden_dim: int) -> Tuple[int, int, int]:
    raw = max(32, int(hidden_dim))
    d_model = (raw + 3) // 4 * 4
    if d_model % 2 != 0:
        d_model += 2
    nhead = 4
    if d_model < 8:
        d_model, nhead = 8, 2
    dim_ff = max(d_model * 4, 128)
    return d_model, nhead, dim_ff


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class _TransformerContextEncoder(nn.Module):
    """Project univariate window to d_model + sinusoidal PE + TransformerEncoder."""

    def __init__(
        self,
        lookback: int,
        d_model: int,
        nhead: int,
        dim_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.d_model = int(d_model)
        self.in_proj = nn.Linear(1, d_model)
        self.register_buffer("pos_encoding", _sinusoidal_pe(lookback, d_model))
        # norm_first=False: fewer nested-tensor / SDPA edge cases on some CUDA stacks.
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_ff,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1), L == lookback
        h = self.in_proj(x) * math.sqrt(self.d_model)
        pe = self.pos_encoding[:, : x.size(1), :].to(device=h.device, dtype=h.dtype)
        h = h + pe
        return _transformer_encode(self.encoder, h)


class _TransformerMultiStepForecaster(nn.Module):
    """Transformer encoder on context -> linear head predicts ``horizon`` future steps."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        d_model, nhead, dim_ff = _d_model_nhead_dimff(hidden_dim)
        self.encoder = _TransformerContextEncoder(
            lookback, d_model, nhead, dim_ff, num_layers, dropout
        )
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h[:, -1, :])


class _ScalarAutoregressiveForecaster(nn.Module):
    """Transformer context encoding + residual MLP rollout (one scalar per step).

    Training: pass ``y_future`` (B, H) for teacher forcing on inputs to the step MLP.
    Evaluation: omit ``y_future`` to roll out from predictions.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        d_model, nhead, dim_ff = _d_model_nhead_dimff(hidden_dim)
        self.encoder = _TransformerContextEncoder(
            lookback, d_model, nhead, dim_ff, num_layers, dropout
        )
        self.step_in = nn.Linear(d_model + 1, d_model)
        self.step_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, y_future: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h_seq = self.encoder(x)
        s = h_seq[:, -1, :]
        inp = x[:, -1, :]
        preds: list[torch.Tensor] = []
        for step in range(self.horizon):
            u = torch.cat([s, inp], dim=-1)
            delta = self.step_out(self.drop(F.gelu(self.step_in(u))))
            s = self.norm(s + delta)
            p = self.head(s).squeeze(-1)
            preds.append(p)
            if y_future is not None:
                inp = y_future[:, step : step + 1]
            else:
                inp = p.unsqueeze(-1)
        return torch.stack(preds, dim=1)


def _collect_pairs_scalar_dim(
    data: np.ndarray,
    times: List[int],
    lookback: int,
    horizon: int,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding (ctx, tgt) for one channel. Drops series shorter than ``lookback + horizon``."""
    ctx_list: list = []
    tgt_list: list = []
    _, _, n_dim = data.shape
    if not (0 <= dim < n_dim):
        raise ValueError(f"dim {dim} out of range for D={n_dim}")

    min_len = lookback + horizon
    for n in range(data.shape[0]):
        L = int(times[n])
        if L < min_len:
            continue
        x = data[n, :L, dim]
        for t in range(lookback, L - horizon + 1):
            ctx_list.append(x[t - lookback : t])
            tgt_list.append(x[t : t + horizon])

    if not ctx_list:
        zc = np.zeros((0, lookback), dtype=np.float32)
        zt = np.zeros((0, horizon), dtype=np.float32)
        return zc, zt
    return np.stack(ctx_list, axis=0), np.stack(tgt_list, axis=0)


def _collect_pairs_scalar_dim_with_extreme_mask(
    data: np.ndarray,
    times: List[int],
    lookback: int,
    horizon: int,
    predict_dim: int,
    value_dim: int,
    extreme_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same sliding windows as ``_collect_pairs_scalar_dim``, plus boolean extreme mask.

    For each window with first target index ``t``, the mask is
    ``data[n, t, ch] > extreme_thr`` where ``ch`` resolves ``value_dim``
    (``-1`` is last channel). Caller applies tail sign flip so extreme is always ``>``.
    """
    ctx_list: list = []
    tgt_list: list = []
    mask_list: list = []
    _, _, n_dim = data.shape
    if not (0 <= predict_dim < n_dim):
        raise ValueError(f"predict_dim {predict_dim} out of range for D={n_dim}")
    ch = value_dim if value_dim >= 0 else n_dim + value_dim
    if not (0 <= ch < n_dim):
        raise ValueError(f"value_dim resolves to channel {ch} out of range for D={n_dim}")

    min_len = lookback + horizon
    thr = float(extreme_thr)
    for n in range(data.shape[0]):
        L = int(times[n])
        if L < min_len:
            continue
        x_d = data[n, :L, predict_dim]
        series = data[n, :L]
        for t in range(lookback, L - horizon + 1):
            ctx_list.append(x_d[t - lookback : t])
            tgt_list.append(x_d[t : t + horizon])
            mask_list.append(bool(series[t, ch] > thr))

    if not ctx_list:
        zc = np.zeros((0, lookback), dtype=np.float32)
        zt = np.zeros((0, horizon), dtype=np.float32)
        zm = np.zeros((0,), dtype=bool)
        return zc, zt, zm
    return (
        np.stack(ctx_list, axis=0).astype(np.float32),
        np.stack(tgt_list, axis=0).astype(np.float32),
        np.asarray(mask_list, dtype=bool),
    )


def predictive_score_metrics(
    ori_data,
    generated_data,
    iterations: int = 500,
    batch_size: int = 128,
    lookback: int = 100,
    horizons: Sequence[int] = (1, 2, 3, 4, 5),
    loss: str = "mae",
    hidden_dim: Optional[int] = None,
    seed: int = 0,
    device: Optional[str] = None,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    eval_batch_size: Optional[int] = None,
    eval_extreme_threshold: Optional[float] = None,
    value_dim: int = -1,
) -> Union[float, Literal["insufficient_extreme_samples"]]:
    """Train one Transformer forecaster per (dimension, horizon) on gen; **mean** error on ori.

    Parameters
    ----------
    transformer_layers : int
        Number of ``TransformerEncoderLayer`` blocks (default 2).
    transformer_dropout : float
        Dropout inside encoder layers and (for AR model) step MLP.
    eval_batch_size : int, optional
        Max batch size when evaluating on real data. Default: same as ``batch_size``.
        A full-graph forward on all origin windows scales Transformer memory as
        O(batch * lookback^2); keep this small if you hit CUDA OOM at eval time.
    eval_extreme_threshold : float, optional
        If set, **training uses only** synthetic windows whose first target time ``t``
        has ``generated_data[n, t, value_dim] > threshold`` (same rule as eval; caller
        tail flip). **Evaluation on ori** uses the same window filter. Requires
        ``horizons=(1,)``. If no such gen windows exist, returns
        ``\"insufficient_extreme_samples\"``.
    value_dim : int, default=-1
        Channel for the extreme label (``-1`` = last dimension).
    """
    if loss not in ("mse", "mae"):
        raise ValueError('loss must be "mse" or "mae"')

    horizons = tuple(int(h) for h in horizons)
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must be non-empty and each horizon >= 1")

    if eval_extreme_threshold is not None and tuple(horizons) != (1,):
        raise ValueError(
            "eval_extreme_threshold requires horizons=(1,): eval averages only one-step "
            "windows whose first target time is extreme on value_dim."
        )

    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)
    _, _, dim = ori_data.shape

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if hidden_dim is None:
        hidden_dim = max(16, 2 * dim)

    ori_time, _ = extract_time(ori_data)
    gen_time, _ = extract_time(generated_data)

    iterations = int(iterations)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    eval_bs = int(eval_batch_size) if eval_batch_size is not None else batch_size
    if eval_bs < 1:
        raise ValueError("eval_batch_size must be >= 1")
    lookback = int(lookback)
    tl = int(transformer_layers)
    td = float(transformer_dropout)

    from tqdm.auto import tqdm

    pair_scores: list[float] = []
    total_steps = dim * len(horizons) * iterations

    with tqdm(total=total_steps, desc="predictive_train", leave=True) as pbar:
        for d in range(dim):
            for h in horizons:
                if eval_extreme_threshold is not None:
                    ctx_g, tgt_g, ext_m_g = _collect_pairs_scalar_dim_with_extreme_mask(
                        generated_data,
                        gen_time,
                        lookback,
                        h,
                        d,
                        value_dim,
                        eval_extreme_threshold,
                    )
                    if ctx_g.shape[0] == 0:
                        return "insufficient_extreme_samples"
                    ctx_g = ctx_g[ext_m_g]
                    tgt_g = tgt_g[ext_m_g]
                    if ctx_g.shape[0] == 0:
                        return "insufficient_extreme_samples"

                    ctx_o, tgt_o, ext_m = _collect_pairs_scalar_dim_with_extreme_mask(
                        ori_data,
                        ori_time,
                        lookback,
                        h,
                        d,
                        value_dim,
                        eval_extreme_threshold,
                    )
                    if ctx_o.shape[0] == 0:
                        return float("nan")
                    ctx_o = ctx_o[ext_m]
                    tgt_o = tgt_o[ext_m]
                else:
                    ctx_g, tgt_g = _collect_pairs_scalar_dim(
                        generated_data, gen_time, lookback, h, d
                    )
                    ctx_o, tgt_o = _collect_pairs_scalar_dim(
                        ori_data, ori_time, lookback, h, d
                    )
                    if ctx_g.shape[0] == 0:
                        return float("nan")
                    if ctx_o.shape[0] == 0:
                        return float("nan")

                model = _TransformerMultiStepForecaster(
                    lookback, h, hidden_dim, num_layers=tl, dropout=td
                ).to(dev)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)

                xg_cpu = torch.from_numpy(ctx_g).unsqueeze(-1)
                yg_cpu = torch.from_numpy(tgt_g)
                m = ctx_g.shape[0]

                model.train()
                for _ in range(iterations):
                    idx = np.random.choice(m, size=min(batch_size, m), replace=False)
                    c = xg_cpu[idx].to(dev)
                    y = yg_cpu[idx].to(dev)
                    opt.zero_grad()
                    pred = model(c)
                    if loss == "mse":
                        loss_t = torch.mean((pred - y) ** 2)
                    else:
                        loss_t = torch.mean(torch.abs(pred - y))
                    loss_t.backward()
                    opt.step()
                    pbar.update(1)

                if ctx_o.shape[0] == 0:
                    pair_scores.append(float("nan"))
                    continue

                model.eval()
                mo = int(ctx_o.shape[0])
                xo_cpu = torch.from_numpy(ctx_o).unsqueeze(-1)
                yo_cpu = torch.from_numpy(tgt_o)
                with torch.no_grad():
                    if loss == "mse":
                        sse = torch.zeros((), device=dev, dtype=torch.float32)
                        for start in range(0, mo, eval_bs):
                            end = min(start + eval_bs, mo)
                            c = xo_cpu[start:end].to(dev)
                            y = yo_cpu[start:end].to(dev)
                            pred = model(c)
                            sse = sse + torch.sum((pred - y) ** 2)
                        err = sse / float(mo * h)
                    else:
                        sabs = torch.zeros((), device=dev, dtype=torch.float32)
                        for start in range(0, mo, eval_bs):
                            end = min(start + eval_bs, mo)
                            c = xo_cpu[start:end].to(dev)
                            y = yo_cpu[start:end].to(dev)
                            pred = model(c)
                            sabs = sabs + torch.sum(torch.abs(pred - y))
                        err = sabs / float(mo * h)
                pair_scores.append(float(err.item()))

    return float(np.nanmean(pair_scores))
