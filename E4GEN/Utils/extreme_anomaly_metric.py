"""Encoder–Decoder AE on gen; two reconstruction scores on ori.

Training uses **only valid sequence content** (per-sample length from ``extract_time``);
when all series share the same length (typical fixed-``T`` arrays), each batch is a
plain ``(B, T, D)`` tensor and loss is MSE over all steps and channels — no padding
loss and **no** extreme mask during training.

Evaluation on **original** data returns:

* **Overall**: MSE over all valid (time × channel).
* **Extreme**: MSE on ``value_dim`` only where ``ori[..., value_dim] > threshold``.

``extreme_reconstruction_score_metrics`` trains the same AE on **synthetic
windows that contain at least one extreme** on ``value_dim`` (full-window MSE,
like standard AE training). Evaluation runs the AE on **full** original series
and reports MSE **only** over timesteps in an expanded extreme-time mask.

Lower is better for both.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from Utils.metric_utils import extract_time


class _EncDecSeqAE(nn.Module):
    """BiGRU encoder -> init multi-layer GRU decoder with teacher forcing."""

    def __init__(
        self,
        dim: int,
        enc_hidden: int,
        dec_hidden: int,
        enc_layers: int = 2,
        dec_layers: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.dec_layers = dec_layers
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.encoder = nn.GRU(
            dim,
            enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=True,
        )
        enc_flat = enc_layers * 2 * enc_hidden
        self.h_proj = nn.Linear(enc_flat, dec_layers * dec_hidden)
        self.decoder = nn.GRU(
            dim,
            dec_hidden,
            num_layers=dec_layers,
            batch_first=True,
        )
        self.head = nn.Linear(dec_hidden, dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T_pad, D); lengths: (B,) int64, sorted descending for pack."""
        b, t, _ = x.shape
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=True
        )
        _, h_n = self.encoder(packed_x)
        h_flat = h_n.permute(1, 0, 2).contiguous().view(b, -1)
        dec_h0 = self.h_proj(h_flat).view(b, self.dec_layers, self.dec_hidden)
        dec_h0 = dec_h0.permute(1, 0, 2).contiguous()

        dec_in = torch.zeros(b, t, self.dim, device=x.device, dtype=x.dtype)
        dec_in[:, 1:, :] = x[:, :-1, :]
        packed_in = nn.utils.rnn.pack_padded_sequence(
            dec_in, lengths.detach().cpu(), batch_first=True, enforce_sorted=True
        )
        dec_out, _ = self.decoder(packed_in, dec_h0)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            dec_out, batch_first=True, total_length=t
        )
        return self.head(unpacked)


def _masked_mse_full(recon: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """MSE over valid timesteps, all channels."""
    b, t, d = recon.shape
    ar = torch.arange(t, device=recon.device).unsqueeze(0).expand(b, -1)
    mask = (ar < lengths.unsqueeze(1)).unsqueeze(-1).to(recon.dtype)
    se = (recon - target) ** 2
    denom = (mask.sum() * d).clamp(min=1.0)
    return (se * mask).sum() / denom


def _expand_extreme_time_mask(raw: np.ndarray, margin: int) -> np.ndarray:
    """Boolean length-L mask: True on ``raw`` and within ``±margin`` steps (inclusive)."""
    r = np.asarray(raw, dtype=bool)
    if margin <= 0:
        return r
    L = int(r.shape[0])
    if L == 0:
        return r
    k = 2 * int(margin) + 1
    v = np.convolve(r.astype(np.float32), np.ones(k, dtype=np.float32), mode="same")
    return v > 0.5


def _collect_windows_containing_extreme(
    data: np.ndarray,
    times: List[int],
    lookback: int,
    value_dim: int,
    extreme_thr: float,
) -> np.ndarray:
    """Stack length-``lookback`` windows that contain ≥1 timestep with ``[..., ch] > thr``."""
    _, _, n_dim = data.shape
    ch = value_dim if value_dim >= 0 else n_dim + value_dim
    if not (0 <= ch < n_dim):
        raise ValueError(f"value_dim resolves to channel {ch} out of range for D={n_dim}")

    lb = int(lookback)
    if lb < 1:
        raise ValueError("lookback must be >= 1")

    thr = float(extreme_thr)
    wins: list = []
    for n in range(data.shape[0]):
        L = int(times[n])
        if L < lb:
            continue
        for t0 in range(0, L - lb + 1):
            win = data[n, t0 : t0 + lb, :]
            if np.any(win[:, ch] > thr):
                wins.append(np.asarray(win, dtype=np.float32))

    if not wins:
        return np.zeros((0, lb, n_dim), dtype=np.float32)
    return np.stack(wins, axis=0)


def extreme_reconstruction_score_metrics(
    ori_data,
    generated_data,
    lookback: int,
    eval_extreme_threshold: float,
    iterations: int = 500,
    batch_size: int = 128,
    enc_hidden: Optional[int] = None,
    dec_hidden: Optional[int] = None,
    enc_layers: int = 2,
    dec_layers: int = 2,
    lr: float = 1e-3,
    value_dim: int = -1,
    seed: int = 0,
    device: Optional[str] = None,
    eval_region_margin: int = 0,
) -> Union[float, Literal["insufficient_extreme_samples"]]:
    """Extreme reconstruction: train like standard AE on windows; eval on expanded extreme bands.

    **Training**: only length-``lookback`` windows from **generated** data that contain at
    least one timestep with ``[..., value_dim] > eval_extreme_threshold``. Loss is
    full-window MSE over all timesteps and channels (same spirit as
    ``extreme_anomaly_score_metrics`` training).

    **Evaluation**: run the AE on each **full** original series; accumulate squared error
    only at timesteps in the **expanded extreme mask**: start from
    ``ori[..., value_dim] > threshold`` along time, then dilate by ``±eval_region_margin``.

    Returns ``\"insufficient_extreme_samples\"`` if no qualifying generated windows exist.
    Returns ``float('nan')`` if no timestep lies in any expanded extreme mask across ori.
    """
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    _, _, dim = ori_data.shape
    if generated_data.shape[-1] != dim or ori_data.shape[0] == 0:
        raise ValueError("ori_data and generated_data must share D and be non-empty")

    ch = value_dim if value_dim >= 0 else dim + value_dim
    if not (0 <= ch < dim):
        raise ValueError(f"value_dim resolves to channel {ch} out of range for D={dim}")

    margin = int(eval_region_margin)
    if margin < 0:
        raise ValueError("eval_region_margin must be >= 0")

    if enc_hidden is None:
        enc_hidden = max(128, 4 * dim)
    if dec_hidden is None:
        dec_hidden = max(128, 4 * dim)

    ori_time, _ = extract_time(ori_data)
    gen_time, _ = extract_time(generated_data)

    thr = float(eval_extreme_threshold)
    win_g = _collect_windows_containing_extreme(
        generated_data, gen_time, int(lookback), value_dim, thr
    )
    if win_g.shape[0] == 0:
        return "insufficient_extreme_samples"

    lb = int(lookback)
    model = _EncDecSeqAE(dim, enc_hidden, dec_hidden, enc_layers, dec_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    m = int(win_g.shape[0])
    iterations = int(iterations)
    batch_size = int(batch_size)
    from tqdm.auto import tqdm

    xg_cpu = torch.from_numpy(win_g)
    model.train()
    for _ in tqdm(range(iterations), desc="extreme_recon_ae_train", total=iterations):
        idx = np.random.choice(m, size=min(batch_size, m), replace=False)
        xb = xg_cpu[idx].to(dev)
        b = xb.shape[0]
        lt = torch.full((b,), lb, dtype=torch.int64, device=dev)
        opt.zero_grad()
        recon = model(xb, lt)
        loss = _masked_mse_full(recon, xb, lt)
        loss.backward()
        opt.step()

    model.eval()
    total_sse = 0.0
    total_cnt = 0

    with torch.no_grad():
        for i in range(len(ori_data)):
            L = int(ori_time[i])
            if L < 1:
                continue
            raw = ori_data[i, :L, ch] > thr
            band = _expand_extreme_time_mask(raw, margin)
            if not np.any(band):
                continue

            x1 = torch.from_numpy(ori_data[i : i + 1, :L]).to(dev)
            l1 = torch.tensor([L], dtype=torch.int64, device=dev)
            recon = model(x1, l1)
            diff = recon[0, :L] - x1[0, :L]
            se = diff.pow(2).cpu().numpy()
            msk = band[:, np.newaxis]
            total_sse += float(np.sum(se * msk))
            total_cnt += int(np.sum(band) * dim)

    if total_cnt <= 0:
        return float("nan")
    return float(total_sse / float(total_cnt))


def extreme_anomaly_score_metrics(
    ori_data,
    generated_data,
    threshold: float,
    iterations: int = 500,
    batch_size: int = 128,
    enc_hidden: Optional[int] = None,
    dec_hidden: Optional[int] = None,
    enc_layers: int = 2,
    dec_layers: int = 2,
    lr: float = 1e-3,
    value_dim: int = -1,
    seed: int = 0,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Return ``(overall_mse, extreme_channel_mse)``; extreme is ``nan`` if no extreme steps."""
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    _, _, dim = ori_data.shape
    ch = value_dim if value_dim >= 0 else dim + value_dim

    if enc_hidden is None:
        enc_hidden = max(128, 4 * dim)
    if dec_hidden is None:
        dec_hidden = max(128, 4 * dim)

    gen_time, _ = extract_time(generated_data)
    ori_time, _ = extract_time(ori_data)

    model = _EncDecSeqAE(dim, enc_hidden, dec_hidden, enc_layers, dec_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_gen = len(generated_data)
    from tqdm.auto import tqdm

    model.train()
    for _ in tqdm(range(int(iterations)), desc="extreme_ae_train", total=int(iterations)):
        idx = np.random.choice(n_gen, size=min(batch_size, n_gen), replace=False)
        xb = torch.from_numpy(generated_data[idx]).to(dev)
        b, t_b, _ = xb.shape
        lt = torch.full((b,), t_b, dtype=torch.int64, device=dev)
        opt.zero_grad()
        recon = model(xb, lt)
        loss = _masked_mse_full(recon, xb, lt)
        loss.backward()
        opt.step()

    model.eval()
    overall_sum = 0.0
    overall_cnt = 0
    ext_sum = 0.0
    ext_cnt = 0

    with torch.no_grad():
        for i in range(len(ori_data)):
            L = int(ori_time[i])
            if L < 1:
                continue
            x1 = torch.from_numpy(ori_data[i : i + 1, :L]).to(dev)
            l1 = torch.tensor([L], dtype=torch.int64, device=dev)
            recon = model(x1, l1)
            diff = recon[0, :L] - x1[0, :L]
            se = diff.pow(2)
            overall_sum += float(se.sum().cpu())
            overall_cnt += L * dim

            vals = ori_data[i, :L, ch]
            mask = vals > float(threshold)
            if np.any(mask):
                ext_sum += float(se[:, ch].cpu().numpy()[mask].sum())
                ext_cnt += int(np.sum(mask))

    overall_mse = overall_sum / overall_cnt if overall_cnt > 0 else float("nan")
    extreme_mse = ext_sum / ext_cnt if ext_cnt > 0 else float("nan")
    return float(overall_mse), float(extreme_mse)
