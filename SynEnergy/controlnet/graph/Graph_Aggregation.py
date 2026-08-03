"""
Region-level graph pipeline for anomaly semantics.

Step 1: U (K, L) — mean of per-sample semantics within each region (no graph matrix).

Step 2: E (K, K) — from U (K, L), attention logits among regions within
    0-hop (self) + 1-hop + 2-hop on the graph induced by affinity M:

    A[r,q] = M[r,q] if q is within 2 hops of r, else 0
    H = U W_att^T,  S = H H^T,  E = S + log(A + eps) with A_{r,q}=0 masked.

Step 3: Alpha (K, K) — row-wise masked softmax over valid neighbors in A:

    alpha[n,q] = exp(e[n,q]) / sum_{q' in N_n} exp(e[n,q']),  q in N_n
    alpha[n,q] = 0,  A[n,q] = 0
    N_n = {q | A[n,q] > 0}

Step 4: U_nbr (K, L) — neighbor-aggregated region context:

    U_nbr = Alpha @ U

Step 5: U_bar (K, L) — gated residual fusion:

    G = sigmoid(W_gate [U; U_nbr]),  U_bar = U + G ⊙ U_nbr

Step 6: X' (N, L) — inject tiny graph signal back to per-sample embeddings:

    delta_r = U_bar[r] - U[r],  X'[i] = X[i] + inject_scale * delta_{cond[i]}
    inject_scale defaults to 0 (no effect on AE until > 0).

Dual-graph extension (M + M1):

    Graph M:  affinity_path (distance) -> U_bar_m
    Graph M1: spatial_matrix + attribute_matrix, each with steps 2-5 ->
              U_bar_spatial, U_bar_attribute
    Relation fusion (M1): U_bar_m1 = sum_j w_j * U_bar_j  (learnable w)
    Final fusion:         U_bar = sum_k w_k * [U_bar_m, U_bar_m1]
    Step 6 unchanged: inject U_bar - U back to samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from Data.build_dataloader import load_cond_npy


def load_sample_semantic(
    path: Union[str, Path],
    expected_n: Optional[int] = None,
    seq_length: Optional[int] = None,
) -> np.ndarray:
    """
    Load per-sample anomaly semantic along time.

    Accepts (N, L), (N, L, 1), or (N, 1, L); returns (N, L).
    Typical source: AE intermediate ``data_f.npy`` with L=180.
    """
    x = np.load(path, allow_pickle=False)
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3:
        if x.shape[-1] == 1:
            x = x[:, :, 0]
        elif x.shape[1] == 1:
            x = x[:, 0, :]
        else:
            raise ValueError(
                f"3D sample semantic must be (N, L, 1) or (N, 1, L); got {x.shape}"
            )
    if x.ndim != 2:
        raise ValueError(f"sample semantic must be (N, L), got shape={x.shape}")

    if expected_n is not None and x.shape[0] != expected_n:
        raise ValueError(f"sample rows {x.shape[0]} != expected N {expected_n}")
    if seq_length is not None and x.shape[1] != seq_length:
        raise ValueError(f"seq_len {x.shape[1]} != expected {seq_length}")
    return x


def load_affinity_matrix(path: Union[str, Path]) -> np.ndarray:
    """Load region affinity matrix M of shape (K, K)."""
    m = np.load(path, allow_pickle=False)
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"affinity must be square (K, K), got shape={m.shape}")
    return m


def load_region_semantic(
    path: Union[str, Path],
    num_regions: Optional[int] = None,
    seq_length: Optional[int] = None,
) -> np.ndarray:
    """Load region-level semantic U of shape (K, L), e.g. step-1 output."""
    u = np.load(path, allow_pickle=False)
    u = np.asarray(u, dtype=np.float32)
    if u.ndim != 2:
        raise ValueError(f"region semantic must be (K, L), got shape={u.shape}")
    if num_regions is not None and u.shape[0] != num_regions:
        raise ValueError(f"region count {u.shape[0]} != expected {num_regions}")
    if seq_length is not None and u.shape[1] != seq_length:
        raise ValueError(f"seq_len {u.shape[1]} != expected {seq_length}")
    return u


def load_adjacency_matrix(path: Union[str, Path]) -> np.ndarray:
    """Alias: load affinity M and use it directly as weighted adjacency A."""
    return load_affinity_matrix(path)


def _cond_to_region_index(cond: np.ndarray) -> np.ndarray:
    """Flatten cond to (N,) int64 region indices."""
    c = np.asarray(cond)
    if c.ndim == 3 and c.shape[-1] == 1:
        c = c[:, :, 0]
    if c.ndim == 2 and c.shape[1] == 1:
        c = c[:, 0]
    elif c.ndim != 1:
        raise ValueError(f"cond must be (N,), (N, 1), or (N, 1, 1); got {c.shape}")
    return c.astype(np.int64, copy=False)


def aggregate_region_mean(
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    num_regions: Optional[int] = None,
) -> np.ndarray:
    """
    Mean-pool per-sample semantics within each region (same cond).

    No graph matrix is used. Empty regions (no samples) stay zero.

    Args:
        sample_semantic: (N, L)
        cond: region index per sample
        num_regions: K; defaults to max(cond)+1

    Returns:
        u: (K, L) e.g. (56, 180)
    """
    x = np.asarray(sample_semantic, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"sample_semantic must be (N, L), got shape={x.shape}")

    region = _cond_to_region_index(cond)
    n, seq_len = x.shape
    if region.shape[0] != n:
        raise ValueError(f"cond N={region.shape[0]} must match sample N={n}")

    k = int(num_regions) if num_regions is not None else int(region.max()) + 1
    if region.min() < 0 or region.max() >= k:
        raise ValueError(
            f"cond values must be in [0, {k}), got [{region.min()}, {region.max()}]"
        )

    u = np.zeros((k, seq_len), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int64)
    np.add.at(u, region, x)
    np.add.at(counts, region, 1)

    nonempty = counts > 0
    u[nonempty] /= counts[nonempty, None]
    return u


def aggregate_region_graph_semantic(
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    affinity: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Region-graph weighted aggregation of per-sample anomaly semantics.

    For each region r and time t:
        Z[r, t] = sum_j M[r, cond[j]] * X[j, t] / (sum_j M[r, cond[j]] + eps)

    Args:
        sample_semantic: (N, L) per-sample anomaly semantic along time
        cond: (N,), (N, 1), or (N, 1, 1) region indices in [0, K-1]
        affinity: (K, K) tract affinity matrix M
        eps: denominator stabilizer

    Returns:
        region_semantic: (K, L) e.g. (56, 180) for Tallahassee
    """
    x = np.asarray(sample_semantic, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"sample_semantic must be (N, L), got shape={x.shape}")

    region = _cond_to_region_index(cond)
    n, seq_len = x.shape
    if region.shape[0] != n:
        raise ValueError(f"cond N={region.shape[0]} must match sample N={n}")

    m = np.asarray(affinity, dtype=np.float32)
    k = m.shape[0]
    if region.min() < 0 or region.max() >= k:
        raise ValueError(
            f"cond values must be in [0, {k}), got [{region.min()}, {region.max()}]"
        )

    # B[r, j] = M[r, region_j]  -> (K, N)
    region_weights = m[:, region]
    denom = region_weights.sum(axis=1, keepdims=True) + float(eps)
    region_semantic = (region_weights @ x) / denom
    return region_semantic.astype(np.float32, copy=False)


def build_k_hop_adjacency(
    affinity: np.ndarray,
    max_hop: int = 2,
    edge_eps: float = 0.0,
) -> np.ndarray:
    """
    Build weighted adjacency A from affinity M using 0..max_hop neighborhood.

    Hop graph: undirected edge r—q when M[r,q] > edge_eps (off-diagonal).
    0-hop always includes self. For max_hop=2: self + 1-hop + 2-hop neighbors.

    A[r,q] = M[r,q] if q is reachable within max_hop, else 0.
    """
    m = np.asarray(affinity, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"affinity must be (K, K), got shape={m.shape}")
    if max_hop < 0:
        raise ValueError(f"max_hop must be >= 0, got {max_hop}")

    k = m.shape[0]
    off_diag = m.copy()
    np.fill_diagonal(off_diag, 0.0)
    binary = (off_diag > float(edge_eps)).astype(np.float32)

    reach = np.eye(k, dtype=np.float32)
    frontier = np.eye(k, dtype=np.float32)
    for _ in range(max_hop):
        frontier = ((frontier @ binary) > 0).astype(np.float32)
        reach = np.clip(reach + frontier, 0.0, 1.0)

    np.fill_diagonal(reach, 1.0)
    adjacency = m * reach
    np.fill_diagonal(adjacency, m.diagonal())
    return adjacency.astype(np.float32, copy=False)


class RegionGraphAttentionScores(nn.Module):
    """
    Learnable region attention logits before softmax.

    U (K, L) -> H = U W_att^T -> S = H H^T -> E = S + log(A + eps), masked by A.
    A is typically the 0/1/2-hop masked affinity from build_k_hop_adjacency.
    """

    def __init__(
        self,
        embed_dim: int = 180,
        bias: bool = False,
        init_identity: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if init_identity:
            nn.init.eye_(self.proj.weight)
            if bias:
                nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        u: torch.Tensor,
        adjacency: torch.Tensor,
        eps: float = 1e-8,
        mask_fill: float = -1e9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            u: (K, L) region embeddings
            adjacency: (K, K) region affinity M used directly as weighted adjacency;
                M_{n,q}=0 means no edge

        Returns:
            e: (K, K) attention logits with no-edge positions masked
            h: (K, L) projected region embeddings
            s: (K, K) semantic similarity H H^T
        """
        if u.ndim != 2:
            raise ValueError(f"u must be (K, L), got {tuple(u.shape)}")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"adjacency must be (K, K), got {tuple(adjacency.shape)}")
        if u.shape[0] != adjacency.shape[0]:
            raise ValueError(
                f"region count mismatch: u K={u.shape[0]} vs adjacency K={adjacency.shape[0]}"
            )
        if u.shape[1] != self.embed_dim:
            raise ValueError(
                f"embed dim mismatch: u L={u.shape[1]} vs module dim={self.embed_dim}"
            )

        h = self.proj(u)
        s = h @ h.transpose(0, 1)
        log_a = torch.log(adjacency + float(eps))
        e = s + log_a
        edge_mask = adjacency <= 0
        e = e.masked_fill(edge_mask, float(mask_fill))
        return e, h, s


def compute_region_attention_logits(
    region_semantic: np.ndarray,
    affinity: np.ndarray,
    w_att: Optional[Union[np.ndarray, torch.Tensor]] = None,
    module: Optional[RegionGraphAttentionScores] = None,
    max_hop: int = 2,
    edge_eps: float = 0.0,
    eps: float = 1e-8,
    mask_fill: float = -1e9,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute E, H, S from U and affinity M.

    Builds 2-hop adjacency A from M, then E = S + log(A+eps) with masking.

    Returns:
        e, h, s, adjacency — shapes (K,K), (K,L), (K,K), (K,K)
    """
    u_np = np.asarray(region_semantic, dtype=np.float32)
    m_np = np.asarray(affinity, dtype=np.float32)
    if u_np.ndim != 2:
        raise ValueError(f"region_semantic must be (K, L), got {u_np.shape}")
    if m_np.ndim != 2 or m_np.shape[0] != m_np.shape[1]:
        raise ValueError(f"affinity must be (K, K), got {m_np.shape}")
    if u_np.shape[0] != m_np.shape[0]:
        raise ValueError("region_semantic K must match affinity K")

    adjacency = build_k_hop_adjacency(m_np, max_hop=max_hop, edge_eps=edge_eps)

    if device is None:
        device = torch.device("cpu")

    embed_dim = u_np.shape[1]
    if module is None:
        module = RegionGraphAttentionScores(
            embed_dim=embed_dim, bias=False, init_identity=True
        ).to(device)
        if w_att is not None:
            weight = torch.as_tensor(w_att, dtype=torch.float32, device=device)
            if weight.shape != module.proj.weight.shape:
                raise ValueError(
                    f"w_att shape {tuple(weight.shape)} != "
                    f"expected {tuple(module.proj.weight.shape)}"
                )
            module.proj.weight.data.copy_(weight)

    u = torch.as_tensor(u_np, dtype=torch.float32, device=device)
    adj = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    e, h, s = module(u, adj, eps=eps, mask_fill=mask_fill)
    return (
        e.detach().cpu().numpy().astype(np.float32),
        h.detach().cpu().numpy().astype(np.float32),
        s.detach().cpu().numpy().astype(np.float32),
        adjacency.astype(np.float32, copy=False),
    )


def run_step1(
    sample_semantic_path: Union[str, Path],
    cond_path: Union[str, Path],
    output_path: Union[str, Path],
    cond_dim: int = 1,
    num_classes_per_cond: Optional[Tuple[int, ...]] = (56,),
    seq_length: int = 180,
    num_regions: int = 56,
) -> np.ndarray:
    """Mean-pool per-sample semantics within each region -> U (K, L)."""
    sample_semantic = load_sample_semantic(
        sample_semantic_path, seq_length=seq_length
    )
    n = sample_semantic.shape[0]

    cond = load_cond_npy(
        str(cond_path),
        cond_dim=cond_dim,
        expected_n=n,
        num_classes_per_cond=list(num_classes_per_cond) if num_classes_per_cond else None,
    )

    region_semantic = aggregate_region_mean(
        sample_semantic, cond, num_regions=num_regions
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, region_semantic)
    return region_semantic


def run_step2(
    region_semantic_path: Union[str, Path],
    affinity_path: Union[str, Path],
    output_path: Union[str, Path],
    num_regions: int = 56,
    seq_length: int = 180,
    max_hop: int = 2,
    edge_eps: float = 0.0,
    eps: float = 1e-8,
    mask_fill: float = -1e9,
    save_h: bool = True,
    save_s: bool = True,
    save_adjacency: bool = True,
    w_att_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attention logits E (K,K) from U using 0/1/2-hop masked affinity A."""
    u = load_region_semantic(
        region_semantic_path,
        num_regions=num_regions,
        seq_length=seq_length,
    )
    affinity = load_affinity_matrix(affinity_path)

    w_att = None
    if w_att_path is not None:
        w_att = np.load(w_att_path, allow_pickle=False)

    e, h, s, adjacency = compute_region_attention_logits(
        u,
        affinity,
        w_att=w_att,
        max_hop=max_hop,
        edge_eps=edge_eps,
        eps=eps,
        mask_fill=mask_fill,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, e)
    if save_h:
        np.save(out.with_name(out.stem + "_H.npy"), h)
    if save_s:
        np.save(out.with_name(out.stem + "_S.npy"), s)
    if save_adjacency:
        np.save(out.with_name(out.stem + "_A2hop.npy"), adjacency)
    return e, h, s, adjacency


def masked_row_softmax(
    logits: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    """
    Row-wise softmax over valid neighbors only.

    For each target region n, normalize over N_n = {q | A[n,q] > 0}.
    Invalid positions (A[n,q] == 0) get alpha[n,q] = 0.

    Args:
        logits: (K, K) attention scores E
        adjacency: (K, K) masked 2-hop adjacency A from step 2

    Returns:
        alpha: (K, K) attention weights; each row sums to 1 if N_n is non-empty
    """
    e = np.asarray(logits, dtype=np.float64)
    adj = np.asarray(adjacency, dtype=np.float32)
    if e.ndim != 2 or e.shape[0] != e.shape[1]:
        raise ValueError(f"logits must be (K, K), got shape={e.shape}")
    if adj.shape != e.shape:
        raise ValueError(f"adjacency shape {adj.shape} must match logits {e.shape}")

    valid = adj > 0
    masked = np.where(valid, e, -np.inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_e = np.exp(masked - row_max)
    exp_e = np.where(valid, exp_e, 0.0)
    denom = exp_e.sum(axis=1, keepdims=True)
    row_has_neighbor = valid.any(axis=1, keepdims=True)
    denom = np.where(row_has_neighbor, np.maximum(denom, 1e-12), 1.0)
    alpha = exp_e / denom
    alpha = np.where(row_has_neighbor, alpha, 0.0)
    return alpha.astype(np.float32, copy=False)


def run_step3(
    logits_path: Union[str, Path],
    adjacency_path: Union[str, Path],
    output_path: Union[str, Path],
    num_regions: int = 56,
) -> np.ndarray:
    """
    Masked row-wise softmax: E (K,K) + A (K,K) -> Alpha (K,K).

    Does not update embeddings; only produces neighbor aggregation weights.
    """
    e = np.load(logits_path, allow_pickle=False)
    e = np.asarray(e, dtype=np.float32)
    adjacency = load_adjacency_matrix(adjacency_path)

    if e.ndim != 2 or e.shape[0] != e.shape[1]:
        raise ValueError(f"logits must be (K, K), got shape={e.shape}")
    if num_regions is not None and e.shape[0] != num_regions:
        raise ValueError(f"logits K={e.shape[0]} != expected {num_regions}")
    if adjacency.shape != e.shape:
        raise ValueError(f"adjacency shape {adjacency.shape} != logits {e.shape}")

    alpha = masked_row_softmax(e, adjacency)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, alpha)
    return alpha


def aggregate_neighbor_embeddings(
    region_semantic: np.ndarray,
    attention_weights: np.ndarray,
) -> np.ndarray:
    """
    Weighted neighbor aggregation: U_nbr = Alpha @ U.

    For each region n:
        u_nbr[n] = sum_q alpha[n,q] * u[q]

    Only neighbors with alpha[n,q] > 0 contribute (masked in step 3).

    Args:
        region_semantic: (K, L) region embeddings U
        attention_weights: (K, K) attention weights Alpha from step 3

    Returns:
        u_nbr: (K, L) neighbor-aggregated region context
    """
    u = np.asarray(region_semantic, dtype=np.float32)
    alpha = np.asarray(attention_weights, dtype=np.float32)
    if u.ndim != 2:
        raise ValueError(f"region_semantic must be (K, L), got shape={u.shape}")
    if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
        raise ValueError(f"attention_weights must be (K, K), got shape={alpha.shape}")
    if alpha.shape[0] != u.shape[0]:
        raise ValueError(
            f"K mismatch: attention {alpha.shape[0]} vs region_semantic {u.shape[0]}"
        )

    u_nbr = alpha @ u
    return u_nbr.astype(np.float32, copy=False)


def run_step4(
    region_semantic_path: Union[str, Path],
    attention_path: Union[str, Path],
    output_path: Union[str, Path],
    num_regions: int = 56,
    seq_length: int = 180,
) -> np.ndarray:
    """
    Neighbor aggregation: Alpha (K,K) + U (K,L) -> U_nbr (K,L).

    Saves U_nbr; does not fuse with U yet (that is the next step).
    """
    u = load_region_semantic(
        region_semantic_path,
        num_regions=num_regions,
        seq_length=seq_length,
    )
    alpha = np.load(attention_path, allow_pickle=False)
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
        raise ValueError(f"attention must be (K, K), got shape={alpha.shape}")
    if alpha.shape[0] != u.shape[0]:
        raise ValueError(f"attention K={alpha.shape[0]} != U K={u.shape[0]}")

    u_nbr = aggregate_neighbor_embeddings(u, alpha)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, u_nbr)
    return u_nbr


class RegionGatedFusion(nn.Module):
    """
    Gated residual fusion: U_bar = U + sigmoid(W_gate [U; U_nbr]) ⊙ U_nbr.

    Default init: W_gate=0, bias=-10 -> G≈0 -> U_bar≈U (no neighbor injection).
    """

    def __init__(self, embed_dim: int = 180, gate_bias_init: float = -10.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.gate = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

    def forward(
        self, u: torch.Tensor, u_nbr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if u.shape != u_nbr.shape:
            raise ValueError(f"u shape {tuple(u.shape)} != u_nbr {tuple(u_nbr.shape)}")
        if u.shape[-1] != self.embed_dim:
            raise ValueError(
                f"embed dim {u.shape[-1]} != module dim {self.embed_dim}"
            )
        x = torch.cat([u, u_nbr], dim=-1)
        g = torch.sigmoid(self.gate(x))
        u_bar = u + g * u_nbr
        return u_bar, g


def fuse_gated_residual(
    region_semantic: np.ndarray,
    neighbor_semantic: np.ndarray,
    module: Optional[RegionGatedFusion] = None,
    gate_bias_init: float = -10.0,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gated residual fusion: U_bar = U + G ⊙ U_nbr.

    Returns:
        u_bar: (K, L)
        gate: (K, L) gate values G in [0, 1]
    """
    u_np = np.asarray(region_semantic, dtype=np.float32)
    nbr_np = np.asarray(neighbor_semantic, dtype=np.float32)
    if u_np.shape != nbr_np.shape:
        raise ValueError(f"U shape {u_np.shape} != U_nbr shape {nbr_np.shape}")
    if u_np.ndim != 2:
        raise ValueError(f"inputs must be (K, L), got U={u_np.shape}")

    if device is None:
        device = torch.device("cpu")

    embed_dim = u_np.shape[1]
    if module is None:
        module = RegionGatedFusion(
            embed_dim=embed_dim, gate_bias_init=gate_bias_init
        ).to(device)

    u = torch.as_tensor(u_np, dtype=torch.float32, device=device)
    u_nbr = torch.as_tensor(nbr_np, dtype=torch.float32, device=device)
    u_bar, g = module(u, u_nbr)
    return (
        u_bar.detach().cpu().numpy().astype(np.float32),
        g.detach().cpu().numpy().astype(np.float32),
    )


def run_step5(
    region_semantic_path: Union[str, Path],
    neighbor_semantic_path: Union[str, Path],
    output_path: Union[str, Path],
    num_regions: int = 56,
    seq_length: int = 180,
    gate_bias_init: float = -10.0,
    save_gate: bool = True,
    gate_ckpt_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gated fusion: U + G⊙U_nbr -> U_bar (K, L).

    Optionally saves gate matrix G and W_gate checkpoint for training.
    """
    u = load_region_semantic(
        region_semantic_path,
        num_regions=num_regions,
        seq_length=seq_length,
    )
    u_nbr = load_region_semantic(
        neighbor_semantic_path,
        num_regions=num_regions,
        seq_length=seq_length,
    )

    module = RegionGatedFusion(embed_dim=seq_length, gate_bias_init=gate_bias_init)
    if gate_ckpt_path is not None and Path(gate_ckpt_path).is_file():
        state = torch.load(gate_ckpt_path, map_location="cpu", weights_only=True)
        module.load_state_dict(state)

    u_bar, gate = fuse_gated_residual(u, u_nbr, module=module)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, u_bar)
    if save_gate:
        np.save(out.with_name(out.stem + "_gate_G.npy"), gate)

    ckpt_out = out.with_name(out.stem + "_W_gate.pt")
    torch.save(module.state_dict(), ckpt_out)

    return u_bar, gate


def map_region_to_samples(
    region_embedding: np.ndarray,
    cond: np.ndarray,
) -> np.ndarray:
    """Broadcast region rows to samples: out[i] = region_embedding[cond[i]]."""
    region = _cond_to_region_index(cond)
    u = np.asarray(region_embedding, dtype=np.float32)
    if u.ndim != 2:
        raise ValueError(f"region_embedding must be (K, L), got {u.shape}")
    if region.min() < 0 or region.max() >= u.shape[0]:
        raise ValueError(
            f"cond values must be in [0, {u.shape[0]}), "
            f"got [{region.min()}, {region.max()}]"
        )
    return u[region].astype(np.float32, copy=False)


def _sample_semantic_to_2d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return (N, L) and whether the input used a trailing channel dim (N, L, 1)."""
    arr = np.asarray(x, dtype=np.float32)
    had_channel = False
    if arr.ndim == 3:
        had_channel = True
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :]
        else:
            raise ValueError(
                f"3D sample semantic must be (N, L, 1) or (N, 1, L); got {arr.shape}"
            )
    if arr.ndim != 2:
        raise ValueError(f"sample semantic must be (N, L), got shape={arr.shape}")
    return arr, had_channel


def _sample_semantic_from_2d(x: np.ndarray, had_channel: bool) -> np.ndarray:
    if had_channel:
        return x[:, :, None].astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


def apply_graph_encoder_input(
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    affinity: np.ndarray,
    inject_scale: float = 0.0,
    max_hop: int = 2,
    edge_eps: float = 0.0,
    eps: float = 1e-8,
    mask_fill: float = -1e9,
    gate_bias_init: float = -10.0,
    num_regions: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Run graph steps 1–6 inline and return X' for the AE encoder.

    When inject_scale=0, returns the input unchanged (no graph computation).
  """
    x2d, had_channel = _sample_semantic_to_2d(sample_semantic)
    if float(inject_scale) == 0.0:
        return _sample_semantic_from_2d(x2d, had_channel), {}

    m = np.asarray(affinity, dtype=np.float32)
    k = int(num_regions) if num_regions is not None else m.shape[0]

    u = aggregate_region_mean(x2d, cond, num_regions=k)
    e, _h, _s, adjacency = compute_region_attention_logits(
        u,
        m,
        max_hop=max_hop,
        edge_eps=edge_eps,
        eps=eps,
        mask_fill=mask_fill,
    )
    alpha = masked_row_softmax(e, adjacency)
    u_nbr = aggregate_neighbor_embeddings(u, alpha)
    u_bar, gate = fuse_gated_residual(
        u, u_nbr, gate_bias_init=gate_bias_init
    )
    x_prime, injection = inject_graph_to_samples(
        x2d,
        u_bar,
        cond,
        region_baseline=u,
        inject_scale=inject_scale,
        use_delta=True,
    )

    meta = {
        "U": u,
        "U_nbr": u_nbr,
        "U_bar": u_bar,
        "gate": gate,
        "injection": injection,
        "inject_scale": float(inject_scale),
    }
    return _sample_semantic_from_2d(x_prime, had_channel), meta


def fuse_relation_weighted(
    branches: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relation-weight fusion over multiple U_bar branches (numpy).

    Args:
        branches: list of (K, L) region embeddings, one per relation graph.
        weights: optional non-negative weights; normalized to sum to 1.

    Returns:
        fused: (K, L)
        w: (R,) normalized weights used
    """
    if not branches:
        raise ValueError("branches must be non-empty")
    arrays = [np.asarray(b, dtype=np.float32) for b in branches]
    k, seq_len = arrays[0].shape
    for i, arr in enumerate(arrays):
        if arr.shape != (k, seq_len):
            raise ValueError(
                f"branch {i} shape {arr.shape} != expected {(k, seq_len)}"
            )

    if weights is None:
        w = np.ones(len(arrays), dtype=np.float32) / float(len(arrays))
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != len(arrays):
            raise ValueError(f"weights len {w.shape[0]} != branches {len(arrays)}")
        w = np.maximum(w, 0.0)
        total = float(w.sum())
        if total <= 0:
            w = np.ones(len(arrays), dtype=np.float32) / float(len(arrays))
        else:
            w = w / total

    fused = np.zeros((k, seq_len), dtype=np.float32)
    for coeff, arr in zip(w, arrays):
        fused += coeff * arr
    return fused, w


def _run_single_graph_branch_numpy(
    u: np.ndarray,
    affinity: np.ndarray,
    max_hop: int = 2,
    edge_eps: float = 0.0,
    eps: float = 1e-8,
    mask_fill: float = -1e9,
    gate_bias_init: float = -10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps 2-5 on one affinity matrix; returns (u_nbr, u_bar, gate)."""
    e, _h, _s, adjacency = compute_region_attention_logits(
        u,
        affinity,
        max_hop=max_hop,
        edge_eps=edge_eps,
        eps=eps,
        mask_fill=mask_fill,
    )
    alpha = masked_row_softmax(e, adjacency)
    u_nbr = aggregate_neighbor_embeddings(u, alpha)
    u_bar, gate = fuse_gated_residual(u, u_nbr, gate_bias_init=gate_bias_init)
    return u_nbr, u_bar, gate


def apply_dual_graph_encoder_input(
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    affinity_m: np.ndarray,
    affinity_spatial: np.ndarray,
    affinity_attribute: np.ndarray,
    inject_scale: float = 0.0,
    max_hop: int = 2,
    edge_eps: float = 0.0,
    eps: float = 1e-8,
    mask_fill: float = -1e9,
    gate_bias_init: float = -10.0,
    num_regions: Optional[int] = None,
    fusion_weights_m1: Optional[Sequence[float]] = None,
    fusion_weights_final: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Dual-graph pipeline: distance M + M1 (spatial & attribute with relation fusion).

    When inject_scale=0, returns input unchanged.
    """
    x2d, had_channel = _sample_semantic_to_2d(sample_semantic)
    if float(inject_scale) == 0.0:
        return _sample_semantic_from_2d(x2d, had_channel), {}

    k = int(num_regions) if num_regions is not None else affinity_m.shape[0]
    u = aggregate_region_mean(x2d, cond, num_regions=k)

    _u_nbr_m, u_bar_m, gate_m = _run_single_graph_branch_numpy(
        u,
        affinity_m,
        max_hop=max_hop,
        edge_eps=edge_eps,
        eps=eps,
        mask_fill=mask_fill,
        gate_bias_init=gate_bias_init,
    )
    _u_nbr_s, u_bar_spatial, gate_spatial = _run_single_graph_branch_numpy(
        u,
        affinity_spatial,
        max_hop=max_hop,
        edge_eps=edge_eps,
        eps=eps,
        mask_fill=mask_fill,
        gate_bias_init=gate_bias_init,
    )
    _u_nbr_a, u_bar_attribute, gate_attribute = _run_single_graph_branch_numpy(
        u,
        affinity_attribute,
        max_hop=max_hop,
        edge_eps=edge_eps,
        eps=eps,
        mask_fill=mask_fill,
        gate_bias_init=gate_bias_init,
    )

    u_bar_m1, w_m1 = fuse_relation_weighted(
        [u_bar_spatial, u_bar_attribute],
        weights=fusion_weights_m1,
    )
    u_bar, w_final = fuse_relation_weighted(
        [u_bar_m, u_bar_m1],
        weights=fusion_weights_final,
    )

    x_prime, injection = inject_graph_to_samples(
        x2d,
        u_bar,
        cond,
        region_baseline=u,
        inject_scale=inject_scale,
        use_delta=True,
    )

    meta = {
        "U": u,
        "U_bar": u_bar,
        "U_bar_m": u_bar_m,
        "U_bar_m1": u_bar_m1,
        "U_bar_spatial": u_bar_spatial,
        "U_bar_attribute": u_bar_attribute,
        "gate_m": gate_m,
        "gate_spatial": gate_spatial,
        "gate_attribute": gate_attribute,
        "fusion_weights_m1": w_m1,
        "fusion_weights_final": w_final,
        "injection": injection,
        "inject_scale": float(inject_scale),
    }
    return _sample_semantic_from_2d(x_prime, had_channel), meta


def inject_graph_to_samples(
    sample_semantic: np.ndarray,
    region_enhanced: np.ndarray,
    cond: np.ndarray,
    region_baseline: Optional[np.ndarray] = None,
    inject_scale: float = 0.0,
    use_delta: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add a tiny graph signal from region space back to per-sample embeddings.

    Default (use_delta=True):
        X'[i] = X[i] + inject_scale * (U_bar[cond[i]] - U[cond[i]])

    When inject_scale=0 or U_bar==U, output equals input (no AE impact).

    Args:
        sample_semantic: (N, L) original per-sample embedding X
        region_enhanced: (K, L) graph-enhanced U_bar
        cond: (N,) / (N,1) region index per sample
        region_baseline: (K, L) step-1 region mean U; required if use_delta=True
        inject_scale: scalar; keep very small (e.g. 1e-4) or 0 for no-op
        use_delta: if False, inject U_bar directly instead of (U_bar - U)

    Returns:
        sample_out: (N, L)
        injection: (N, L) the term actually added (inject_scale * delta or u_bar)
    """
    x = np.asarray(sample_semantic, dtype=np.float32)
    u_bar = np.asarray(region_enhanced, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"sample_semantic must be (N, L), got {x.shape}")
    if u_bar.ndim != 2:
        raise ValueError(f"region_enhanced must be (K, L), got {u_bar.shape}")

    region = _cond_to_region_index(cond)
    if region.shape[0] != x.shape[0]:
        raise ValueError(f"cond N={region.shape[0]} must match sample N={x.shape[0]}")
    if x.shape[1] != u_bar.shape[1]:
        raise ValueError(
            f"L mismatch: sample {x.shape[1]} vs region_enhanced {u_bar.shape[1]}"
        )

    mapped = map_region_to_samples(u_bar, region)
    if use_delta:
        if region_baseline is None:
            raise ValueError("region_baseline U is required when use_delta=True")
        u_base = np.asarray(region_baseline, dtype=np.float32)
        if u_base.shape != u_bar.shape:
            raise ValueError(f"region_baseline shape {u_base.shape} != U_bar {u_bar.shape}")
        mapped_base = map_region_to_samples(u_base, region)
        term = mapped - mapped_base
    else:
        term = mapped

    injection = float(inject_scale) * term
    sample_out = x + injection
    return sample_out.astype(np.float32, copy=False), injection.astype(np.float32, copy=False)


def run_step6(
    sample_semantic_path: Union[str, Path],
    region_enhanced_path: Union[str, Path],
    cond_path: Union[str, Path],
    output_path: Union[str, Path],
    region_baseline_path: Optional[Union[str, Path]] = None,
    inject_scale: float = 0.0,
    use_delta: bool = True,
    cond_dim: int = 1,
    num_classes_per_cond: Optional[Tuple[int, ...]] = (56,),
    seq_length: int = 180,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map graph-enhanced region embedding back to N samples with tiny injection.

    Default inject_scale=0 -> identical to input (zero impact on AE phase-1).
    """
    x = load_sample_semantic(sample_semantic_path, seq_length=seq_length)
    n = x.shape[0]
    cond = load_cond_npy(
        str(cond_path),
        cond_dim=cond_dim,
        expected_n=n,
        num_classes_per_cond=list(num_classes_per_cond) if num_classes_per_cond else None,
    )
    u_bar = load_region_semantic(region_enhanced_path, seq_length=seq_length)

    u_base = None
    if use_delta:
        if region_baseline_path is None:
            raise ValueError("region_baseline_path (step-1 U) required when use_delta=True")
        u_base = load_region_semantic(region_baseline_path, seq_length=seq_length)

    sample_out, injection = inject_graph_to_samples(
        x,
        u_bar,
        cond,
        region_baseline=u_base,
        inject_scale=inject_scale,
        use_delta=use_delta,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, sample_out)
    np.save(out.with_name(out.stem + "_injection.npy"), injection)
    return sample_out, injection


def _add_step1_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sample_semantic_path",
        type=str,
        required=True,
        help="Per-sample anomaly semantic, e.g. AE intermediate data_f.npy (N, 180, 1)",
    )
    p.add_argument(
        "--cond_path",
        type=str,
        default="./Data/Power/Tallahassee_seq_180_sample_10000_condition.npy",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="./OUTPUT/graph_agg/region_semantic_step1.npy",
    )
    p.add_argument("--cond_dim", type=int, default=1)
    p.add_argument("--num_classes", type=int, default=56)
    p.add_argument("--seq_length", type=int, default=180)


def _add_step2_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--region_semantic_path",
        type=str,
        required=True,
        help="Region semantic U from step 1, shape (K, L)",
    )
    p.add_argument(
        "--affinity_path",
        type=str,
        default="./Data/Power/Tallahassee_tract_distance_matrix_56.npy",
        help="Affinity M; step2 keeps 0/1/2-hop edges as weighted adjacency A",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="./OUTPUT/graph_agg/region_attention_logits_E.npy",
    )
    p.add_argument("--num_regions", type=int, default=56)
    p.add_argument("--seq_length", type=int, default=180)
    p.add_argument("--max_hop", type=int, default=2, help="0-hop self + hops up to 2")
    p.add_argument("--edge_eps", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--mask_fill", type=float, default=-1e9)
    p.add_argument("--w_att_path", type=str, default=None)


def _add_step3_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--logits_path",
        type=str,
        required=True,
        help="Attention logits E from step 2, shape (K, K)",
    )
    p.add_argument(
        "--adjacency_path",
        type=str,
        required=True,
        help="2-hop adjacency A from step 2, e.g. region_attention_logits_E_A2hop.npy",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="./OUTPUT/graph_agg/region_attention_alpha.npy",
        help="Save attention weights Alpha (K, K)",
    )
    p.add_argument("--num_regions", type=int, default=56)


def _add_step4_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--region_semantic_path",
        type=str,
        required=True,
        help="Region embedding U from step 1, shape (K, L)",
    )
    p.add_argument(
        "--attention_path",
        type=str,
        required=True,
        help="Attention weights Alpha from step 3, shape (K, K)",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="./OUTPUT/graph_agg/region_neighbor_semantic_U_nbr.npy",
        help="Save neighbor-aggregated U_nbr (K, L)",
    )
    p.add_argument("--num_regions", type=int, default=56)
    p.add_argument("--seq_length", type=int, default=180)


def _add_step5_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--region_semantic_path",
        type=str,
        required=True,
        help="Region embedding U from step 1, shape (K, L)",
    )
    p.add_argument(
        "--neighbor_semantic_path",
        type=str,
        required=True,
        help="Neighbor context U_nbr from step 4, shape (K, L)",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="./OUTPUT/graph_agg/region_enhanced_U_bar.npy",
        help="Save graph-enhanced region embedding U_bar (K, L)",
    )
    p.add_argument("--num_regions", type=int, default=56)
    p.add_argument("--seq_length", type=int, default=180)
    p.add_argument(
        "--gate_bias_init",
        type=float,
        default=-10.0,
        help="Initial gate bias; -10 keeps G≈0 so U_bar≈U at start",
    )
    p.add_argument(
        "--gate_ckpt_path",
        type=str,
        default=None,
        help="Optional trained W_gate checkpoint to load",
    )


def _add_step6_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sample_semantic_path",
        type=str,
        required=True,
        help="Original per-sample embedding X (N, L), e.g. AE data_f.npy",
    )
    p.add_argument(
        "--region_enhanced_path",
        type=str,
        required=True,
        help="Graph-enhanced region embedding U_bar from step 5 (K, L)",
    )
    p.add_argument(
        "--region_baseline_path",
        type=str,
        default=None,
        help="Step-1 region mean U (K, L); required for delta injection (default)",
    )
    p.add_argument(
        "--cond_path",
        type=str,
        default="./Data/Power/Tallahassee_seq_180_sample_10000_condition.npy",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="./OUTPUT/graph_agg/sample_semantic_graph_enhanced.npy",
        help="Enhanced per-sample embedding X' (N, L)",
    )
    p.add_argument(
        "--inject_scale",
        type=float,
        default=0.0,
        help="Tiny scale for graph injection; 0 = no change to AE inputs",
    )
    p.add_argument(
        "--inject_full_u_bar",
        action="store_true",
        help="If set, inject U_bar instead of (U_bar - U) delta",
    )
    p.add_argument("--cond_dim", type=int, default=1)
    p.add_argument("--num_classes", type=int, default=56)
    p.add_argument("--seq_length", type=int, default=180)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Region graph pipeline: step1..step6."
    )
    sub = parser.add_subparsers(dest="step", required=True)

    p1 = sub.add_parser("step1", help="(N, L) -> U (K, L)")
    _add_step1_args(p1)

    p2 = sub.add_parser("step2", help="U (K, L) -> E (K, K)")
    _add_step2_args(p2)

    p3 = sub.add_parser("step3", help="E (K, K) -> Alpha (K, K)")
    _add_step3_args(p3)

    p4 = sub.add_parser("step4", help="Alpha + U -> U_nbr (K, L)")
    _add_step4_args(p4)

    p5 = sub.add_parser("step5", help="U + G⊙U_nbr -> U_bar (K, L)")
    _add_step5_args(p5)

    p6 = sub.add_parser("step6", help="U_bar -> X' (N, L) tiny sample injection")
    _add_step6_args(p6)

    args = parser.parse_args()
    if args.step == "step1":
        region_semantic = run_step1(
            sample_semantic_path=args.sample_semantic_path,
            cond_path=args.cond_path,
            output_path=args.output_path,
            cond_dim=args.cond_dim,
            num_classes_per_cond=(args.num_classes,),
            seq_length=args.seq_length,
            num_regions=args.num_classes,
        )
        print(f"U shape = {region_semantic.shape}")
        print(f"saved: {Path(args.output_path).resolve()}")
        return

    if args.step == "step3":
        alpha = run_step3(
            logits_path=args.logits_path,
            adjacency_path=args.adjacency_path,
            output_path=args.output_path,
            num_regions=args.num_regions,
        )
        row_sums = alpha.sum(axis=1)
        print(f"Alpha shape = {alpha.shape}")
        print(
            f"row sum: min={row_sums.min():.6f}, max={row_sums.max():.6f}, "
            f"mean={row_sums.mean():.6f}"
        )
        print(f"saved: {Path(args.output_path).resolve()}")
        return

    if args.step == "step4":
        u_nbr = run_step4(
            region_semantic_path=args.region_semantic_path,
            attention_path=args.attention_path,
            output_path=args.output_path,
            num_regions=args.num_regions,
            seq_length=args.seq_length,
        )
        print(f"U_nbr shape = {u_nbr.shape}")
        print(f"saved: {Path(args.output_path).resolve()}")
        return

    if args.step == "step5":
        u_bar, gate = run_step5(
            region_semantic_path=args.region_semantic_path,
            neighbor_semantic_path=args.neighbor_semantic_path,
            output_path=args.output_path,
            num_regions=args.num_regions,
            seq_length=args.seq_length,
            gate_bias_init=args.gate_bias_init,
            gate_ckpt_path=args.gate_ckpt_path,
        )
        print(f"U_bar shape = {u_bar.shape}, G shape = {gate.shape}")
        print(f"G range: min={gate.min():.6f}, max={gate.max():.6f}, mean={gate.mean():.6f}")
        print(f"saved: {Path(args.output_path).resolve()}")
        return

    if args.step == "step6":
        sample_out, injection = run_step6(
            sample_semantic_path=args.sample_semantic_path,
            region_enhanced_path=args.region_enhanced_path,
            cond_path=args.cond_path,
            output_path=args.output_path,
            region_baseline_path=args.region_baseline_path,
            inject_scale=args.inject_scale,
            use_delta=not args.inject_full_u_bar,
            cond_dim=args.cond_dim,
            num_classes_per_cond=(args.num_classes,),
            seq_length=args.seq_length,
        )
        inj_norm = float(np.linalg.norm(injection) / max(injection.size, 1) ** 0.5)
        print(f"X' shape = {sample_out.shape}, injection shape = {injection.shape}")
        print(f"inject_scale = {args.inject_scale}, injection RMS = {inj_norm:.6e}")
        print(f"saved: {Path(args.output_path).resolve()}")
        return

    e, h, s, adjacency = run_step2(
        region_semantic_path=args.region_semantic_path,
        affinity_path=args.affinity_path,
        output_path=args.output_path,
        num_regions=args.num_regions,
        seq_length=args.seq_length,
        max_hop=args.max_hop,
        edge_eps=args.edge_eps,
        eps=args.eps,
        mask_fill=args.mask_fill,
        w_att_path=args.w_att_path,
    )
    print(
        f"E shape = {e.shape}, H shape = {h.shape}, "
        f"S shape = {s.shape}, A_2hop shape = {adjacency.shape}"
    )
    print(f"saved: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
