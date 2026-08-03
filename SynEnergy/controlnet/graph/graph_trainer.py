"""Trainable homogeneous region graph with L_reg + lambda_pre * L_pre."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from controlnet.graph.Graph_Aggregation import (
    RegionGatedFusion,
    RegionGraphAttentionScores,
    _cond_to_region_index,
    _sample_semantic_from_2d,
    _sample_semantic_to_2d,
    build_k_hop_adjacency,
    inject_graph_to_samples,
    load_affinity_matrix,
)


def _cond_to_torch_index(cond: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(cond, torch.Tensor):
        c = cond.reshape(-1).long()
    else:
        c = torch.as_tensor(_cond_to_region_index(cond), dtype=torch.long, device=device)
    return c.to(device)


def aggregate_region_mean_torch(
    sample_emb: torch.Tensor,
    cond: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    """Mean-pool sample embeddings within each region -> U (K, L)."""
    if sample_emb.ndim != 2:
        raise ValueError(f"sample_emb must be (N, L), got {tuple(sample_emb.shape)}")
    cond = cond.reshape(-1).long()
    if sample_emb.shape[0] != cond.shape[0]:
        raise ValueError("sample_emb N must match cond N")

    k = int(num_regions)
    n, embed_dim = sample_emb.shape
    u = torch.zeros(k, embed_dim, device=sample_emb.device, dtype=sample_emb.dtype)
    counts = torch.zeros(k, device=sample_emb.device, dtype=sample_emb.dtype)
    u.index_add_(0, cond, sample_emb)
    counts.index_add_(0, cond, torch.ones(n, device=sample_emb.device, dtype=sample_emb.dtype))
    nonempty = counts > 0
    u[nonempty] = u[nonempty] / counts[nonempty].unsqueeze(1)
    return u


def masked_row_softmax_torch(
    logits: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Row-wise softmax over valid neighbors; invalid edges get weight 0."""
    valid = adjacency > 0
    masked = logits.masked_fill(~valid, float("-inf"))
    row_max = torch.where(
        valid.any(dim=1, keepdim=True),
        masked.max(dim=1, keepdim=True).values,
        torch.zeros_like(masked[:, :1]),
    )
    exp_e = torch.exp(masked - row_max)
    exp_e = torch.where(valid, exp_e, torch.zeros_like(exp_e))
    denom = exp_e.sum(dim=1, keepdim=True).clamp(min=1e-12)
    alpha = exp_e / denom
    alpha = torch.where(valid.any(dim=1, keepdim=True), alpha, torch.zeros_like(alpha))
    return alpha


@dataclass
class GraphForwardOutput:
    u: torch.Tensor
    u_bar: torch.Tensor
    u_nbr: torch.Tensor
    gate: torch.Tensor
    alpha: torch.Tensor


class TrainableRegionGraph(nn.Module):
    """Homogeneous region graph: attention over spatial neighbors + gated fusion."""

    def __init__(
        self,
        embed_dim: int = 180,
        gate_bias_init: float = -10.0,
        init_identity: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.attn = RegionGraphAttentionScores(
            embed_dim=self.embed_dim,
            bias=False,
            init_identity=init_identity,
        )
        self.fusion = RegionGatedFusion(
            embed_dim=self.embed_dim,
            gate_bias_init=gate_bias_init,
        )

    def forward(
        self,
        u: torch.Tensor,
        adjacency: torch.Tensor,
        eps: float = 1e-8,
        mask_fill: float = -1e9,
    ) -> GraphForwardOutput:
        e, _h, _s = self.attn(u, adjacency, eps=eps, mask_fill=mask_fill)
        alpha = masked_row_softmax_torch(e, adjacency)
        u_nbr = alpha @ u
        u_bar, gate = self.fusion(u, u_nbr)
        return GraphForwardOutput(
            u=u,
            u_bar=u_bar,
            u_nbr=u_nbr,
            gate=gate,
            alpha=alpha,
        )


class RelationWeightFusion(nn.Module):
    """Learnable softmax weights to fuse multiple relation-specific U_bar branches."""

    def __init__(self, num_relations: int):
        super().__init__()
        if int(num_relations) < 1:
            raise ValueError("num_relations must be >= 1")
        self.num_relations = int(num_relations)
        self.logits = nn.Parameter(torch.zeros(self.num_relations))

    def forward(
        self, branches: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(branches) != self.num_relations:
            raise ValueError(
                f"expected {self.num_relations} branches, got {len(branches)}"
            )
        for i, branch in enumerate(branches):
            if branch.shape != branches[0].shape:
                raise ValueError(
                    f"branch {i} shape {tuple(branch.shape)} != "
                    f"{tuple(branches[0].shape)}"
                )
        weights = torch.softmax(self.logits, dim=0)
        stacked = torch.stack(list(branches), dim=0)
        fused = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
        return fused, weights


@dataclass
class DualGraphForwardOutput:
    u: torch.Tensor
    u_bar: torch.Tensor
    u_bar_m: torch.Tensor
    u_bar_m1: torch.Tensor
    u_bar_spatial: torch.Tensor
    u_bar_attribute: torch.Tensor
    fusion_weights_m1: torch.Tensor
    fusion_weights_final: torch.Tensor
    out_m: GraphForwardOutput
    out_spatial: GraphForwardOutput
    out_attribute: GraphForwardOutput


class TrainableDualRegionGraph(nn.Module):
    """
    Dual region graphs:
      - Graph M on distance affinity
      - Graph M1 on spatial + attribute affinities with relation-weight fusion
      - Final relation-weight fusion between M and M1
    """

    def __init__(
        self,
        embed_dim: int = 180,
        gate_bias_init: float = -10.0,
        init_identity: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        graph_kwargs = {
            "embed_dim": self.embed_dim,
            "gate_bias_init": gate_bias_init,
            "init_identity": init_identity,
        }
        self.graph_m = TrainableRegionGraph(**graph_kwargs)
        self.graph_spatial = TrainableRegionGraph(**graph_kwargs)
        self.graph_attribute = TrainableRegionGraph(**graph_kwargs)
        self.fusion_m1 = RelationWeightFusion(num_relations=2)
        self.fusion_final = RelationWeightFusion(num_relations=2)

    def forward(
        self,
        u: torch.Tensor,
        adjacency_m: torch.Tensor,
        adjacency_spatial: torch.Tensor,
        adjacency_attribute: torch.Tensor,
        eps: float = 1e-8,
        mask_fill: float = -1e9,
    ) -> DualGraphForwardOutput:
        out_m = self.graph_m(u, adjacency_m, eps=eps, mask_fill=mask_fill)
        out_spatial = self.graph_spatial(
            u, adjacency_spatial, eps=eps, mask_fill=mask_fill
        )
        out_attribute = self.graph_attribute(
            u, adjacency_attribute, eps=eps, mask_fill=mask_fill
        )
        u_bar_m1, w_m1 = self.fusion_m1([out_spatial.u_bar, out_attribute.u_bar])
        u_bar, w_final = self.fusion_final([out_m.u_bar, u_bar_m1])
        return DualGraphForwardOutput(
            u=u,
            u_bar=u_bar,
            u_bar_m=out_m.u_bar,
            u_bar_m1=u_bar_m1,
            u_bar_spatial=out_spatial.u_bar,
            u_bar_attribute=out_attribute.u_bar,
            fusion_weights_m1=w_m1,
            fusion_weights_final=w_final,
            out_m=out_m,
            out_spatial=out_spatial,
            out_attribute=out_attribute,
        )


def region_representation_loss(
    z: torch.Tensor,
    cond: torch.Tensor,
    u_bar: torch.Tensor,
) -> torch.Tensor:
    """
    L_reg: per-region mean ||z_{n,m} - u_bar_n||_2^2, then equal average over regions.

    Each sample is matched only to its own region's u_bar (no cross-region matching).
    """
    cond = cond.reshape(-1).long()
    if z.shape[0] != cond.shape[0]:
        raise ValueError("z and cond must have the same number of samples")
    if u_bar.ndim != 2:
        raise ValueError(f"u_bar must be (K, L), got {tuple(u_bar.shape)}")

    per_sample = ((z - u_bar[cond]) ** 2).sum(dim=1)
    k = u_bar.shape[0]
    sums = torch.zeros(k, device=z.device, dtype=z.dtype)
    counts = torch.zeros(k, device=z.device, dtype=z.dtype)
    sums.scatter_add_(0, cond, per_sample)
    counts.scatter_add_(0, cond, torch.ones_like(per_sample))
    valid = counts > 0
    if not bool(valid.any()):
        return per_sample.mean() * 0.0
    region_means = sums[valid] / counts[valid]
    return region_means.mean()


def preservation_loss(u: torch.Tensor, u_bar: torch.Tensor) -> torch.Tensor:
    """L_pre: (1/K) sum_n ||u_bar_n - u_n||_2^2."""
    if u.shape != u_bar.shape:
        raise ValueError(f"u shape {tuple(u.shape)} != u_bar {tuple(u_bar.shape)}")
    return ((u_bar - u) ** 2).sum(dim=1).mean()


def graph_training_loss(
    z: torch.Tensor,
    cond: torch.Tensor,
    u: torch.Tensor,
    u_bar: torch.Tensor,
    lambda_pre: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """L = L_reg + lambda_pre * L_pre."""
    l_reg = region_representation_loss(z, cond, u_bar)
    l_pre = preservation_loss(u, u_bar)
    loss = l_reg + float(lambda_pre) * l_pre
    return {
        "loss": loss,
        "loss_reg": l_reg,
        "loss_pre": l_pre,
    }


def forward_trainable_graph(
    module: TrainableRegionGraph,
    sample_emb: torch.Tensor,
    cond: torch.Tensor,
    adjacency: torch.Tensor,
    **kwargs,
) -> GraphForwardOutput:
    """Sample embeddings -> region mean U -> graph -> U_bar."""
    u = aggregate_region_mean_torch(sample_emb, cond, adjacency.shape[0])
    out = module(u, adjacency, **kwargs)
    return GraphForwardOutput(
        u=u,
        u_bar=out.u_bar,
        u_nbr=out.u_nbr,
        gate=out.gate,
        alpha=out.alpha,
    )


def forward_trainable_dual_graph(
    module: TrainableDualRegionGraph,
    sample_emb: torch.Tensor,
    cond: torch.Tensor,
    adjacency_m: torch.Tensor,
    adjacency_spatial: torch.Tensor,
    adjacency_attribute: torch.Tensor,
    **kwargs,
) -> DualGraphForwardOutput:
    """Sample embeddings -> region mean U -> dual graphs -> fused U_bar."""
    u = aggregate_region_mean_torch(sample_emb, cond, adjacency_m.shape[0])
    return module(
        u,
        adjacency_m,
        adjacency_spatial,
        adjacency_attribute,
        **kwargs,
    )


def _build_adjacency_torch(
    affinity: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    adjacency_np = build_k_hop_adjacency(
        np.asarray(affinity, dtype=np.float32),
        max_hop=int(cfg.get("max_hop", 2)),
        edge_eps=float(cfg.get("edge_eps", 0.0)),
    )
    return torch.as_tensor(adjacency_np, dtype=torch.float32, device=device)


def train_region_graph(
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    affinity: np.ndarray,
    cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TrainableRegionGraph, Dict[str, List[float]]]:
    """
    Train homogeneous region graph on per-sample embeddings z (e.g. data_f).

    Uses only L_reg + lambda_pre * L_pre (no global coverage loss).

    When cfg['dual_graph']=True, trains TrainableDualRegionGraph (M + M1).
    """
    cfg = dict(cfg or {})
    if cfg.get("dual_graph", False):
        return train_dual_region_graph(sample_semantic, cond, cfg=cfg, device=device)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_np, _ = _sample_semantic_to_2d(sample_semantic)
    m_np = np.asarray(affinity, dtype=np.float32)
    adjacency_np = build_k_hop_adjacency(
        m_np,
        max_hop=int(cfg.get("max_hop", 2)),
        edge_eps=float(cfg.get("edge_eps", 0.0)),
    )

    embed_dim = z_np.shape[1]
    model = TrainableRegionGraph(
        embed_dim=embed_dim,
        gate_bias_init=float(cfg.get("gate_bias_init", -10.0)),
        init_identity=bool(cfg.get("init_identity", True)),
    ).to(device)

    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    cond_t = _cond_to_torch_index(cond, device)
    adjacency = torch.as_tensor(adjacency_np, dtype=torch.float32, device=device)

    lr = float(cfg.get("graph_lr", 1e-3))
    num_epochs = int(cfg.get("graph_epochs", 50))
    lambda_pre = float(cfg.get("lambda_pre", 0.1))
    eps = float(cfg.get("eps", 1e-8))
    mask_fill = float(cfg.get("mask_fill", -1e9))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {
        "loss": [],
        "loss_reg": [],
        "loss_pre": [],
    }

    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        out = forward_trainable_graph(
            model, z, cond_t, adjacency, eps=eps, mask_fill=mask_fill
        )
        losses = graph_training_loss(z, cond_t, out.u, out.u_bar, lambda_pre=lambda_pre)
        losses["loss"].backward()
        optimizer.step()

        for key in history:
            history[key].append(float(losses[key].detach().cpu().item()))

    model.eval()
    return model, history


def train_dual_region_graph(
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[TrainableDualRegionGraph, Dict[str, List[float]]]:
    """Train dual graphs (distance M + spatial/attribute M1) with relation fusion."""
    cfg = dict(cfg or {})
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_np, _ = _sample_semantic_to_2d(sample_semantic)
    affinity_m = load_affinity_matrix(cfg["affinity_path"])
    affinity_spatial = load_affinity_matrix(cfg["spatial_affinity_path"])
    affinity_attribute = load_affinity_matrix(cfg["attribute_affinity_path"])

    embed_dim = z_np.shape[1]
    model = TrainableDualRegionGraph(
        embed_dim=embed_dim,
        gate_bias_init=float(cfg.get("gate_bias_init", -10.0)),
        init_identity=bool(cfg.get("init_identity", True)),
    ).to(device)

    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    cond_t = _cond_to_torch_index(cond, device)
    adj_m = _build_adjacency_torch(affinity_m, cfg, device)
    adj_spatial = _build_adjacency_torch(affinity_spatial, cfg, device)
    adj_attribute = _build_adjacency_torch(affinity_attribute, cfg, device)

    lr = float(cfg.get("graph_lr", 1e-3))
    num_epochs = int(cfg.get("graph_epochs", 50))
    lambda_pre = float(cfg.get("lambda_pre", 0.1))
    eps = float(cfg.get("eps", 1e-8))
    mask_fill = float(cfg.get("mask_fill", -1e9))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {
        "loss": [],
        "loss_reg": [],
        "loss_pre": [],
    }

    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        out = forward_trainable_dual_graph(
            model,
            z,
            cond_t,
            adj_m,
            adj_spatial,
            adj_attribute,
            eps=eps,
            mask_fill=mask_fill,
        )
        losses = graph_training_loss(z, cond_t, out.u, out.u_bar, lambda_pre=lambda_pre)
        losses["loss"].backward()
        optimizer.step()

        for key in history:
            history[key].append(float(losses[key].detach().cpu().item()))

    model.eval()
    return model, history


def apply_graph_encoder_input_torch(
    module: Union[TrainableRegionGraph, TrainableDualRegionGraph],
    sample_semantic: np.ndarray,
    cond: np.ndarray,
    affinity: np.ndarray,
    inject_scale: float = 0.0,
    max_hop: int = 2,
    edge_eps: float = 0.0,
    eps: float = 1e-8,
    mask_fill: float = -1e9,
    device: Optional[torch.device] = None,
    dual_graph: bool = False,
    affinity_spatial: Optional[np.ndarray] = None,
    affinity_attribute: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Run trained graph module and inject U_bar - U back to per-sample embeddings."""
    x2d, had_channel = _sample_semantic_to_2d(sample_semantic)
    if float(inject_scale) == 0.0:
        return _sample_semantic_from_2d(x2d, had_channel), {}

    if device is None:
        device = next(module.parameters()).device

    module = module.to(device)
    module.eval()
    with torch.no_grad():
        z = torch.as_tensor(x2d, dtype=torch.float32, device=device)
        cond_t = _cond_to_torch_index(cond, device)

        if dual_graph:
            if not isinstance(module, TrainableDualRegionGraph):
                raise TypeError("dual_graph=True requires TrainableDualRegionGraph")
            if affinity_spatial is None or affinity_attribute is None:
                raise ValueError(
                    "affinity_spatial and affinity_attribute are required for dual graph"
                )
            adj_m = _build_adjacency_torch(
                affinity,
                {"max_hop": max_hop, "edge_eps": edge_eps},
                device,
            )
            adj_spatial = _build_adjacency_torch(
                affinity_spatial,
                {"max_hop": max_hop, "edge_eps": edge_eps},
                device,
            )
            adj_attribute = _build_adjacency_torch(
                affinity_attribute,
                {"max_hop": max_hop, "edge_eps": edge_eps},
                device,
            )
            out = forward_trainable_dual_graph(
                module,
                z,
                cond_t,
                adj_m,
                adj_spatial,
                adj_attribute,
                eps=eps,
                mask_fill=mask_fill,
            )
            u = out.u.detach().cpu().numpy()
            u_bar = out.u_bar.detach().cpu().numpy()
            meta_extra = {
                "U_bar_m": out.u_bar_m.detach().cpu().numpy(),
                "U_bar_m1": out.u_bar_m1.detach().cpu().numpy(),
                "U_bar_spatial": out.u_bar_spatial.detach().cpu().numpy(),
                "U_bar_attribute": out.u_bar_attribute.detach().cpu().numpy(),
                "fusion_weights_m1": out.fusion_weights_m1.detach().cpu().numpy(),
                "fusion_weights_final": out.fusion_weights_final.detach().cpu().numpy(),
            }
        else:
            if not isinstance(module, TrainableRegionGraph):
                raise TypeError("single-graph mode requires TrainableRegionGraph")
            m_np = np.asarray(affinity, dtype=np.float32)
            adjacency_np = build_k_hop_adjacency(
                m_np, max_hop=max_hop, edge_eps=edge_eps
            )
            adjacency = torch.as_tensor(adjacency_np, dtype=torch.float32, device=device)
            out = forward_trainable_graph(
                module, z, cond_t, adjacency, eps=eps, mask_fill=mask_fill
            )
            u = out.u.detach().cpu().numpy()
            u_bar = out.u_bar.detach().cpu().numpy()
            meta_extra = {}

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
        "injection": injection,
        "inject_scale": float(inject_scale),
        **meta_extra,
    }
    return _sample_semantic_from_2d(x_prime, had_channel), meta


GRAPH_CKPT_NAME = "graph.pt"


def save_graph_checkpoint(
    module: Union[TrainableRegionGraph, TrainableDualRegionGraph],
    path: Union[str, Path],
    cfg: Optional[dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_type = "dual" if isinstance(module, TrainableDualRegionGraph) else "single"
    payload = {
        "graph_type": graph_type,
        "model_state_dict": module.state_dict(),
        "embed_dim": module.embed_dim,
        "config": dict(cfg or {}),
    }
    torch.save(payload, path)
    return path


def load_graph_checkpoint(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
    cfg: Optional[dict] = None,
) -> Union[TrainableRegionGraph, TrainableDualRegionGraph]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    merged_cfg = dict(payload.get("config", {}))
    if cfg:
        merged_cfg.update(cfg)

    embed_dim = int(payload.get("embed_dim", merged_cfg.get("embed_dim", 180)))
    graph_type = payload.get(
        "graph_type",
        "dual" if merged_cfg.get("dual_graph", False) else "single",
    )
    if graph_type == "dual":
        model: Union[TrainableRegionGraph, TrainableDualRegionGraph] = TrainableDualRegionGraph(
            embed_dim=embed_dim,
            gate_bias_init=float(merged_cfg.get("gate_bias_init", -10.0)),
            init_identity=bool(merged_cfg.get("init_identity", True)),
        )
    else:
        model = TrainableRegionGraph(
            embed_dim=embed_dim,
            gate_bias_init=float(merged_cfg.get("gate_bias_init", -10.0)),
            init_identity=bool(merged_cfg.get("init_identity", True)),
        )
    model.load_state_dict(payload["model_state_dict"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model
