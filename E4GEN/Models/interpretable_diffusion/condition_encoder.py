import torch
from torch import nn


class ConditionEncoder(nn.Module):
    """
    Map discrete condition indices (B, C1) to label_emb (B, d_model).
    Each condition channel has its own Embedding(K_j + 1, d_model); null index = K_j.
    """

    def __init__(self, num_classes_per_cond, d_model: int):
        super().__init__()
        self.num_classes_per_cond = [int(n) for n in num_classes_per_cond]
        self.cond_dim = len(self.num_classes_per_cond)
        self.null_ids = self.num_classes_per_cond
        self.embeds = nn.ModuleList([
            nn.Embedding(n_classes + 1, d_model)
            for n_classes in self.num_classes_per_cond
        ])

    def forward(self, cond: torch.Tensor, cond_dropout: float = 0.0) -> torch.Tensor:
        """
        cond: (B, C1) long tensor with values in [0, K_j - 1] per channel j.
        """
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"cond must be (B, {self.cond_dim}), got {tuple(cond.shape)}"
            )

        label_emb = None
        for j, embed in enumerate(self.embeds):
            cj = cond[:, j].long()
            if self.training and cond_dropout > 0:
                drop_mask = torch.rand(cj.shape[0], device=cj.device) < cond_dropout
                null_id = self.null_ids[j]
                cj = torch.where(drop_mask, torch.full_like(cj, null_id), cj)
            emb_j = embed(cj)
            label_emb = emb_j if label_emb is None else label_emb + emb_j
        return label_emb

    def null_label_emb(self, batch_size: int, device: torch.device) -> torch.Tensor:
        null_cond = torch.tensor(
            self.null_ids,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        return self.forward(null_cond, cond_dropout=0.0)
