"""Models used by the ADNI classification-only pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation, GINEConv


class ConnectivityEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int = 90,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_norm = nn.LayerNorm(num_nodes)
        self.input_projection = nn.Linear(num_nodes, hidden_dim)
        self.roi_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.normal_(self.roi_embedding, std=0.02)

        self.edge_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, data):
        if data.x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} connectivity features, got {data.x.size(1)}.")
        x = self.input_projection(self.input_norm(data.x))
        roi_index = torch.arange(data.num_nodes, device=x.device) % self.num_nodes
        x = x + self.roi_embedding[roi_index]
        edge_attr = self.edge_encoder(data.edge_attr.reshape(-1, 1))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + conv(x, data.edge_index, edge_attr=edge_attr))
            x = F.gelu(x)
        pooled = self.pool(x, data.batch)
        return F.normalize(self.projection(pooled), dim=1)


class ADNIClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x):
        return self.net(x)


def symmetric_contrastive_loss(z_fmri, z_dti, temperature: float = 0.1):
    logits = z_fmri @ z_dti.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
