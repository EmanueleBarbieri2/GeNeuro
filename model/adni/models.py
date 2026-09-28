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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.30,
        num_classes: int = 3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ADNIMissingModalityVAE(nn.Module):
    """Two-modality version of the repository's variational generator.

    The token order is ``[fMRI, DTI]``.  For DTI->fMRI reconstruction the
    fMRI token is masked, so the posterior and decoded fMRI representation
    are conditioned only on DTI.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_registers: int = 2,
        mlp_depth: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")

        def make_mlp():
            layers = []
            for _ in range(mlp_depth - 1):
                layers.extend(
                    [nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout)]
                )
            layers.append(nn.Linear(embed_dim, embed_dim))
            return nn.Sequential(*layers)

        self.embed_dim = embed_dim
        self.modality_tokens = nn.Parameter(torch.randn(2, 1, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.mu_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.sigma_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.register_tokens = nn.Parameter(
            torch.randn(num_registers, 1, embed_dim) * 0.02
        )
        self.modality_projectors = nn.ModuleList([make_mlp(), make_mlp()])
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)
        self.modality_decoders = nn.ModuleList([make_mlp(), make_mlp()])

    def encode(self, modality_embeddings, missing_mask):
        if modality_embeddings.ndim != 3 or modality_embeddings.size(1) != 2:
            raise ValueError("Expected modality embeddings with shape [batch, 2, embed_dim].")
        x = modality_embeddings + self.modality_tokens.transpose(0, 1)
        x = torch.stack(
            [projector(x[:, index]) for index, projector in enumerate(self.modality_projectors)],
            dim=1,
        )
        masked = self.mask_token.expand_as(x)
        x = torch.where(missing_mask.unsqueeze(-1), masked, x)
        batch_size = x.size(0)
        tokens = torch.cat(
            [
                self.mu_token.expand(batch_size, -1, -1),
                self.sigma_token.expand(batch_size, -1, -1),
                self.register_tokens.transpose(0, 1).expand(batch_size, -1, -1),
                x,
            ],
            dim=1,
        )
        encoded = self.transformer(tokens)
        mu = self.fc_mu(encoded[:, 0])
        logvar = self.fc_logvar(encoded[:, 1]).clamp(-10.0, 10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent):
        return torch.stack([decoder(latent) for decoder in self.modality_decoders], dim=1)

    def forward(self, modality_embeddings, missing_mask, sample: bool = True):
        mu, logvar = self.encode(modality_embeddings, missing_mask)
        latent = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(latent), mu, logvar

    def reconstruct_fmri_from_dti(self, dti_embedding, sample: bool = True):
        inputs = torch.zeros(
            dti_embedding.size(0), 2, self.embed_dim,
            dtype=dti_embedding.dtype,
            device=dti_embedding.device,
        )
        inputs[:, 1] = dti_embedding
        missing = torch.tensor([True, False], device=dti_embedding.device).expand(
            dti_embedding.size(0), -1
        )
        decoded, mu, logvar = self(inputs, missing, sample=sample)
        return decoded[:, 0], decoded[:, 1], mu, logvar


def symmetric_contrastive_loss(z_fmri, z_dti, temperature: float = 0.1):
    logits = z_fmri @ z_dti.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )
