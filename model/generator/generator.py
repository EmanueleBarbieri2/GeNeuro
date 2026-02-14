# generator.py (Sweep-Ready Version)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProM3E_Generator(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dim=512, num_heads=8, 
                 num_layers=3, num_registers=4, mlp_depth=2, dropout=0.1):
        super().__init__()
        
        # 1. Modality & Placeholder Tokens
        self.modality_tokens = nn.Parameter(torch.randn(4, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mu_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.sigma_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.randn(num_registers, 1, embed_dim))

        # 2. Sweepable Modality Projectors
        def make_mlp(in_d, out_d, depth):
            layers = []
            for _ in range(depth - 1):
                layers.extend([nn.Linear(in_d, out_d), nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(out_d, out_d))
            return nn.Sequential(*layers)

        self.modality_projectors = nn.ModuleList([make_mlp(embed_dim, embed_dim, mlp_depth) for _ in range(4)])
        
        # 3. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True, 
            norm_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Probabilistic Heads
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_var = nn.Linear(embed_dim, embed_dim)

        # 5. Sweepable Modality Decoders
        self.modality_decoders = nn.ModuleList([make_mlp(embed_dim, embed_dim, mlp_depth) for _ in range(4)])
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z_input, mask_indices):
        B, S, D = z_input.shape
        
        # Positional/Identity Encoding
        x = z_input + self.modality_tokens.transpose(0, 1)

        # Project per modality
        x_proj = [self.modality_projectors[i](x[:, i, :]) for i in range(4)]
        x = torch.stack(x_proj, dim=1)
        
        # Masking
        mask_tokens_batch = self.mask_token.expand(B, S, D)
        x = torch.where(mask_indices.unsqueeze(-1), mask_tokens_batch, x)
        
        # Concat Latent Tokens
        mu_tok = self.mu_token.expand(B, 1, D)
        sigma_tok = self.sigma_token.expand(B, 1, D)
        reg_tok = self.register_tokens.transpose(0, 1).expand(B, -1, D)
        x_in = torch.cat([mu_tok, sigma_tok, reg_tok, x], dim=1)

        x_out = self.transformer(x_in)
        
        # Latent Distribution
        mu = self.fc_mu(x_out[:, 0:1, :])
        logvar = self.fc_var(x_out[:, 1:2, :])
        
        # STABILITY: Guard against exploding KL divergence
        logvar = torch.clamp(logvar, min=-10, max=10) 
        
        z_sampled = self.reparameterize(mu, logvar)

        # Multi-modal Decoding
        z_rep = z_sampled.expand(B, 4, D)
        z_recon = torch.stack([self.modality_decoders[i](z_rep[:, i, :]) for i in range(4)], dim=1)

        return z_recon, mu, logvar