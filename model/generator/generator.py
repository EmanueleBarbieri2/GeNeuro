# models_prom3e.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProM3E_Generator(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dim=512, num_heads=8, num_layers=3, num_registers=4):
        super().__init__()
        
        # 1. Modality Embeddings (to tell the Transformer "This vector is MRI")
        # 0: SPECT, 1: MRI, 2: fMRI, 3: DTI
        self.modality_tokens = nn.Parameter(torch.randn(4, 1, embed_dim))
        
        # 2. The Mask Token (learned placeholder for missing data)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. Mu/Sigma tokens and register tokens
        self.mu_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.sigma_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.randn(num_registers, 1, embed_dim))

        # 4. Modality projectors (2-layer MLPs)
        self.modality_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(4)
            ]
        )
        
        # 5. The Transformer Backbone (Inter-modality attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim*4, 
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6. Probabilistic Heads (Variational Information Bottleneck)
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_var = nn.Linear(embed_dim, embed_dim)

        # 7. Modality-specific decoders (2-layer MLPs)
        self.modality_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(4)
            ]
        )
        
    def reparameterize(self, mu, logvar):
        """Standard VAE sampling trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z_input, mask_indices):
        """
        z_input: - The input embeddings (with 0s or noise in missing slots)
        mask_indices: - Boolean tensor (True = Missing/Masked, False = Present)
        """
        B, S, D = z_input.shape
        
        # Add Modality Identity to the vectors so the model knows position
        x = z_input + self.modality_tokens.transpose(0, 1) # Broadcast to batch

        # Project per modality
        x_proj = []
        for i in range(4):
            x_proj.append(self.modality_projectors[i](x[:, i, :]))
        x = torch.stack(x_proj, dim=1)
        
        # Replace missing/masked modalities with the learnable MASK token
        # We expand mask token to match batch size
        mask_tokens_batch = self.mask_token.expand(B, S, D)
        
        # Where mask is True, use mask_token, else use input x
        x = torch.where(mask_indices.unsqueeze(-1), mask_tokens_batch, x)
        
        # Build transformer input with register, mu, sigma tokens
        mu_tok = self.mu_token.expand(B, 1, D)
        sigma_tok = self.sigma_token.expand(B, 1, D)
        reg_tok = self.register_tokens.transpose(0, 1).expand(B, -1, D)
        x_in = torch.cat([mu_tok, sigma_tok, reg_tok, x], dim=1)

        # Run Transformer
        x_out = self.transformer(x_in)
        
        # Predict distribution from mu/sigma tokens
        mu = self.fc_mu(x_out[:, 0:1, :])
        logvar = self.fc_var(x_out[:, 1:2, :])
        
        # Sample latent z (during inference you might just use mu)
        z_sampled = self.reparameterize(mu, logvar)  # [B,1,D]

        # Decode per modality
        z_rep = z_sampled.expand(B, 4, D)
        recon = []
        for i in range(4):
            recon.append(self.modality_decoders[i](z_rep[:, i, :]))
        z_recon = torch.stack(recon, dim=1)

        return z_recon, mu, logvar