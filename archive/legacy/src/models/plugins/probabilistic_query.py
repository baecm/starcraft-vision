# src/model/plugins/probabilistic_query.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

class ProbabilisticLatentQuery(nn.Module):
    """
    Models Object Queries as latent Gaussian distributions N(mu, sigma^2).
    Allows sampling latent positional/content embeddings during forward pass
    and computes KL-divergence loss for uncertainty modeling in dense scenes.
    """
    def __init__(self, num_queries: int = 100, d_model: int = 256, latent_dim: int = 64):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.log_var_head = nn.Linear(d_model, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, d_model)

    def forward(self, batch_size: int, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        base_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        mu = self.mu_head(base_embed)
        log_var = self.log_var_head(base_embed)

        if is_training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        latent_feature = self.latent_proj(z)
        queries = base_embed + latent_feature

        kl_loss = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        return queries, kl_loss
