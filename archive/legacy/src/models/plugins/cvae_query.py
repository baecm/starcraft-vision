# src/models/plugins/cvae_query.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAELatentQueryInjector(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) Latent Query Injector.
    Models 5-rater human gaze / GT noise consensus vs confusion in a continuous probabilistic latent space z.

    - Posterior q_\phi(z | F, H): Encodes spatio-temporal features F and 5-rater human heatmaps H.
    - Prior p_\theta(z | F): Encodes spatio-temporal features F (used during Zero-shot / Test inference).
    - Reparameterization z = \mu + \sigma \odot \epsilon
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_raters: int = 5,
        latent_dim: int = 64,
        embed_dim: int = 128,
        num_queries: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # Posterior Encoder q(z | F, H)
        self.posterior_net = nn.Sequential(
            nn.Conv2d(feat_dim + num_raters, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.post_mu = nn.Linear(128, latent_dim)
        self.post_logvar = nn.Linear(128, latent_dim)

        # Prior Encoder p(z | F)
        self.prior_net = nn.Sequential(
            nn.Conv2d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.prior_mu = nn.Linear(128, latent_dim)
        self.prior_logvar = nn.Linear(128, latent_dim)

        # Latent projection to query embeddings (B, K, embed_dim)
        self.latent_to_query = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_queries * embed_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        feat_map: torch.Tensor,
        human_heatmaps: Optional[torch.Tensor] = None,
        use_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        feat_map: (B, feat_dim, H_out, W_out)
        human_heatmaps: Optional (B, num_raters, H_out, W_out)

        Returns dict containing:
          - 'latent_query': (B, K, embed_dim)
          - 'z': (B, latent_dim)
          - 'mu': (B, latent_dim)
          - 'logvar': (B, latent_dim)
          - 'prior_mu': (B, latent_dim)
          - 'prior_logvar': (B, latent_dim)
        """
        B = feat_map.shape[0]

        # Prior computation p(z | F)
        p_emb = self.prior_net(feat_map)
        prior_mu = self.prior_mu(p_emb)
        prior_logvar = self.prior_logvar(p_emb)

        if use_posterior and human_heatmaps is not None:
            # Resize human heatmaps to match feature map resolution if needed
            if human_heatmaps.shape[-2:] != feat_map.shape[-2:]:
                human_heatmaps = F.interpolate(
                    human_heatmaps, size=feat_map.shape[-2:], mode="bilinear", align_corners=False
                )

            concat_in = torch.cat([feat_map, human_heatmaps], dim=1)
            post_emb = self.posterior_net(concat_in)
            mu = self.post_mu(post_emb)
            logvar = self.post_logvar(post_emb)
        else:
            mu, logvar = prior_mu, prior_logvar

        z = self.reparameterize(mu, logvar)
        query_flat = self.latent_to_query(z)
        latent_query = query_flat.view(B, self.num_queries, self.embed_dim)

        return {
            "latent_query": latent_query,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
        }
