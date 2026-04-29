# src/linear_encoder.py
"""
Supervised linear encoder for PPF framework comparison.

LinearEncoder replaces the unsupervised PCA projection with a single affine
layer trained to minimize MSE against privileged clean states. This isolates
the effect of encoder nonlinearity: both LinearEncoder and PlainEncoder receive
the same privileged training signal, but LinearEncoder has no hidden layers,
making it the linear counterpart to PlainEncoder within the PPF framework.

Forward interface returns (z, None) to match DisentangledEncoder / PlainEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    """
    Supervised linear encoder: a single affine map from noisy observations to
    the clean state space.

    Unlike PCA (which finds principal components of the noisy distribution),
    LinearEncoder is directly supervised to map noisy observations toward clean
    states using the privileged training signal available in the PPF framework.

    latent_dim should be set to true_state_dim so the output space matches the
    clean state dimensionality, consistent with how PCA was configured.
    """

    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        self.projection = nn.Linear(state_dim, latent_dim, bias=True)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: Noisy observations [B, state_dim]
        Returns:
            (z, None): z has shape [B, latent_dim]; None for API compatibility
                       with DisentangledEncoder and PlainEncoder.
        """
        return self.projection(obs), None
