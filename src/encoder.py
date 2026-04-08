# src/encoder.py
"""
Encoder modules for robust offline RL experiments.

This module provides two encoder variants:

- DisentangledEncoder:
    Returns (z_task, z_irrel) and includes auxiliary heads for
    next-state prediction and reward prediction.

- PlainEncoder:
    Baseline encoder that returns (z, None) to keep the same interface
    as DisentangledEncoder. It uses the same capacity scaling rule and
    auxiliary heads for fair comparison.

Both encoders use an auto-scaling hidden dimension based on the input
state dimension to keep model capacity reasonably aligned across
different noise settings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DisentangledEncoder(nn.Module):
    """
    Dual-branch encoder that separates task-relevant and nuisance features.

    Outputs:
        z_task: latent representation intended to capture task-relevant information
        z_irrel: latent representation intended to capture nuisance information

    Auxiliary heads:
        state_predictor: predicts the next clean state from (z_task, action)
        reward_predictor: predicts scalar reward from z_task
    """

    def __init__(self, state_dim, action_dim, true_state_dim, latent_dim, aux_target_dim=None):
        super().__init__()

        # aux_target_dim controls the output size of state_predictor.
        # Defaults to true_state_dim (privileged clean state).
        # Pass aux_target_dim=state_dim for the no-privilege ablation,
        # where the target is the noisy next observation instead.
        _aux_target_dim = aux_target_dim if aux_target_dim is not None else true_state_dim

        # Scale encoder width with input dimensionality.
        hidden_dim = int(max(256, state_dim * 4))
        print(f"Auto-scaling Encoder: State Dim={state_dim} -> Hidden Dim={hidden_dim}")

        # Task-relevant representation branch.
        self.task_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
        )

        # Nuisance / irrelevant representation branch.
        self.irrel_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
        )

        # Scale auxiliary predictor width with latent dimensionality.
        pred_hidden = int(max(256, latent_dim * 4))

        # Predict the next state from task representation and action.
        # Output dimension is _aux_target_dim: true_state_dim (default) or state_dim (no-priv).
        self.state_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, _aux_target_dim),
        )

        # Predict scalar reward from the task representation.
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, 1),
        )

    def forward(self, obs):
        """
        Encode observations into task and nuisance latents.

        Args:
            obs: Input observations of shape [batch_size, state_dim]

        Returns:
            (z_task, z_irrel)
        """
        z_task = self.task_encoder(obs)
        z_irrel = self.irrel_encoder(obs)
        return z_task, z_irrel


class PlainEncoder(nn.Module):
    """
    Single-branch baseline encoder.

    This encoder is designed to be capacity-matched with
    DisentangledEncoder for fair comparison. It returns (z, None)
    so downstream code can use the same interface for both encoders.

    Auxiliary heads:
        state_predictor: predicts the next clean state from (z, action)
        reward_predictor: predicts scalar reward from z
    """

    def __init__(self, state_dim, action_dim, true_state_dim, latent_dim, aux_target_dim=None):
        super().__init__()

        # aux_target_dim controls the output size of state_predictor.
        # Defaults to true_state_dim (privileged clean state).
        # Pass aux_target_dim=state_dim for the no-privilege ablation,
        # where the target is the noisy next observation instead.
        _aux_target_dim = aux_target_dim if aux_target_dim is not None else true_state_dim

        # Use the same width scaling rule as the disentangled encoder.
        hidden_dim = int(max(256, state_dim * 4))
        print(f"Auto-scaling Plain Encoder: State Dim={state_dim} -> Hidden Dim={hidden_dim}")

        # Single latent representation branch.
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
        )

        # Use the same auxiliary predictor width as the disentangled encoder.
        pred_hidden = int(max(256, latent_dim * 4))

        # Predict the next state from latent representation and action.
        # Output dimension is _aux_target_dim: true_state_dim (default) or state_dim (no-priv).
        self.state_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, _aux_target_dim),
        )

        # Predict scalar reward from the latent representation.
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),

            nn.Linear(pred_hidden, 1),
        )

    def forward(self, obs):
        """
        Encode observations into a single latent representation.

        Args:
            obs: Input observations of shape [batch_size, state_dim]

        Returns:
            (z, None)
        """
        z = self.encoder(obs)
        return z, None
    