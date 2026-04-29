# src/denoised_mdp_encoder.py
"""
Denoised MDP encoder for use as an external comparison method.

Based on: "Denoised MDPs: Learning World Models Better Than the World Itself"
Wang et al., ICML 2022.

Core idea: separate observations into task-relevant (controllable + reward-relevant)
and task-irrelevant (uncontrollable / reward-irrelevant) components without
privileged clean-state supervision.

Key differences from DisentangledEncoder (PPF):
  - No privileged target: dynamics loss predicts next z_task from noisy next_obs
    (self-supervised), not from clean next state.
  - irrel_dynamics head: predicts next z_irrel from z_irrel only (no action),
    enforcing that z_irrel is uncontrollable.
  - task_dynamics head: predicts next z_task from (z_task, action), enforcing
    that z_task is action-controllable.
  - Independence constraint between z_task and z_irrel (same as other
    disentangled variants; covariance penalty used in the notebook).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenoisedMDPEncoder(nn.Module):
    """
    Dual-branch encoder that separates task-relevant and nuisance features
    without privileged clean-state supervision.

    Outputs:
        z_task: latent capturing controllable, reward-relevant information
        z_irrel: latent capturing uncontrollable, reward-irrelevant information

    Auxiliary heads:
        task_dynamics:  (z_task, action) -> predicted next z_task
        irrel_dynamics: z_irrel          -> predicted next z_irrel  (no action)
        reward_predictor: z_task         -> predicted scalar reward
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        super().__init__()

        hidden_dim = int(max(256, state_dim * 4))
        pred_hidden = int(max(256, latent_dim * 4))

        print(f"DenoisedMDPEncoder: state_dim={state_dim} -> hidden_dim={hidden_dim}, latent_dim={latent_dim}")

        # Task-relevant branch (controllable + reward-relevant).
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

        # Task-irrelevant branch (uncontrollable / reward-irrelevant).
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

        # Predicts next z_task from (z_task, action).
        # Action dependency enforces controllability of z_task.
        self.task_dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, latent_dim),
        )

        # Predicts next z_irrel from z_irrel only (no action).
        # Absence of action input enforces uncontrollability of z_irrel.
        self.irrel_dynamics = nn.Sequential(
            nn.Linear(latent_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, latent_dim),
        )

        # Predicts scalar reward from z_task only.
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, 1),
        )

    def forward(self, obs: torch.Tensor):
        """
        Encode observations into task and nuisance latents.

        Args:
            obs: Input observations [batch_size, state_dim]

        Returns:
            (z_task, z_irrel)
        """
        z_task = self.task_encoder(obs)
        z_irrel = self.irrel_encoder(obs)
        return z_task, z_irrel
