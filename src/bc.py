# src/bc.py
"""
Behavioral Cloning (BC) implementation for offline RL ablation.

BC is the simplest offline policy: supervised regression from frozen
latent states to dataset actions, with no RL objective whatsoever.
It serves as the lower bound for the algorithm ablation — if the
encoder's disentanglement benefit persists even without any RL signal,
it is attributable to the representation quality alone.

Loss:
    L_BC = MSE(pi(z), a_data)

The public interface mirrors IQLAgent and TD3BCAgent:
    - agent.train_step(z, act, next_z, rew, done) -> actor_loss
      (next_z, rew, done are accepted but unused)
    - agent.actor.get_action(z)                   -> action tensor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeterministicActor(nn.Module):
    """
    Simple deterministic policy network for BC.

    Outputs a tanh-squashed action in [-max_action, max_action].
    Capacity matches the actors used in IQL and TD3+BC for a
    fair architectural comparison.
    """

    def __init__(self, latent_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = float(max_action)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) * self.max_action

    def get_action(self, z: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action."""
        return self.forward(z)


class BCAgent:
    """
    Behavioral Cloning agent.

    Trains a deterministic actor to imitate dataset actions via MSE
    regression. next_z, reward, and done are accepted in train_step
    for interface compatibility but are not used in any computation.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        device: torch.device,
        lr: float = 3e-4,
    ):
        self.device = device
        self.actor = DeterministicActor(latent_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def train_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        next_z: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Run one BC update step.

        Args:
            z:       Current latent states  [B, latent_dim]
            action:  Dataset actions        [B, action_dim]
            next_z:  Unused (interface compatibility)
            reward:  Unused (interface compatibility)
            done:    Unused (interface compatibility)

        Returns:
            actor_loss as a scalar tensor.
        """
        pi = self.actor(z)
        actor_loss = F.mse_loss(pi, action)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.detach()
