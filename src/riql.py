# src/riql.py
"""
Robust Implicit Q-Learning (RIQL) for offline RL under observation corruption.

Based on: "Towards Robust Offline Reinforcement Learning under Diverse Data Corruption"
Yang et al., ICLR 2024 (Spotlight).

Key differences from standard IQL (src/iql.py):
  1. Q network: twin → ensemble of N critics
  2. Critic loss: MSE → Huber loss  (handles heavy-tailed corrupted data)
  3. Value target: min(Q1,Q2) → lower quantile over ensemble  (pessimism under uncertainty)
  4. Actor/Value networks: unchanged from IQL
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .iql import ValueNetwork, ActorNetwork


class EnsembleQNetwork(nn.Module):
    """Ensemble of N independent Q networks for uncertainty estimation."""

    def __init__(self, latent_dim: int, action_dim: int, n_critics: int = 10):
        super().__init__()
        self.n_critics = n_critics
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            for _ in range(n_critics)
        ])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Returns Q values from all critics stacked along dim 0.
        Output shape: [n_critics, batch_size, 1]
        """
        x = torch.cat([z, a], dim=-1)
        return torch.stack([critic(x) for critic in self.critics], dim=0)


class RIQLAgent:
    """
    Robust Implicit Q-Learning agent.

    Compared to IQLAgent:
      - Uses an ensemble of Q networks instead of twin Q.
      - Critic update uses Huber loss instead of MSE.
      - Value target uses a lower quantile of the Q ensemble (pessimistic).
      - Actor advantage is computed from the ensemble mean Q (unbiased).
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        device: torch.device,
        expectile: float = 0.7,
        temperature: float = 3.0,
        discount: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        n_critics: int = 10,
        quantile: float = 0.25,
        huber_delta: float = 1.0,
    ):
        self.device = device
        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.discount = float(discount)
        self.tau = float(tau)
        self.n_critics = int(n_critics)
        self.quantile = float(quantile)
        self.huber_delta = float(huber_delta)

        self.v_net = ValueNetwork(latent_dim).to(device)
        self.q_net = EnsembleQNetwork(latent_dim, action_dim, n_critics).to(device)
        self.q_target = EnsembleQNetwork(latent_dim, action_dim, n_critics).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.actor = ActorNetwork(latent_dim, action_dim).to(device)

        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, (1.0 - self.expectile))
        return weight * (diff ** 2)

    def train_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        next_z: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        One optimization step for value, critic, and actor.

        Returns:
            (value_loss, critic_loss, actor_loss)
        """

        reward = reward.unsqueeze(-1) if reward.dim() == 1 else reward  # [B, 1]
        done   = done.unsqueeze(-1)   if done.dim()   == 1 else done    # [B, 1]

        # ------------------------------------------------------------
        # 1) Value update
        # Use lower quantile of the target ensemble as the Q target.
        # This gives a pessimistic value estimate that is more robust
        # to corrupted samples that inflate Q values.
        # ------------------------------------------------------------
        with torch.no_grad():
            q_all = self.q_target(z, action)          # [N, B, 1]
            q_target = q_all.quantile(self.quantile, dim=0)  # [B, 1]

        v = self.v_net(z)
        v_loss = self.expectile_loss(q_target - v).mean()

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # ------------------------------------------------------------
        # 2) Critic update
        # Huber loss is used instead of MSE: it is linear for large
        # residuals, making it less sensitive to the heavy-tailed
        # Q-value distribution induced by corrupted observations.
        # ------------------------------------------------------------
        with torch.no_grad():
            next_v = self.v_net(next_z)
            q_backup = reward + self.discount * (1.0 - done) * next_v  # [B, 1]

        q_all_pred = self.q_net(z, action)            # [N, B, 1]
        q_backup_expanded = q_backup.unsqueeze(0).expand_as(q_all_pred)
        q_loss = F.huber_loss(q_all_pred, q_backup_expanded, delta=self.huber_delta)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Soft-update target ensemble.
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # ------------------------------------------------------------
        # 3) Actor update
        # Advantage is computed from the ensemble mean Q (not the
        # pessimistic quantile) to give an unbiased signal for AWR.
        # ------------------------------------------------------------
        with torch.no_grad():
            q_all = self.q_target(z, action)
            q_val = q_all.mean(dim=0)                 # [B, 1] ensemble mean
            v_val = self.v_net(z)
            advantage = q_val - v_val

            weights = torch.exp(advantage / self.temperature).clamp(max=100.0)
            weights = weights / (weights.mean() + 1e-6)

        mu, log_std = self.actor(z)
        log_std = torch.clamp(log_std, -5, 2)
        dist = torch.distributions.Normal(mu, log_std.exp())
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        actor_loss = -(log_prob * weights).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return v_loss.detach(), q_loss.detach(), actor_loss.detach()
