# src/iql.py
"""
Core IQL components for offline reinforcement learning.

This module implements:
- ValueNetwork: state value function V(z)
- CriticNetwork: twin Q networks Q1(z, a), Q2(z, a)
- ActorNetwork: Gaussian policy over actions
- IQLAgent: training logic for value, critic, and actor updates

The implementation follows the standard IQL structure:
1. Expectile regression for the value network
2. Bellman regression for the critic
3. Advantage-weighted regression for the actor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ValueNetwork(nn.Module):
    """State value function V(z)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent states of shape [B, latent_dim] to values of shape [B, 1]."""
        return self.net(z)


class CriticNetwork(nn.Module):
    """Twin Q networks used to reduce overestimation bias."""

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        """Concatenate latent states and actions, then evaluate both Q networks."""
        x = torch.cat([z, a], dim=1)
        return self.q1(x), self.q2(x)


class ActorNetwork(nn.Module):
    """
    Gaussian policy network without tanh squashing.

    Actions are sampled from a Gaussian distribution and then clipped to
    [-max_action, max_action].
    """

    def __init__(self, latent_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = float(max_action)

    def forward(self, z: torch.Tensor):
        """Return the Gaussian mean and log standard deviation."""
        x = self.net(z)
        mu = self.mu(x)
        log_std = self.log_std(x)

        # Clamp log_std for numerical stability.
        log_std = torch.clamp(log_std, -5, 2)
        return mu, log_std

    def get_action(self, z: torch.Tensor, deterministic: bool = False):
        """
        Sample an action from the Gaussian policy.

        If deterministic=True, use the mean action directly.
        """
        mu, log_std = self.forward(z)
        log_std = torch.clamp(log_std, -5, 2)

        if deterministic:
            action = mu
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()

        return torch.clamp(action, -self.max_action, self.max_action)


class IQLAgent:
    """
    Implicit Q-Learning (IQL) agent.

    Training consists of three updates:
    1. Value update via expectile regression
    2. Critic update via Bellman backup
    3. Actor update via advantage-weighted regression
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
    ):
        self.device = device
        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.discount = float(discount)
        self.tau = float(tau)

        # Main networks.
        self.v_net = ValueNetwork(latent_dim).to(device)
        self.q_net = CriticNetwork(latent_dim, action_dim).to(device)
        self.q_target = CriticNetwork(latent_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.actor = ActorNetwork(latent_dim, action_dim).to(device)

        # Optimizers.
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectile regression loss.

        Positive residuals are weighted by `expectile`, while negative
        residuals are weighted by `1 - expectile`.
        """
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
        Run one optimization step for value, critic, and actor.

        Args:
            z: Current latent states.
            action: Dataset actions.
            next_z: Next latent states.
            reward: Rewards.
            done: Terminal flags.

        Returns:
            Tuple of (value_loss, critic_loss, actor_loss).
        """

        # ------------------------------------------------------------
        # 1) Value update
        # Regress V(z) toward min(Q1, Q2) using expectile loss.
        # ------------------------------------------------------------
        with torch.no_grad():
            q1, q2 = self.q_target(z, action)
            q_target = torch.min(q1, q2)

        v = self.v_net(z)
        v_loss = self.expectile_loss(q_target - v).mean()

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # ------------------------------------------------------------
        # 2) Critic update
        # Bellman target: r + gamma * (1 - done) * V(next_z)
        # ------------------------------------------------------------
        with torch.no_grad():
            next_v = self.v_net(next_z)
            q_backup = reward + self.discount * (1.0 - done) * next_v

        q1_pred, q2_pred = self.q_net(z, action)
        q_loss = F.mse_loss(q1_pred, q_backup) + F.mse_loss(q2_pred, q_backup)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Soft-update the target critic.
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # ------------------------------------------------------------
        # 3) Actor update
        # Perform advantage-weighted regression on dataset actions.
        # ------------------------------------------------------------
        with torch.no_grad():
            q1, q2 = self.q_target(z, action)
            q_val = torch.min(q1, q2)
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

        return float(v_loss.item()), float(q_loss.item()), float(actor_loss.item())
    
