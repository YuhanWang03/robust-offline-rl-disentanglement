# src/td3bc.py
"""
TD3+BC implementation for offline reinforcement learning.

TD3+BC (Fujimoto & Gu, 2021) augments the TD3 actor loss with a
behavioral cloning regularization term. The two objectives are
automatically balanced by normalizing the Q gradient with the mean
absolute Q value, controlled by the hyperparameter `alpha`.

Actor loss:
    L_pi = -lambda * Q(z, pi(z)) + ||pi(z) - a_data||^2
    lambda = alpha / (1/N * sum|Q(z, a_data)|)

Critic loss: standard TD3 Bellman backup without policy-smoothing
noise (offline setting — no environment interaction).

The public interface mirrors IQLAgent so that train_eval utilities
and eval_policy_on_env can be reused without modification:
    - agent.train_step(z, act, next_z, rew, done) -> (q_loss, actor_loss)
    - agent.actor.get_action(z)                   -> action tensor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeterministicActor(nn.Module):
    """
    Deterministic policy network.

    Outputs a tanh-squashed action in [-max_action, max_action].
    get_action() is deterministic (no sampling), matching the TD3 style.
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
        """Return the deterministic action (no stochastic sampling)."""
        return self.forward(z)


class TwinCritic(nn.Module):
    """
    Twin Q networks to reduce Q-value overestimation bias.

    Identical architecture to IQL's CriticNetwork; kept separate
    to avoid cross-file coupling.
    """

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
        x = torch.cat([z, a], dim=1)
        return self.q1(x), self.q2(x)


class TD3BCAgent:
    """
    TD3+BC offline RL agent.

    Training consists of two updates per step:
    1. Critic update via Bellman backup (no target policy smoothing).
    2. Actor update via BC-regularized policy gradient.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        device: torch.device,
        alpha: float = 2.5,
        discount: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
    ):
        self.device = device
        self.alpha = float(alpha)
        self.discount = float(discount)
        self.tau = float(tau)

        self.actor = DeterministicActor(latent_dim, action_dim).to(device)
        self.q_net = TwinCritic(latent_dim, action_dim).to(device)
        self.q_target = TwinCritic(latent_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def train_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        next_z: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Run one optimization step for critic and actor.

        Args:
            z:       Current latent states  [B, latent_dim]
            action:  Dataset actions        [B, action_dim]
            next_z:  Next latent states     [B, latent_dim]
            reward:  Rewards                [B]
            done:    Terminal flags         [B]

        Returns:
            (q_loss, actor_loss) as scalar tensors.
        """
        # ----------------------------------------------------------------
        # 1. Critic update
        # Bellman target uses the current actor (no target policy noise,
        # since we are in the offline setting).
        # ----------------------------------------------------------------
        with torch.no_grad():
            next_action = self.actor(next_z).clamp(-1.0, 1.0)
            q1_next, q2_next = self.q_target(next_z, next_action)
            q_backup = (
                reward.view(-1)
                + self.discount * (1.0 - done.view(-1)) * torch.min(q1_next, q2_next).squeeze(-1)
            )

        q1, q2 = self.q_net(z, action)
        q_loss = F.mse_loss(q1.squeeze(-1), q_backup) + F.mse_loss(q2.squeeze(-1), q_backup)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Soft-update target critic.
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        # ----------------------------------------------------------------
        # 2. Actor update: BC-regularized policy gradient
        # lambda = alpha / E[|Q(z, a_data)|] normalizes the Q gradient
        # so that the RL and BC terms are on the same scale.
        # ----------------------------------------------------------------
        pi = self.actor(z)
        q1_pi, _ = self.q_net(z, pi)

        lmbda = self.alpha / (q1_pi.abs().mean().detach() + 1e-6)
        actor_loss = -lmbda * q1_pi.mean() + F.mse_loss(pi, action)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return q_loss.detach(), actor_loss.detach()
