# src/dataset.py
"""
Dataset utilities for offline RL experiments with synthetic observation noise.

This module provides a PyTorch Dataset wrapper around D4RL offline datasets.
It supports three observation corruption modes:

- concat: append Gaussian noise dimensions directly
- project: apply a random orthogonal mixing matrix after concatenation
- nonlinear: apply two orthogonal transforms with a tanh nonlinearity in between

The dataset returns both:
1. corrupted observations used for policy learning, and
2. clean normalized state slices used as supervision targets for encoder pretraining.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import gym
import d4rl  # noqa: F401  # Registers D4RL environments in Gym.


class NoisyOfflineRLDataset(Dataset):
    """
    Offline RL dataset with controllable synthetic observation corruption.

    Each sample is returned as:
        (
            noisy_obs,
            action,
            noisy_next_obs,
            reward,
            done,
            pure_obs,
            pure_next_obs,
        )

    where:
        noisy_obs / noisy_next_obs:
            Corrupted observations used by the policy and encoder.

        pure_obs / pure_next_obs:
            Clean normalized state-only slices (the first obs_dim dimensions),
            typically used as supervision targets during encoder pretraining.
    """

    def __init__(
        self,
        env_name: str,
        noise_dim: int = 20,
        noise_scale: float = 1.0,
        seed: int = 42,
        use_timeouts: bool = True,
        noise_type: str = "concat",
    ):
        rng = np.random.default_rng(seed)

        # Load the offline dataset from the D4RL environment.
        env = gym.make(env_name)
        dataset = env.get_dataset()

        obs = dataset["observations"]
        next_obs = dataset["next_observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]

        # Use timeout flags as terminal signals when requested and available.
        if use_timeouts and "timeouts" in dataset:
            dones = terminals | dataset["timeouts"]
        else:
            dones = terminals

        # Store dataset metadata.
        obs_dim = obs.shape[1]
        self.obs_dim = obs_dim
        self.noise_type = noise_type

        # Generate per-transition Gaussian noise.
        if noise_dim > 0:
            noise = rng.standard_normal((obs.shape[0], noise_dim)) * noise_scale
            next_noise = rng.standard_normal((next_obs.shape[0], noise_dim)) * noise_scale
        else:
            noise = np.zeros((obs.shape[0], 0))
            next_noise = np.zeros((next_obs.shape[0], 0))

        # Construct the base noisy observations by concatenating state and noise.
        noisy_obs = np.concatenate([obs, noise], axis=1)
        noisy_next_obs = np.concatenate([next_obs, next_noise], axis=1)

        # Normalize only the clean state portion and keep the noise portion unchanged.
        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0) + 1e-6
        self.obs_mean = obs_mean
        self.obs_std = obs_std

        noisy_obs[:, :obs_dim] = (noisy_obs[:, :obs_dim] - obs_mean) / obs_std
        noisy_next_obs[:, :obs_dim] = (noisy_next_obs[:, :obs_dim] - obs_mean) / obs_std

        # Preserve the normalized clean state slices for representation supervision.
        pure_obs = noisy_obs[:, :obs_dim].copy()
        pure_next_obs = noisy_next_obs[:, :obs_dim].copy()

        # Apply optional post-concatenation mixing.
        if self.noise_type == "project" and noise_dim > 0:
            mix_rng = np.random.RandomState(seed + 999)
            total_dim = obs_dim + noise_dim

            random_mat = mix_rng.randn(total_dim, total_dim)
            Q, _ = np.linalg.qr(random_mat)
            self.mixing_matrix = Q.astype(np.float32)

            noisy_obs = np.dot(noisy_obs, self.mixing_matrix)
            noisy_next_obs = np.dot(noisy_next_obs, self.mixing_matrix)

        elif self.noise_type == "nonlinear" and noise_dim > 0:
            mix_rng = np.random.RandomState(seed + 999)
            total_dim = obs_dim + noise_dim

            # First orthogonal transform.
            random_mat_1 = mix_rng.randn(total_dim, total_dim)
            W1, _ = np.linalg.qr(random_mat_1)
            self.W1 = W1.astype(np.float32)

            # Second orthogonal transform.
            random_mat_2 = mix_rng.randn(total_dim, total_dim)
            W2, _ = np.linalg.qr(random_mat_2)
            self.W2 = W2.astype(np.float32)

            # Apply the same nonlinear mixing to current and next observations.
            h_obs = np.dot(noisy_obs, self.W1)
            h_obs = np.tanh(h_obs)
            noisy_obs = np.dot(h_obs, self.W2)

            h_next_obs = np.dot(noisy_next_obs, self.W1)
            h_next_obs = np.tanh(h_next_obs)
            noisy_next_obs = np.dot(h_next_obs, self.W2)

        # Materialize tensors once for efficient indexed access.
        self.noisy_obs = torch.from_numpy(noisy_obs).float()
        self.noisy_next_obs = torch.from_numpy(noisy_next_obs).float()
        self.actions = torch.from_numpy(actions).float()
        self.rewards = torch.from_numpy(rewards).float()
        self.dones = torch.from_numpy(dones).float()
        self.pure_obs = torch.from_numpy(pure_obs).float()
        self.pure_next_obs = torch.from_numpy(pure_next_obs).float()

    def __len__(self) -> int:
        """Return the number of transitions in the offline dataset."""
        return len(self.noisy_obs)

    def __getitem__(self, idx: int):
        """
        Return a single transition tuple.

        Returns:
            (
                noisy_obs,
                action,
                noisy_next_obs,
                reward,
                done,
                pure_obs,
                pure_next_obs,
            )
        """
        return (
            self.noisy_obs[idx],
            self.actions[idx],
            self.noisy_next_obs[idx],
            self.rewards[idx],
            self.dones[idx],
            self.pure_obs[idx],
            self.pure_next_obs[idx],
        )
    