# src/vis.py
"""
Visualization utilities for latent-space inspection.

This module currently provides a t-SNE-based visualization function for
encoder outputs. It supports both:
- disentangled models that return (z_task, z_irrel), and
- plain encoders that return (z, None).

Samples are colored by reward for quick qualitative inspection.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_latent_space(model, loader, device, num_samples: int = 2000):
    """
    Visualize latent representations with 2D t-SNE projections.

    Args:
        model: Encoder model. Expected to return either (z_task, z_irrel)
            or (z, None) when called with observations.
        loader: DataLoader built from the offline RL dataset.
        device: Torch device.
        num_samples: Maximum number of samples to visualize.

    Notes:
        The current dataset format is expected to return:
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

    model.eval()
    all_z1 = []
    all_z2 = []
    all_rewards = []

    with torch.no_grad():
        for batch in loader:
            (
                obs,
                _act,
                _next_obs,
                reward,
                _done,
                _pure_obs,
                _pure_next_obs,
            ) = [b.to(device) for b in batch]

            z1, z2 = model(obs)

            all_z1.append(z1.detach().cpu().numpy())
            if z2 is not None:
                all_z2.append(z2.detach().cpu().numpy())
            all_rewards.append(reward.detach().cpu().numpy())

            if len(np.concatenate(all_z1, axis=0)) >= num_samples:
                break

    z1 = np.concatenate(all_z1, axis=0)[:num_samples]
    rewards = np.concatenate(all_rewards, axis=0)[:num_samples].flatten()

    tsne = TSNE(n_components=2, random_state=42)
    z1_2d = tsne.fit_transform(z1)

    if len(all_z2) > 0:
        z2 = np.concatenate(all_z2, axis=0)[:num_samples]
        z2_2d = tsne.fit_transform(z2)
    else:
        z2_2d = None

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(z1_2d[:, 0], z1_2d[:, 1], c=rewards, s=10)
    plt.colorbar(label="Reward")
    plt.title("t-SNE of z_task (or z)")

    plt.subplot(1, 2, 2)
    if z2_2d is None:
        plt.text(0.1, 0.5, "No z_irrel available", fontsize=12)
        plt.axis("off")
    else:
        plt.scatter(z2_2d[:, 0], z2_2d[:, 1], c=rewards, s=10)
        plt.colorbar(label="Reward")
        plt.title("t-SNE of z_irrel")

    plt.tight_layout()
    plt.show()
