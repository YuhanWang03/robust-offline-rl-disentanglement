# src/pca_encoder.py
"""
PCA-based linear encoder for use as an external baseline.

PCAEncoder projects noisy observations onto the top-k principal components,
providing a classical linear denoising baseline that requires no privileged
information and no neural network training.

Public interface mirrors DisentangledEncoder for drop-in compatibility:
    encoder(obs_tensor) -> (z_tensor, None)
    encoder.eval()      -> self  (no-op)
    encoder.parameters() -> empty iterator
    encoder.fit(obs)    -> fits PCA on the provided observation tensor

Typical usage:
    pca_encoder = PCAEncoder(n_components=true_state_dim)
    pca_encoder.fit(dataset.noisy_obs)
    # Then pass directly to train_iql_from_loader with repr_mode="pca"
"""

from __future__ import annotations

import numpy as np
import torch


class PCAEncoder:
    """
    Linear PCA baseline encoder.

    Fits sklearn PCA on the offline dataset observations and projects
    each observation onto the top-k principal components at inference time.
    No privileged information, no neural network, no task-aware objective.

    The explained variance ratio after fitting indicates how much of the
    signal is retained; for structured noise (project/nonlinear types),
    this ratio is expected to be poor, demonstrating the need for nonlinear
    task-aware pretraining (PPF).
    """

    def __init__(self, n_components: int):
        """
        Args:
            n_components: Target latent dimension. Typically set to
                          true_state_dim to match the clean state size.
        """
        from sklearn.decomposition import PCA  # lazy import

        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        self._fitted = False

    # ── Core interface ─────────────────────────────────────────────────────

    def fit(self, obs_tensor: torch.Tensor) -> "PCAEncoder":
        """
        Fit PCA on the full offline dataset.

        Args:
            obs_tensor: Noisy observations of shape [N, obs_dim].

        Returns:
            self (for method chaining).
        """
        X = obs_tensor.detach().cpu().numpy()
        self.pca.fit(X)
        self._fitted = True

        explained = float(self.pca.explained_variance_ratio_.sum())
        print(
            "[PCAEncoder] fitted on {} samples, {} components → "
            "explained variance = {:.3f}".format(len(X), self.n_components, explained)
        )
        return self

    def __call__(self, obs_tensor: torch.Tensor):
        """
        Project observations to PCA latent space.

        Args:
            obs_tensor: Noisy observations [B, obs_dim].

        Returns:
            (z, None): z has shape [B, n_components], None for interface compat.
        """
        assert self._fitted, "Call fit() before using PCAEncoder."
        device = obs_tensor.device
        X = obs_tensor.detach().cpu().numpy()
        z = self.pca.transform(X).astype(np.float32)
        return torch.from_numpy(z).to(device), None

    # ── Duck-typing compatibility with nn.Module interface ─────────────────

    def eval(self) -> "PCAEncoder":
        """No-op. Required for interface compatibility with neural encoders."""
        return self

    def train(self, mode: bool = True) -> "PCAEncoder":
        """No-op. Required for interface compatibility with neural encoders."""
        return self

    def parameters(self):
        """Returns empty iterator. PCA has no gradient parameters."""
        return iter([])

    def to(self, device) -> "PCAEncoder":
        """No-op. PCA computation is CPU-based (numpy)."""
        return self

    # ── Checkpoint utilities ───────────────────────────────────────────────

    def save(self, path) -> None:
        """
        Save PCA components to a .npz file for reproducibility.

        Args:
            path: File path (str or Path). Will be written as .npz.
        """
        assert self._fitted, "Cannot save an unfitted PCAEncoder."
        np.savez(
            str(path),
            components=self.pca.components_,
            mean=self.pca.mean_,
            explained_variance=self.pca.explained_variance_,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
            n_components=np.array([self.n_components]),
        )
        print("[PCAEncoder] saved to:", path)

    @classmethod
    def load(cls, path) -> "PCAEncoder":
        """
        Restore a saved PCAEncoder from a .npz file.

        Args:
            path: File path written by save().

        Returns:
            Fitted PCAEncoder instance.
        """
        data = np.load(str(path))
        n_components = int(data["n_components"][0])
        encoder = cls(n_components=n_components)

        from sklearn.decomposition import PCA

        encoder.pca = PCA(n_components=n_components)
        encoder.pca.components_ = data["components"]
        encoder.pca.mean_ = data["mean"]
        encoder.pca.explained_variance_ = data["explained_variance"]
        encoder.pca.explained_variance_ratio_ = data["explained_variance_ratio"]
        encoder.pca.n_components_ = n_components
        encoder._fitted = True
        print("[PCAEncoder] loaded from:", path)
        return encoder
