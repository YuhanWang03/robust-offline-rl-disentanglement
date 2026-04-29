"""
Minimal training and evaluation utilities used directly by the experiment notebooks.

Provided utilities:
- load_and_freeze_encoder: load a saved encoder checkpoint and freeze it
- train_iql_from_loader:   train IQL from a notebook-managed DataLoader
- train_td3bc_from_loader: train TD3+BC from a notebook-managed DataLoader
- train_bc_from_loader:    train BC from a notebook-managed DataLoader
- save_metrics_json: save evaluation metrics and metadata for aggregation
- eval_policy_on_env: evaluate a trained policy in the real environment
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import d4rl  # noqa: F401
import gym
import numpy as np
import torch
from tqdm import tqdm

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_and_freeze_encoder(
    encoder: torch.nn.Module,
    ckpt_path: PathLike,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load encoder weights from a checkpoint and freeze all parameters.
    """
    ckpt_path = Path(ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def train_iql_from_loader(
    iql,
    train_loader,
    device: torch.device,
    epochs: int,
    ckpt_dir: PathLike,
    method: str,
    save_every: int = 10,
    encoder: Optional[torch.nn.Module] = None,
    repr_mode: str = "disentangled",
    use_tqdm: bool = False,
) -> List[Dict[str, float]]:
    """
    Train IQL using batches from a notebook-managed DataLoader.

    Expected batch format:
        (obs, act, next_obs, rew, done, pure_obs, pure_next_obs)
    """
    repr_mode = repr_mode.lower()
    valid_modes = {"disentangled", "plain", "raw_noisy", "true_only", "plain_no_priv", "disentangled_no_priv", "pca", "denoised_mdp", "linear"}
    if repr_mode not in valid_modes:
        raise ValueError(
            "Unsupported repr_mode: {}. Expected one of {}.".format(
                repr_mode, sorted(valid_modes)
            )
        )

    if repr_mode in {"disentangled", "plain", "plain_no_priv", "disentangled_no_priv", "pca", "denoised_mdp", "linear"} and encoder is None:
        raise ValueError("encoder must be provided for repr_mode='{}'".format(repr_mode))

    ckpt_dir = _ensure_dir(ckpt_dir)
    history = []

    if encoder is not None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    for epoch in range(1, int(epochs) + 1):
        value_losses = []
        q_losses = []
        actor_losses = []

        iterator = train_loader
        if use_tqdm:
            try:
                iterator = tqdm(train_loader, desc="[iql][{}][epoch {}]".format(method, epoch))
            except Exception:
                iterator = train_loader

        for batch in iterator:
            (
                obs,
                act,
                next_obs,
                rew,
                done,
                pure_obs,
                pure_next_obs,
            ) = [b.to(device) for b in batch]

            with torch.no_grad():
                if repr_mode == "raw_noisy":
                    z = obs
                    next_z = next_obs
                elif repr_mode == "true_only":
                    z = pure_obs
                    next_z = pure_next_obs
                else:
                    # Covers: disentangled, plain, plain_no_priv, disentangled_no_priv, denoised_mdp
                    z, _ = encoder(obs)
                    next_z, _ = encoder(next_obs)

            v_loss, q_loss, a_loss = iql.train_step(z, act, next_z, rew, done)
            value_losses.append(v_loss)
            q_losses.append(q_loss)
            actor_losses.append(a_loss)

        epoch_summary = {
            "epoch": epoch,
            "value_loss": float(torch.stack(value_losses).mean().item()) if value_losses else 0.0,
            "q_loss": float(torch.stack(q_losses).mean().item()) if q_losses else 0.0,
            "actor_loss": float(torch.stack(actor_losses).mean().item()) if actor_losses else 0.0,
        }
        history.append(epoch_summary)

        print(
            "[iql][{}] epoch {}: V={:.4f}, Q={:.4f}, A={:.4f}".format(
                method,
                epoch,
                epoch_summary["value_loss"],
                epoch_summary["q_loss"],
                epoch_summary["actor_loss"],
            )
        )

        if epoch % int(save_every) == 0 or epoch == int(epochs):
            ckpt_path = ckpt_dir / "iql_epoch_{}.pth".format(epoch)
            torch.save(
                {
                    "actor": iql.actor.state_dict(),
                    "q_net": iql.q_net.state_dict(),
                    "v_net": iql.v_net.state_dict(),
                    "method": method,
                    "repr_mode": repr_mode,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print("Saved:", ckpt_path)

    return history


def train_td3bc_from_loader(
    agent,
    train_loader,
    device: torch.device,
    epochs: int,
    ckpt_dir: PathLike,
    method: str,
    save_every: int = 10,
    encoder: Optional[torch.nn.Module] = None,
    repr_mode: str = "disentangled",
    use_tqdm: bool = False,
) -> List[Dict[str, float]]:
    """
    Train a TD3+BC agent using batches from a notebook-managed DataLoader.

    Checkpoint keys differ from IQL: saves 'actor' and 'q_net' (no 'v_net').
    All repr_mode handling is identical to train_iql_from_loader.

    Expected batch format:
        (obs, act, next_obs, rew, done, pure_obs, pure_next_obs)
    """
    repr_mode = repr_mode.lower()
    valid_modes = {"disentangled", "plain", "raw_noisy", "true_only", "plain_no_priv", "disentangled_no_priv", "pca", "denoised_mdp", "linear"}
    if repr_mode not in valid_modes:
        raise ValueError(
            "Unsupported repr_mode: {}. Expected one of {}.".format(
                repr_mode, sorted(valid_modes)
            )
        )

    if repr_mode in {"disentangled", "plain", "plain_no_priv", "disentangled_no_priv", "pca", "denoised_mdp", "linear"} and encoder is None:
        raise ValueError("encoder must be provided for repr_mode='{}'".format(repr_mode))

    ckpt_dir = _ensure_dir(ckpt_dir)
    history = []

    if encoder is not None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    for epoch in range(1, int(epochs) + 1):
        q_losses = []
        actor_losses = []

        iterator = train_loader
        if use_tqdm:
            try:
                from tqdm import tqdm
                iterator = tqdm(train_loader, desc="[td3bc][{}][epoch {}]".format(method, epoch))
            except Exception:
                iterator = train_loader

        for batch in iterator:
            (
                obs,
                act,
                next_obs,
                rew,
                done,
                pure_obs,
                pure_next_obs,
            ) = [b.to(device) for b in batch]

            with torch.no_grad():
                if repr_mode == "raw_noisy":
                    z = obs
                    next_z = next_obs
                elif repr_mode == "true_only":
                    z = pure_obs
                    next_z = pure_next_obs
                else:
                    z, _ = encoder(obs)
                    next_z, _ = encoder(next_obs)

            q_loss, actor_loss = agent.train_step(z, act, next_z, rew, done)
            q_losses.append(float(q_loss))
            actor_losses.append(float(actor_loss))

        epoch_summary = {
            "epoch": epoch,
            "q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        }
        history.append(epoch_summary)

        print(
            "[td3bc][{}] epoch {}: Q={:.4f}, A={:.4f}".format(
                method,
                epoch,
                epoch_summary["q_loss"],
                epoch_summary["actor_loss"],
            )
        )

        if epoch % int(save_every) == 0 or epoch == int(epochs):
            ckpt_path = ckpt_dir / "td3bc_epoch_{}.pth".format(epoch)
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "q_net": agent.q_net.state_dict(),
                    "q_target": agent.q_target.state_dict(),
                    "method": method,
                    "repr_mode": repr_mode,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print("Saved:", ckpt_path)

    return history


def train_bc_from_loader(
    agent,
    train_loader,
    device: torch.device,
    epochs: int,
    ckpt_dir: PathLike,
    method: str,
    save_every: int = 10,
    encoder: Optional[torch.nn.Module] = None,
    repr_mode: str = "disentangled",
    use_tqdm: bool = False,
) -> List[Dict[str, float]]:
    """
    Train a BC agent using batches from a notebook-managed DataLoader.

    next_z, reward, and done are computed but passed to agent.train_step
    for interface compatibility; BC ignores them internally.
    Checkpoint saves only 'actor' (no Q networks).

    Expected batch format:
        (obs, act, next_obs, rew, done, pure_obs, pure_next_obs)
    """
    repr_mode = repr_mode.lower()
    valid_modes = {"disentangled", "plain", "raw_noisy", "true_only", "plain_no_priv", "disentangled_no_priv", "pca", "denoised_mdp", "linear"}
    if repr_mode not in valid_modes:
        raise ValueError(
            "Unsupported repr_mode: {}. Expected one of {}.".format(
                repr_mode, sorted(valid_modes)
            )
        )

    if repr_mode in {"disentangled", "plain", "plain_no_priv", "disentangled_no_priv", "pca", "denoised_mdp", "linear"} and encoder is None:
        raise ValueError("encoder must be provided for repr_mode='{}'".format(repr_mode))

    ckpt_dir = _ensure_dir(ckpt_dir)
    history = []

    if encoder is not None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    for epoch in range(1, int(epochs) + 1):
        actor_losses = []

        iterator = train_loader
        if use_tqdm:
            try:
                from tqdm import tqdm
                iterator = tqdm(train_loader, desc="[bc][{}][epoch {}]".format(method, epoch))
            except Exception:
                iterator = train_loader

        for batch in iterator:
            (
                obs,
                act,
                next_obs,
                rew,
                done,
                pure_obs,
                pure_next_obs,
            ) = [b.to(device) for b in batch]

            with torch.no_grad():
                if repr_mode == "raw_noisy":
                    z = obs
                    next_z = next_obs
                elif repr_mode == "true_only":
                    z = pure_obs
                    next_z = pure_next_obs
                else:
                    z, _ = encoder(obs)
                    next_z, _ = encoder(next_obs)

            actor_loss = agent.train_step(z, act, next_z, rew, done)
            actor_losses.append(float(actor_loss))

        epoch_summary = {
            "epoch": epoch,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        }
        history.append(epoch_summary)

        print(
            "[bc][{}] epoch {}: BC_loss={:.4f}".format(
                method,
                epoch,
                epoch_summary["actor_loss"],
            )
        )

        if epoch % int(save_every) == 0 or epoch == int(epochs):
            ckpt_path = ckpt_dir / "bc_epoch_{}.pth".format(epoch)
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "method": method,
                    "repr_mode": repr_mode,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print("Saved:", ckpt_path)

    return history


def save_metrics_json(
    metrics_dir: PathLike,
    metrics: Dict,
    env_name: str,
    method: str,
    seed: int,
    noise_dim: Optional[int] = None,
    noise_scale: Optional[float] = None,
    noise_type: Optional[str] = None,
    extra: Optional[Dict] = None,
    filename: str = "metrics.json",
) -> Path:
    """
    Save evaluation metrics and metadata as a JSON file.
    """
    metrics_dir = _ensure_dir(metrics_dir)

    payload = {
        "env": env_name,
        "method": method,
        "seed": int(seed),
    }

    if noise_dim is not None:
        payload["noise_dim"] = int(noise_dim)
    if noise_scale is not None:
        payload["noise_scale"] = float(noise_scale)
    if noise_type is not None:
        payload["noise_type"] = noise_type

    payload.update(metrics)

    if extra is not None:
        payload.update(extra)

    metrics_path = metrics_dir / filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Saved metrics:", metrics_path)
    return metrics_path


def train_riql_from_loader(
    riql,
    train_loader,
    device: torch.device,
    epochs: int,
    ckpt_dir: PathLike,
    method: str,
    save_every: int = 10,
    encoder: Optional[torch.nn.Module] = None,
    repr_mode: str = "raw_noisy",
    use_tqdm: bool = False,
) -> List[Dict[str, float]]:
    """
    Train an RIQLAgent using batches from a notebook-managed DataLoader.

    When encoder is None (repr_mode='raw_noisy'), RIQL operates directly on
    noisy observations. When an encoder is provided, RIQL operates on the
    encoder's latent representation z, enabling PPF-framework evaluation.

    Expected batch format: (obs, act, next_obs, rew, done, pure_obs, pure_next_obs)
    """
    ckpt_dir = _ensure_dir(ckpt_dir)
    history = []

    if encoder is not None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    for epoch in range(1, int(epochs) + 1):
        value_losses = []
        q_losses = []
        actor_losses = []

        iterator = train_loader
        if use_tqdm:
            try:
                iterator = tqdm(train_loader, desc="[riql][{}][epoch {}]".format(method, epoch))
            except Exception:
                iterator = train_loader

        for batch in iterator:
            (
                obs,
                act,
                next_obs,
                rew,
                done,
                _pure_obs,
                _pure_next_obs,
            ) = [b.to(device) for b in batch]

            with torch.no_grad():
                if encoder is None:
                    z = obs
                    next_z = next_obs
                else:
                    z, _ = encoder(obs)
                    next_z, _ = encoder(next_obs)

            v_loss, q_loss, a_loss = riql.train_step(z, act, next_z, rew, done)
            value_losses.append(v_loss)
            q_losses.append(q_loss)
            actor_losses.append(a_loss)

        epoch_summary = {
            "epoch": epoch,
            "value_loss": float(torch.stack(value_losses).mean().item()) if value_losses else 0.0,
            "q_loss": float(torch.stack(q_losses).mean().item()) if q_losses else 0.0,
            "actor_loss": float(torch.stack(actor_losses).mean().item()) if actor_losses else 0.0,
        }
        history.append(epoch_summary)

        print(
            "[riql][{}] epoch {}: V={:.4f}, Q={:.4f}, A={:.4f}".format(
                method,
                epoch,
                epoch_summary["value_loss"],
                epoch_summary["q_loss"],
                epoch_summary["actor_loss"],
            )
        )

        if epoch % int(save_every) == 0 or epoch == int(epochs):
            ckpt_path = ckpt_dir / "riql_epoch_{}.pth".format(epoch)
            torch.save(
                {
                    "actor": riql.actor.state_dict(),
                    "q_net": riql.q_net.state_dict(),
                    "v_net": riql.v_net.state_dict(),
                    "method": method,
                    "repr_mode": repr_mode,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print("Saved:", ckpt_path)

    return history


@torch.no_grad()
def eval_policy_on_env(
    iql,
    env_name: str,
    encoder: Optional[torch.nn.Module] = None,
    method: str = "disentangled",
    obs_mean=None,
    obs_std=None,
    true_state_dim=None,
    noise_dim: int = 0,
    noise_scale: float = 1.0,
    noise_type: str = "concat",
    episodes: int = 20,
    max_steps: int = 1000,
    seed: int = 0,
    device: Optional[torch.device] = None,
    use_fixed_noise: bool = True,
):
    """
    Evaluate a trained policy by interacting with the real Gym/D4RL environment.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)

    try:
        env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    rng = np.random.default_rng(seed + 100) if use_fixed_noise else None
    mixing_matrix = None
    W1 = None
    W2 = None

    if noise_type == "project" and method != "true_only" and noise_dim > 0:
        mix_rng = np.random.RandomState(seed + 999)
        total_dim = true_state_dim + noise_dim
        random_mat = mix_rng.randn(total_dim, total_dim)
        Q, _ = np.linalg.qr(random_mat)
        mixing_matrix = Q.astype(np.float32)
    elif noise_type == "nonlinear" and method != "true_only" and noise_dim > 0:
        mix_rng = np.random.RandomState(seed + 999)
        total_dim = true_state_dim + noise_dim
        random_mat_1 = mix_rng.randn(total_dim, total_dim)
        W1, _ = np.linalg.qr(random_mat_1)
        W1 = W1.astype(np.float32)
        random_mat_2 = mix_rng.randn(total_dim, total_dim)
        W2, _ = np.linalg.qr(random_mat_2)
        W2 = W2.astype(np.float32)

    if encoder is not None:
        encoder.eval()
    iql.actor.eval()

    returns = []
    for _ in range(int(episodes)):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_return = 0.0

        for _step in range(int(max_steps)):
            obs = np.asarray(obs, dtype=np.float32)
            obs_norm = (obs - obs_mean) / (obs_std + 1e-6)

            if method == "true_only":
                x = obs_norm[:true_state_dim].astype(np.float32)
            else:
                if noise_dim > 0:
                    noise = (
                        rng.standard_normal(noise_dim)
                        if rng is not None
                        else np.random.randn(noise_dim)
                    ).astype(np.float32)
                    noise = noise * float(noise_scale)
                    x = np.concatenate([obs_norm.astype(np.float32), noise], axis=0)
                    if noise_type == "project":
                        x = np.dot(x, mixing_matrix)
                    elif noise_type == "nonlinear":
                        h = np.dot(x, W1)
                        h = np.tanh(h)
                        x = np.dot(h, W2)
                else:
                    x = obs_norm.astype(np.float32)

            model_in = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

            if encoder is None:
                z = model_in
            elif method.startswith(("plain", "disentangled", "linear")) or method in {"pca", "denoised_mdp"}:
                z, _ = encoder(model_in)
            else:
                raise ValueError("Unknown method: {}".format(method))

            action = iql.actor.get_action(z)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy().flatten()

            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, _info = step_out

            episode_return += reward
            obs = next_obs
            if done:
                break

        returns.append(episode_return)

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    normalized_score = env.get_normalized_score(avg_return) * 100.0
    env.close()

    return {
        "avg_return": float(avg_return),
        "std_return": float(std_return),
        "normalized_score": float(normalized_score),
    }
