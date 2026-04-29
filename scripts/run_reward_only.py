"""
Direct Python training script for reward-only ablation (B2).
Replaces nbconvert-based execution to avoid JupyterLab conflicts on RunPod.

Usage:
    METHOD=disentangled_barlow_reward_only \
    ENV_NAME=halfcheetah-medium-v2 SEED=2 NOISE_DIM=4 NOISE_SCALE=0.5 NOISE_TYPE=nonlinear \
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_reward_only.py
"""

import os
import sys
from pathlib import Path

# Resolve project root
def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists():
            return candidate
    return start

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent.parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import gym
import d4rl  # noqa: F401

from src.experiment_config import *
from src.config import CHECKPOINTS_DIR, RAW_METRICS_DIR, OBS_STATS_DIR
from src.dataset import NoisyOfflineRLDataset
from src.encoder import DisentangledEncoder
from src.iql import IQLAgent
from src.train_eval import (
    eval_policy_on_env,
    load_and_freeze_encoder,
    train_iql_from_loader,
    save_metrics_json,
)

# ------------------------------------------------------------------ #
# Method selection
# ------------------------------------------------------------------ #
METHOD = os.environ.get("METHOD", "disentangled_barlow_reward_only")
VALID_METHODS = [
    "plain_reward_only",
    "disentangled_barlow_reward_only",
    "disentangled_dcor_reward_only",
    "disentangled_hsic_reward_only",
]
assert METHOD in VALID_METHODS, f"Unknown METHOD={METHOD}. Choose from {VALID_METHODS}"

# ------------------------------------------------------------------ #
# Loss functions
# ------------------------------------------------------------------ #
def barlow_loss(z1, z2):
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    c = torch.mm(z1n.T, z2n) / z1.size(0)
    return torch.sum(c ** 2)

def _center_distance(D):
    return D - D.mean(1, keepdim=True) - D.mean(0, keepdim=True) + D.mean()

def dcor_loss(z1, z2):
    A = _center_distance(torch.cdist(z1, z1, p=2))
    B = _center_distance(torch.cdist(z2, z2, p=2))
    dcov2 = (A * B).mean()
    dvar1 = (A * A).mean()
    dvar2 = (B * B).mean()
    return dcov2 / torch.sqrt(dvar1 * dvar2 + 1e-12)

def rbf_kernel(x, sigma=1.0):
    return torch.exp(-torch.cdist(x, x, p=2.0) ** 2 / (2.0 * sigma ** 2))

def _center_kernel(K):
    return K - K.mean(1, keepdim=True) - K.mean(0, keepdim=True) + K.mean()

def hsic_loss(z1, z2, sigma=1.0):
    n = z1.size(0)
    Kc = _center_kernel(rbf_kernel(z1, sigma))
    Lc = _center_kernel(rbf_kernel(z2, sigma))
    return (Kc * Lc).sum() / ((n - 1) ** 2)

LOSS_FN = {
    "plain_reward_only":               None,
    "disentangled_barlow_reward_only": barlow_loss,
    "disentangled_dcor_reward_only":   dcor_loss,
    "disentangled_hsic_reward_only":   hsic_loss,
}[METHOD]

REPR_MODE = "plain" if METHOD == "plain_reward_only" else "disentangled"

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
def noise_tag(noise_dim, noise_scale, noise_type):
    ns = str(noise_scale).replace(".", "p")
    return f"nd{noise_dim}_ns{ns}_{noise_type}"

NOISE_TAG = noise_tag(NOISE_DIM, NOISE_SCALE, NOISE_TYPE)
SEED_TAG  = f"seed_{SEED}"

CKPT_DIR    = CHECKPOINTS_DIR / METHOD / ENV_NAME / NOISE_TAG / SEED_TAG
METRICS_DIR = RAW_METRICS_DIR / METHOD / ENV_NAME / NOISE_TAG / SEED_TAG
OBS_DIR     = OBS_STATS_DIR   / METHOD / ENV_NAME / NOISE_TAG / SEED_TAG

CKPT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
OBS_DIR.mkdir(parents=True, exist_ok=True)

print(f"METHOD:      {METHOD}")
print(f"ENV_NAME:    {ENV_NAME}")
print(f"SEED:        {SEED}")
print(f"NOISE_TAG:   {NOISE_TAG}")
print(f"DEVICE:      {DEVICE}")
print(f"CKPT_DIR:    {CKPT_DIR}")

# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #
dataset = NoisyOfflineRLDataset(
    env_name=ENV_NAME,
    noise_dim=NOISE_DIM,
    noise_scale=NOISE_SCALE,
    seed=SEED,
    use_timeouts=True,
    noise_type=NOISE_TYPE,
)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)

state_dim      = dataset.noisy_obs.shape[1]
action_dim     = dataset.actions.shape[1]
true_state_dim = dataset.obs_dim
LATENT_DIM     = int(max(true_state_dim, NOISE_DIM) * 1.5)

np.savez(OBS_DIR / "obs_stats.npz",
         obs_mean=dataset.obs_mean,
         obs_std=dataset.obs_std,
         true_state_dim=true_state_dim)

print(f"state_dim={state_dim}, true_state_dim={true_state_dim}, "
      f"action_dim={action_dim}, latent_dim={LATENT_DIM}")

# ------------------------------------------------------------------ #
# Encoder pretraining (reward only)
# ------------------------------------------------------------------ #
torch.manual_seed(SEED)
np.random.seed(SEED)

encoder = DisentangledEncoder(
    state_dim=state_dim,
    action_dim=action_dim,
    true_state_dim=true_state_dim,
    latent_dim=LATENT_DIM,
).to(DEVICE)

optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
pretrain_loader = DataLoader(dataset, batch_size=PRETRAIN_BS, shuffle=True, drop_last=True,
                             num_workers=4, pin_memory=True, persistent_workers=True)

INDEP_WEIGHT = 0.05

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    encoder.train()
    losses = []
    for obs, act, next_obs, rew, done, pure_obs, pure_next_obs in pretrain_loader:
        obs = obs.to(DEVICE)
        rew = rew.to(DEVICE)

        z_task, z_irrel = encoder(obs)
        pred_rew = encoder.reward_predictor(z_task)
        loss = torch.nn.functional.mse_loss(pred_rew.squeeze(-1), rew)

        if LOSS_FN is not None:
            loss = loss + INDEP_WEIGHT * LOSS_FN(z_task, z_irrel)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    print(f"[pretrain] epoch {epoch}/{PRETRAIN_EPOCHS}, loss={np.mean(losses):.4f}")

CKPT_ENCODER = CKPT_DIR / f"encoder_epoch_{PRETRAIN_EPOCHS}.pth"
torch.save(encoder.state_dict(), CKPT_ENCODER)
print(f"Saved encoder: {CKPT_ENCODER}")

# ------------------------------------------------------------------ #
# IQL training
# ------------------------------------------------------------------ #
encoder = load_and_freeze_encoder(encoder=encoder, ckpt_path=CKPT_ENCODER, device=DEVICE)

iql = IQLAgent(
    latent_dim=LATENT_DIM,
    action_dim=action_dim,
    device=DEVICE,
    expectile=0.7,
    temperature=3.0,
    discount=0.99,
)

iql_history = train_iql_from_loader(
    iql=iql,
    train_loader=train_loader,
    device=DEVICE,
    epochs=EPOCHS,
    ckpt_dir=CKPT_DIR,
    method=METHOD,
    save_every=10,
    encoder=encoder,
    repr_mode=REPR_MODE,
    use_tqdm=False,
)

# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #
print("Start evaluating ...")
metrics = eval_policy_on_env(
    iql=iql,
    env_name=ENV_NAME,
    encoder=encoder,
    method=METHOD,
    obs_mean=dataset.obs_mean,
    obs_std=dataset.obs_std,
    true_state_dim=true_state_dim,
    noise_dim=NOISE_DIM,
    noise_scale=NOISE_SCALE,
    noise_type=NOISE_TYPE,
    episodes=20,
    max_steps=1000,
    seed=SEED,
    device=DEVICE,
    use_fixed_noise=True,
)

metrics_path = save_metrics_json(
    metrics_dir=METRICS_DIR,
    metrics=metrics,
    env_name=ENV_NAME,
    method=METHOD,
    seed=SEED,
    noise_dim=NOISE_DIM,
    noise_scale=NOISE_SCALE,
    noise_type=NOISE_TYPE,
    extra={
        "latent_dim": LATENT_DIM,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "iql_epochs": EPOCHS,
        "encoder_checkpoint": str(CKPT_ENCODER),
        "obs_stats_path": str(OBS_DIR / "obs_stats.npz"),
        "iql_history": iql_history,
    },
)

print(f"Saved metrics: {metrics_path}")
print(metrics)
