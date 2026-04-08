"""
Regenerate all obs_stats.npz files using the updated normalization logic.

The new logic normalizes the full concatenated vector (state + noise) uniformly,
instead of only normalizing the state portion. This script scans all existing
obs_stats.npz files, parses their parameters from the path, recreates the dataset
with those parameters, and overwrites the cached statistics.

Usage:
    python scripts/regenerate_obs_stats.py

No training is involved; only dataset construction and stat computation are run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure project root is on the path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import OBS_STATS_DIR
from src.dataset import NoisyOfflineRLDataset


def parse_noise_tag(tag: str):
    """
    Parse noise parameters from a directory tag such as:
        nd10_ns1p0_concat   -> (10, 1.0, "concat")
        nd40_ns2p0_project  -> (40, 2.0, "project")
        nd20_ns0p5_nonlinear -> (20, 0.5, "nonlinear")
        nd40_ns1p0          -> (40, 1.0, "concat")   # legacy format without type suffix
    """
    # Full format: nd<dim>_ns<scale>_<type>
    m = re.fullmatch(r"nd(\d+)_ns([\dp]+)_(concat|project|nonlinear)", tag)
    if m:
        noise_dim = int(m.group(1))
        noise_scale = float(m.group(2).replace("p", "."))
        noise_type = m.group(3)
        return noise_dim, noise_scale, noise_type

    # Legacy format: nd<dim>_ns<scale>  (no type suffix → concat)
    m = re.fullmatch(r"nd(\d+)_ns([\dp]+)", tag)
    if m:
        noise_dim = int(m.group(1))
        noise_scale = float(m.group(2).replace("p", "."))
        return noise_dim, noise_scale, "concat"

    return None


def parse_seed(seed_tag: str) -> int:
    m = re.fullmatch(r"seed_(\d+)", seed_tag)
    if m:
        return int(m.group(1))
    return None


def main():
    all_stats_files = sorted(OBS_STATS_DIR.rglob("obs_stats.npz"))
    print(f"Found {len(all_stats_files)} obs_stats.npz files.\n")

    skipped = 0
    updated = 0
    failed = 0

    for stats_path in all_stats_files:
        # Path structure: OBS_STATS_DIR / method / env_name / noise_tag / seed_tag / obs_stats.npz
        parts = stats_path.relative_to(OBS_STATS_DIR).parts
        if len(parts) != 5:
            print(f"[SKIP] Unexpected path structure: {stats_path}")
            skipped += 1
            continue

        method, env_name, noise_tag, seed_tag, _ = parts

        # true_only has no noise; its obs_stats are already correct.
        if method == "true_only":
            print(f"[SKIP] true_only does not use noise: {stats_path}")
            skipped += 1
            continue

        parsed = parse_noise_tag(noise_tag)
        if parsed is None:
            print(f"[SKIP] Cannot parse noise tag '{noise_tag}': {stats_path}")
            skipped += 1
            continue

        noise_dim, noise_scale, noise_type = parsed
        seed = parse_seed(seed_tag)
        if seed is None:
            print(f"[SKIP] Cannot parse seed tag '{seed_tag}': {stats_path}")
            skipped += 1
            continue

        print(
            f"[UPDATE] {method} | {env_name} | nd={noise_dim} ns={noise_scale} {noise_type} | seed={seed}"
        )

        try:
            dataset = NoisyOfflineRLDataset(
                env_name=env_name,
                noise_dim=noise_dim,
                noise_scale=noise_scale,
                seed=seed,
                use_timeouts=True,
                noise_type=noise_type,
            )

            np.savez(
                stats_path,
                obs_mean=dataset.obs_mean,
                obs_std=dataset.obs_std,
                true_state_dim=dataset.obs_dim,
            )
            print(f"         Saved: {stats_path}")
            updated += 1

        except Exception as e:
            print(f"[FAIL]   {e}")
            failed += 1

    print(f"\nDone. updated={updated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
