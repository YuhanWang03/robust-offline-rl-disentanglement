"""
Visualization utilities for robustness analysis in offline RL experiments.

This module centralizes:
- metrics loading,
- dataframe preparation,
- figure output path management, and
- plot generation for paper-ready analysis figures.

Saved figures follow an environment-first layout:

results/
└── figures/
    └── paper_ready/
        └── <env>/
            ├── heatmaps/
            │   └── <noise_type>/
            ├── gain_heatmaps/
            │   └── <noise_type>/
            ├── curves/
            │   ├── dim_curves/
            │   │   └── <noise_type>/
            │   └── scale_curves/
            │       └── <noise_type>/
            ├── rankings/
            └── summaries/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_METRICS_DIR = PROJECT_ROOT / "results" / "raw_metrics"
PAPER_READY_DIR = PROJECT_ROOT / "results" / "figures" / "paper_ready"


# ------------------------------------------------------------
# Method labels and default order
# ------------------------------------------------------------
METHOD_LABELS = {
    "true_only": "No Noise",
    "disentangled_cov": "Covariance",
    "disentangled_hsic": "HSIC",
    "disentangled_barlow": "Barlow",
    "disentangled_dcor": "dCor",
    "disentangled_infonce": "InfoNCE",
    "disentangled_l1": "L1",
    "plain": "Plain Encoder",
    "plain_encoder": "Plain Encoder",
    "raw_noisy": "Raw Noisy",
}

METHOD_ORDER = [
    "true_only",
    "disentangled_cov",
    "disentangled_hsic",
    "disentangled_barlow",
    "disentangled_dcor",
    "disentangled_infonce",
    "disentangled_l1",
    "plain",
    "raw_noisy",
]


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------
def _sanitize_tag(value: object) -> str:
    """Convert a value into a filename-safe tag."""
    text = str(value)
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def _label(method: str) -> str:
    """Return a display label for a method."""
    return METHOD_LABELS.get(method, method)


def _save_current_figure(path: Path, save: bool) -> Optional[Path]:
    """Save the current figure if requested."""
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved:", path)
        return path
    return None


def get_figure_output_dir(
    env: str,
    figure_family: str,
    noise_type: Optional[str] = None,
    curve_family: Optional[str] = None,
) -> Path:
    """
    Build an environment-first output directory for a figure family.

    Args:
        env: Environment name.
        figure_family: One of "heatmaps", "gain_heatmaps", "curves",
            "rankings", or "summaries".
        noise_type: Optional noise type subfolder.
        curve_family: Required when figure_family == "curves".
            Must be "dim_curves" or "scale_curves".
    """
    base = PAPER_READY_DIR / env

    if figure_family in ("heatmaps", "gain_heatmaps"):
        if noise_type is None:
            raise ValueError(f"noise_type is required for {figure_family}")
        out_dir = base / figure_family / noise_type

    elif figure_family == "curves":
        if curve_family is None:
            raise ValueError("curve_family is required when figure_family='curves'")
        out_dir = base / "curves" / curve_family
        if noise_type is not None:
            out_dir = out_dir / noise_type

    elif figure_family in ("rankings", "summaries"):
        out_dir = base / figure_family

    else:
        raise ValueError(f"Unknown figure_family: {figure_family}")

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------------------------------------------------
# Data loading and preparation
# ------------------------------------------------------------
def load_metrics_df(raw_metrics_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all metrics.json files under results/raw_metrics into a dataframe.
    """
    raw_metrics_dir = Path(raw_metrics_dir) if raw_metrics_dir is not None else RAW_METRICS_DIR

    records = []

    for path in raw_metrics_dir.rglob("metrics.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"Skip {path}: {exc}")
            continue

        method = payload.get("method")
        env = payload.get("env") or payload.get("env_name")
        seed = payload.get("seed")
        score = payload.get("normalized_score")

        record = {
            "method": method,
            "env": env,
            "seed": seed,
            "normalized_score": score,
            "noise_dim": payload.get("noise_dim"),
            "noise_scale": payload.get("noise_scale"),
            "noise_type": payload.get("noise_type"),
            "path": str(path),
        }
        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(f"No metrics.json found under {raw_metrics_dir}")

    df["method"] = df["method"].replace({"plain_encoder": "plain"})

    for col in ("noise_dim", "noise_scale", "seed", "normalized_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def split_metrics_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a full metrics dataframe into true_only and noisy subsets.
    """
    df_true = df[df["method"] == "true_only"].copy()
    df_noisy = df[df["method"] != "true_only"].copy()
    return df_true, df_noisy


def get_available_methods(df: pd.DataFrame, include_true_only: bool = True) -> List[str]:
    """
    Return methods present in the dataframe, ordered by METHOD_ORDER.
    """
    methods = METHOD_ORDER if include_true_only else [m for m in METHOD_ORDER if m != "true_only"]
    return [m for m in methods if m in df["method"].unique()]


# ------------------------------------------------------------
# Plot functions
# ------------------------------------------------------------
def plot_method_heatmap(
    df: pd.DataFrame,
    method: str,
    env: str,
    noise_type: str,
    score_col: str = "normalized_score",
    agg: str = "mean",
    save: bool = True,
):
    """
    Plot a heatmap over (noise_scale, noise_dim) for one method.
    """
    sub = df[
        (df["method"] == method) &
        (df["env"] == env) &
        (df["noise_type"] == noise_type)
    ].copy()

    if sub.empty:
        print(f"No data for method={method}, env={env}, noise_type={noise_type}")
        return None

    if agg == "mean":
        table = sub.pivot_table(
            index="noise_scale",
            columns="noise_dim",
            values=score_col,
            aggfunc="mean",
        )
    elif agg == "std":
        table = sub.pivot_table(
            index="noise_scale",
            columns="noise_dim",
            values=score_col,
            aggfunc="std",
        )
    else:
        raise ValueError("agg must be 'mean' or 'std'")

    table = table.sort_index().sort_index(axis=1)

    plt.figure(figsize=(7, 5))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"{_label(method)} | {env} | {noise_type} | {agg}")
    plt.xlabel("Noise Dimension")
    plt.ylabel("Noise Scale")
    plt.tight_layout()

    out = get_figure_output_dir(env, "heatmaps", noise_type=noise_type) / f"heatmap_{method}_{agg}.png"
    saved_path = _save_current_figure(out, save)
    plt.show()
    return saved_path


def plot_multi_method_heatmaps(
    df: pd.DataFrame,
    methods: Sequence[str],
    env: str,
    noise_type: str,
    score_col: str = "normalized_score",
    save: bool = True,
):
    """
    Plot multiple method heatmaps side by side.
    """
    methods = list(methods)
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        sub = df[
            (df["method"] == method) &
            (df["env"] == env) &
            (df["noise_type"] == noise_type)
        ].copy()

        if sub.empty:
            ax.set_title(f"{method}\n(no data)")
            ax.axis("off")
            continue

        table = sub.pivot_table(
            index="noise_scale",
            columns="noise_dim",
            values=score_col,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        sns.heatmap(table, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar=(ax == axes[-1]))
        ax.set_title(_label(method))
        ax.set_xlabel("Noise Dimension")
        ax.set_ylabel("Noise Scale")

    plt.suptitle(f"{env} | {noise_type} | Mean Normalized Score", y=1.03, fontsize=14)
    plt.tight_layout()

    tag = "_".join(methods)
    out = get_figure_output_dir(env, "heatmaps", noise_type=noise_type) / f"heatmap_compare_{tag}.png"
    saved_path = _save_current_figure(out, save)
    plt.show()
    return saved_path


def plot_relative_gain_heatmap(
    df: pd.DataFrame,
    method: str,
    baseline: str,
    env: str,
    noise_type: str,
    save: bool = True,
):
    """
    Plot a heatmap of mean(method score - baseline score) over the grid.
    """
    sub = df[
        (df["env"] == env) &
        (df["noise_type"] == noise_type) &
        (df["method"].isin([method, baseline]))
    ].copy()

    if sub.empty:
        print(f"No data for env={env}, noise_type={noise_type}")
        return None

    grouped = (
        sub.groupby(["method", "noise_scale", "noise_dim"])["normalized_score"]
        .mean()
        .reset_index()
    )

    pivot_method = grouped[grouped["method"] == method].pivot(
        index="noise_scale", columns="noise_dim", values="normalized_score"
    )
    pivot_base = grouped[grouped["method"] == baseline].pivot(
        index="noise_scale", columns="noise_dim", values="normalized_score"
    )

    gain = pivot_method - pivot_base
    gain = gain.sort_index().sort_index(axis=1)

    plt.figure(figsize=(7, 5))
    sns.heatmap(gain, annot=True, fmt=".1f", cmap="coolwarm", center=0.0)
    plt.title(
        f"Gain over {_label(baseline)}\n"
        f"{_label(method)} | {env} | {noise_type}"
    )
    plt.xlabel("Noise Dimension")
    plt.ylabel("Noise Scale")
    plt.tight_layout()

    out = (
        get_figure_output_dir(env, "gain_heatmaps", noise_type=noise_type)
        / f"gain_{method}_vs_{baseline}.png"
    )
    saved_path = _save_current_figure(out, save)
    plt.show()
    return saved_path


def plot_average_gain_bar(
    df: pd.DataFrame,
    methods: Sequence[str],
    baseline: str,
    env: str,
    noise_type: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Plot average gain over a baseline across the full grid.
    """
    methods = list(methods)
    sub = df[
        (df["env"] == env) &
        (df["noise_type"] == noise_type) &
        (df["method"].isin(methods + [baseline]))
    ].copy()

    grouped = (
        sub.groupby(["method", "noise_scale", "noise_dim"])["normalized_score"]
        .mean()
        .reset_index()
    )

    base = grouped[grouped["method"] == baseline][["noise_scale", "noise_dim", "normalized_score"]]
    base = base.rename(columns={"normalized_score": "baseline_score"})

    rows = []
    for method in methods:
        cur = grouped[grouped["method"] == method].copy()
        merged = cur.merge(base, on=["noise_scale", "noise_dim"], how="inner")
        merged["gain"] = merged["normalized_score"] - merged["baseline_score"]
        rows.append({"method": method, "avg_gain": merged["gain"].mean()})

    out_df = pd.DataFrame(rows).sort_values("avg_gain", ascending=False)

    plt.figure(figsize=(8, 4))
    plt.bar([_label(m) for m in out_df["method"]], out_df["avg_gain"])
    plt.axhline(0, linestyle="--")
    plt.ylabel(f"Average Gain over {_label(baseline)}")
    plt.title(f"{env} | {noise_type}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = (
        get_figure_output_dir(env, "summaries")
        / f"avg_gain_bar_{noise_type}_vs_{baseline}.png"
    )
    _save_current_figure(out, save)
    plt.show()

    return out_df


def plot_dim_curve_with_error(
    df: pd.DataFrame,
    methods: Sequence[str],
    env: str,
    noise_type: str,
    fixed_scale: float,
    save: bool = True,
):
    """
    Plot mean ± std curves over noise dimension at a fixed noise scale.
    """
    methods = list(methods)
    sub = df[
        (df["env"] == env) &
        (df["noise_type"] == noise_type) &
        (df["noise_scale"] == fixed_scale) &
        (df["method"].isin(methods))
    ].copy()

    if sub.empty:
        print("No matching data.")
        return None

    stats = (
        sub.groupby(["method", "noise_dim"])["normalized_score"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["method", "noise_dim"])
    )

    plt.figure(figsize=(8, 5))
    for method in methods:
        cur = stats[stats["method"] == method]
        if cur.empty:
            continue
        plt.errorbar(
            cur["noise_dim"],
            cur["mean"],
            yerr=cur["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=_label(method),
        )

    plt.xlabel(f"Noise Dimension ({noise_type})")
    plt.ylabel("Normalized D4RL Score")
    plt.title(f"{env} | Fixed Scale = {fixed_scale}")
    plt.legend()
    plt.tight_layout()

    out = (
        get_figure_output_dir(env, "curves", noise_type=noise_type, curve_family="dim_curves")
        / f"dim_curve_scale_{_sanitize_tag(fixed_scale)}.png"
    )
    saved_path = _save_current_figure(out, save)
    plt.show()
    return saved_path


def plot_scale_curve_with_error(
    df: pd.DataFrame,
    methods: Sequence[str],
    env: str,
    noise_type: str,
    fixed_dim: int,
    save: bool = True,
):
    """
    Plot mean ± std curves over noise scale at a fixed noise dimension.
    """
    methods = list(methods)
    sub = df[
        (df["env"] == env) &
        (df["noise_type"] == noise_type) &
        (df["noise_dim"] == fixed_dim) &
        (df["method"].isin(methods))
    ].copy()

    if sub.empty:
        print("No matching data.")
        return None

    stats = (
        sub.groupby(["method", "noise_scale"])["normalized_score"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["method", "noise_scale"])
    )

    plt.figure(figsize=(8, 5))
    for method in methods:
        cur = stats[stats["method"] == method]
        if cur.empty:
            continue
        plt.errorbar(
            cur["noise_scale"],
            cur["mean"],
            yerr=cur["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=_label(method),
        )

    plt.xlabel(f"Noise Scale ({noise_type})")
    plt.ylabel("Normalized D4RL Score")
    plt.title(f"{env} | Fixed Dim = {fixed_dim}")
    plt.legend()
    plt.tight_layout()

    out = (
        get_figure_output_dir(env, "curves", noise_type=noise_type, curve_family="scale_curves")
        / f"scale_curve_dim_{fixed_dim}.png"
    )
    saved_path = _save_current_figure(out, save)
    plt.show()
    return saved_path


def plot_overall_ranking(
    df: pd.DataFrame,
    env: str,
    noise_type: str,
    methods: Optional[Sequence[str]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Plot the overall average normalized score for each method under one noise type.
    """
    sub = df[
        (df["env"] == env) &
        (df["noise_type"] == noise_type)
    ].copy()

    if methods is not None:
        methods = list(methods)
        sub = sub[sub["method"].isin(methods)]

    summary = (
        sub.groupby("method")["normalized_score"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    plt.figure(figsize=(8, 4))
    plt.bar(
        [_label(m) for m in summary["method"]],
        summary["mean"],
        yerr=summary["std"].fillna(0.0),
        capsize=4,
    )
    plt.ylabel("Average Normalized Score")
    plt.title(f"Overall Robustness Ranking | {env} | {noise_type}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = get_figure_output_dir(env, "rankings") / f"overall_ranking_{noise_type}.png"
    _save_current_figure(out, save)
    plt.show()

    return summary


def plot_overall_ranking_by_noise_type(
    df: pd.DataFrame,
    env: str,
    methods: Sequence[str],
    noise_types: Sequence[str] = ("concat", "project", "nonlinear"),
    save: bool = True,
):
    """
    Plot side-by-side overall rankings for multiple noise types.
    """
    methods = list(methods)
    noise_types = list(noise_types)

    fig, axes = plt.subplots(1, len(noise_types), figsize=(6 * len(noise_types), 4), squeeze=False)
    axes = axes[0]

    for ax, noise_type in zip(axes, noise_types):
        sub = df[
            (df["env"] == env) &
            (df["noise_type"] == noise_type) &
            (df["method"].isin(methods))
        ].copy()

        summary = (
            sub.groupby("method")["normalized_score"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        ax.bar(
            [_label(m) for m in summary["method"]],
            summary["mean"],
            yerr=summary["std"].fillna(0.0),
            capsize=3,
        )
        ax.set_title(noise_type)
        ax.set_ylabel("Average Normalized Score")
        ax.tick_params(axis="x", rotation=35)

    plt.suptitle(f"Overall Robustness Ranking by Noise Type | {env}", y=1.03, fontsize=14)
    plt.tight_layout()

    out = get_figure_output_dir(env, "rankings") / "overall_ranking_by_noise_type.png"
    saved_path = _save_current_figure(out, save)
    plt.show()
    return saved_path
