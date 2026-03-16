# src/utils.py
"""
Utility helpers for reproducibility, device selection, and Torch I/O.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    This function also enables deterministic CuDNN behavior for improved
    reproducibility.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return the preferred execution device.

    Args:
        prefer_cuda: If True, use CUDA when available.

    Returns:
        torch.device("cuda") if available and requested, otherwise CPU.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Union[str, os.PathLike]) -> str:
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path converted to string.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_torch(obj: Any, path: str) -> None:
    """
    Save a PyTorch object to disk.

    The parent directory is created automatically if needed.
    """
    ensure_dir(Path(path).parent)
    torch.save(obj, path)


def load_torch(path: str, map_location: Optional[torch.device] = None) -> Any:
    """
    Load a PyTorch object from disk.

    Args:
        path: File path to load.
        map_location: Optional target device.

    Returns:
        The deserialized PyTorch object.
    """
    return torch.load(path, map_location=map_location)
