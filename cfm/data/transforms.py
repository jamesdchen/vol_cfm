"""Transforms and scaling utilities for the vol-CFM pipeline."""

import numpy as np


def sqrt_transform(x: np.ndarray) -> np.ndarray:
    """Element-wise square root."""
    return np.sqrt(x)


def normalize_to_proportions(intraday: np.ndarray, daily_rv: float) -> np.ndarray:
    """Convert intraday RV to proportions of daily total."""
    return intraday / daily_rv


def denormalize_from_proportions(proportions: np.ndarray, daily_rv: float) -> np.ndarray:
    """Convert proportions back to intraday RV values."""
    return proportions * daily_rv


def fit_scaler(conditions: np.ndarray) -> dict:
    """Compute per-column mean and std for standardization.

    Parameters
    ----------
    conditions : np.ndarray, shape (N, D)

    Returns
    -------
    dict with keys "mean" and "std", each shape (D,).
    """
    return {
        "mean": conditions.mean(axis=0),
        "std": conditions.std(axis=0),
    }


def apply_scaler(conditions: np.ndarray, stats: dict) -> np.ndarray:
    """Standardize conditions using precomputed stats."""
    if "mean" not in stats or "std" not in stats:
        raise KeyError(f"stats must contain 'mean' and 'std' keys, got {list(stats.keys())}")
    return (conditions - stats["mean"]) / (stats["std"] + 1e-8)


def inverse_scaler(conditions: np.ndarray, stats: dict) -> np.ndarray:
    """Reverse standardization."""
    return conditions * (stats["std"] + 1e-8) + stats["mean"]
