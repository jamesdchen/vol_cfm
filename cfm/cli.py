"""Shared CLI utility functions used across vol-cfm scripts."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv
from cfm.logging import get_logger
from cfm.model import load_model_from_checkpoint

if TYPE_CHECKING:
    import argparse

    from matplotlib.figure import Figure

    from cfm.config import CFMConfig
    from cfm.model.vector_field import ConditionalVectorField


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """Configure root cfm logger from CLI ``--verbose`` / ``--quiet`` flags.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected to have ``verbose`` and ``quiet``
        boolean attributes.

    Returns
    -------
    logging.Logger
        Configured logger named ``"cfm"``.
    """
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    return get_logger("cfm", level)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_and_device(
    checkpoint_path: str,
) -> tuple[ConditionalVectorField, dict, CFMConfig, torch.device]:
    """Load a trained checkpoint and auto-detect device.

    Combines :func:`get_device` with :func:`~cfm.model.load_model_from_checkpoint`.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pt`` checkpoint saved by :class:`~cfm.training.trainer.CFMTrainer`.

    Returns
    -------
    model : ConditionalVectorField
        Model with loaded weights, in eval mode on *device*.
    scaler_stats : dict
        Per-column mean/std used during training.
    config : CFMConfig
        Experiment configuration stored in the checkpoint.
    device : torch.device
        Device the model was loaded onto.
    """
    device = get_device()
    model, scaler_stats, config = load_model_from_checkpoint(checkpoint_path, device)
    return model, scaler_stats, config, device


def build_dataloaders_from_config(
    harxhar_path: str,
    config: CFMConfig,
    seed: int | None = None,
) -> tuple:
    """Build train/val/test dataloaders from a :class:`CFMConfig`.

    Wraps :func:`~cfm.data.dataset.build_dataloaders` with ``getattr``
    fallbacks for backward compatibility with older configs that may lack
    intermediate feature fields.

    Parameters
    ----------
    harxhar_path : str
        Path to all30min parquet directory.
    config : CFMConfig
        Experiment configuration (typically from a checkpoint).
    seed : int or None
        Random seed override. If *None*, uses ``config.seed``.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, test_loader, scaler_stats)``
    """
    from cfm.data.dataset import build_dataloaders

    return build_dataloaders(
        harxhar_path=harxhar_path,
        context_days=config.context_days,
        train_end=config.train_end,
        val_end=config.val_end,
        batch_size=config.batch_size,
        seed=seed if seed is not None else config.seed,
        intermediate_blocks=getattr(config, "intermediate_blocks", []),
        intermediate_representation=getattr(config, "intermediate_representation", "sqrt"),
    )


def load_raw_pairs_from_config(
    harxhar_path: str,
    config: CFMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw CFM pairs (conditions, proportions, dates) from config.

    Runs the full pipeline: :func:`~cfm.data.loading.load_rv` ->
    :func:`~cfm.data.loading.compute_daily_rv` ->
    :func:`~cfm.data.loading.build_cfm_pairs` with intermediate feature
    settings from *config* (with backward-compat fallbacks).

    Parameters
    ----------
    harxhar_path : str
        Path to all30min parquet directory.
    config : CFMConfig
        Experiment configuration.

    Returns
    -------
    conditions : np.ndarray, shape (N, cond_dim)
    proportions : np.ndarray, shape (N, 48)
    dates : np.ndarray, shape (N,)
    """
    rv_df = load_rv(harxhar_path)
    daily_df = compute_daily_rv(rv_df)
    return build_cfm_pairs(
        daily_df,
        config.context_days,
        getattr(config, "intermediate_blocks", []),
        getattr(config, "intermediate_representation", "sqrt"),
    )


def get_split_mask(
    dates: np.ndarray,
    config: CFMConfig,
    split: str,
) -> np.ndarray:
    """Return a boolean mask selecting dates belonging to *split*.

    Parameters
    ----------
    dates : np.ndarray
        Array of date values (e.g. from :func:`load_raw_pairs_from_config`).
    config : CFMConfig
        Must have ``train_end`` and ``val_end`` date strings (YYYY-MM-DD).
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.

    Returns
    -------
    np.ndarray
        Boolean mask with the same length as *dates*.

    Raises
    ------
    ValueError
        If *split* is not one of the recognised names.
    """
    train_end_dt = datetime.date.fromisoformat(config.train_end)
    val_end_dt = datetime.date.fromisoformat(config.val_end)

    if split == "train":
        return dates <= np.datetime64(train_end_dt)
    elif split == "val":
        return (dates > np.datetime64(train_end_dt)) & (dates <= np.datetime64(val_end_dt))
    elif split == "test":
        return dates > np.datetime64(val_end_dt)
    else:
        raise ValueError(f"Unknown split {split!r}; expected 'train', 'val', or 'test'")


def save_figure(fig: Figure, path: Path, logger: logging.Logger) -> None:
    """Save a matplotlib figure and log the output path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : Path
        Destination file path.
    logger : logging.Logger
        Logger instance for the "Saved: ..." message.
    """
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", path)
