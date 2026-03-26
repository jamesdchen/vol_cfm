"""PyTorch dataset and dataloader construction for inpainting CFM."""

from __future__ import annotations

import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cfm.config import CFMConfig
from cfm.data.loading import build_inpainting_pairs, compute_daily_rv, load_rv_source
from cfm.data.masking import sample_training_mask
from cfm.data.transforms import compute_diurnal_prior


class InpaintingDataset(Dataset):
    """Dataset for inpainting-style flow matching.

    Each ``__getitem__`` call generates a fresh stochastic mask so the model
    sees diverse observation patterns across epochs.

    Parameters
    ----------
    proportions : np.ndarray, shape (N, 48)
    daily_rvs : np.ndarray, shape (N,)
    intraday_raw : np.ndarray, shape (N, 48)
    config : CFMConfig
    seed : int
    """

    def __init__(
        self,
        proportions: np.ndarray,
        daily_rvs: np.ndarray,
        intraday_raw: np.ndarray,
        config: CFMConfig,
        seed: int = 42,
    ):
        self.proportions = torch.tensor(proportions, dtype=torch.float32)
        self.daily_rvs = daily_rvs  # keep as numpy for masking
        self.intraday_raw = intraday_raw  # keep as numpy for masking
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.proportions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_1, cond) where cond = [mask, known_values, sqrt(daily_rv)].

        Returns
        -------
        x_1 : torch.Tensor, shape (48,)
            Target proportions.
        cond : torch.Tensor, shape (97,)
            Channel conditioning: mask(48) + known_values(48) + sqrt(daily_rv)(1).
        """
        x_1 = self.proportions[idx]

        mask, known_values = sample_training_mask(
            self.intraday_raw[idx],
            self.daily_rvs[idx],
            self.config.block_granularities,
            self.config.sparse_bar_prob,
            self.config.mask_schedule,
            self.rng,
        )

        sqrt_daily_rv = np.sqrt(self.daily_rvs[idx]).astype(np.float32)
        cond = np.concatenate([mask, known_values, [sqrt_daily_rv]])
        return x_1, torch.tensor(cond, dtype=torch.float32)


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's RNG differently."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, InpaintingDataset):
            dataset.rng = np.random.default_rng(dataset.seed + worker_id)


def _build_source_inpainting(
    harxhar_path: str,
    parquet_file: str,
    rv_column: str,
    date_start: str | None = None,
    date_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single RV source and build filtered inpainting data."""
    rv_df = load_rv_source(harxhar_path, parquet_file, rv_column)
    daily_df = compute_daily_rv(rv_df)
    proportions, daily_rvs, intraday_raw, dates = build_inpainting_pairs(daily_df)

    # Date filtering
    mask = np.ones(len(dates), dtype=bool)
    if date_start is not None:
        dt_start = datetime.date.fromisoformat(date_start)
        mask &= dates > np.datetime64(dt_start)
    if date_end is not None:
        dt_end = datetime.date.fromisoformat(date_end)
        mask &= dates <= np.datetime64(dt_end)

    return proportions[mask], daily_rvs[mask], intraday_raw[mask], dates[mask]


def build_inpainting_dataloaders(
    harxhar_path: str,
    config: CFMConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test dataloaders for inpainting CFM.

    Parameters
    ----------
    harxhar_path : str
        Path to all30min parquet directory.
    config : CFMConfig

    Returns
    -------
    train_loader, val_loader, test_loader, metadata
        metadata contains ``diurnal_mean`` (48,) from training data.
    """
    source_columns = config.source_columns

    train_prop, train_rv, train_raw, _ = _build_source_inpainting(
        harxhar_path,
        config.train_source,
        source_columns[config.train_source],
        date_end=config.train_end,
    )
    val_prop, val_rv, val_raw, _ = _build_source_inpainting(
        harxhar_path,
        config.val_source,
        source_columns[config.val_source],
        date_start=config.train_end,
        date_end=config.val_end,
    )
    test_prop, test_rv, test_raw, _ = _build_source_inpainting(
        harxhar_path,
        config.test_source,
        source_columns[config.test_source],
        date_start=config.val_end,
    )

    # Compute diurnal prior from training proportions
    diurnal_mean = compute_diurnal_prior(train_prop)

    # Build datasets
    train_ds = InpaintingDataset(train_prop, train_rv, train_raw, config, seed=config.seed)
    val_ds = InpaintingDataset(val_prop, val_rv, val_raw, config, seed=config.seed + 1)
    test_ds = InpaintingDataset(test_prop, test_rv, test_raw, config, seed=config.seed + 2)

    # DataLoader kwargs
    worker_kwargs: dict = {}
    if config.num_workers > 0:
        worker_kwargs = {
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers,
            "prefetch_factor": config.prefetch_factor,
        }

    g = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=_worker_init_fn,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=_worker_init_fn,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=_worker_init_fn,
        **worker_kwargs,
    )

    metadata = {"diurnal_mean": diurnal_mean}
    return train_loader, val_loader, test_loader, metadata
