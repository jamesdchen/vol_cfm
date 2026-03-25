"""PyTorch dataset and dataloader construction for the vol-CFM pipeline."""

import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv
from cfm.data.transforms import apply_scaler, fit_scaler


class CFMDataset(Dataset):
    """Dataset of (proportion_vector, condition_vector) pairs for flow matching.

    Parameters
    ----------
    conditions : np.ndarray, shape (N, D)
    proportions : np.ndarray, shape (N, 48)
    """

    def __init__(self, conditions: np.ndarray, proportions: np.ndarray):
        self.conditions = torch.tensor(conditions, dtype=torch.float32)
        self.proportions = torch.tensor(proportions, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_1, cond) — proportions first, condition second."""
        return self.proportions[idx], self.conditions[idx]


def build_dataloaders(
    harxhar_path: str,
    context_days: int = 5,
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    batch_size: int = 256,
    seed: int = 42,
    intermediate_blocks: list[int] | None = None,
    intermediate_representation: str = "sqrt",
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """End-to-end dataloader construction.

    Parameters
    ----------
    harxhar_path : str
        Path to all30min parquet directory.
    context_days : int
        Number of lagged daily RV values in condition.
    train_end : str
        Last date (inclusive) for training set, format YYYY-MM-DD.
    val_end : str
        Last date (inclusive) for validation set.
    batch_size : int
    seed : int
        Random seed for train loader shuffling.
    intermediate_blocks : list[int] or None
        Block sizes for sub-daily RV conditioning.
    intermediate_representation : str
        Encoding for intermediate RV: "sqrt", "proportion", or "both".

    Returns
    -------
    train_loader, val_loader, test_loader, scaler_stats
    """
    # Build full dataset
    rv_df = load_rv(harxhar_path)
    daily_df = compute_daily_rv(rv_df)
    conditions, proportions, dates = build_cfm_pairs(
        daily_df, context_days, intermediate_blocks, intermediate_representation,
    )

    # Parse split dates
    train_end_dt = datetime.date.fromisoformat(train_end)
    val_end_dt = datetime.date.fromisoformat(val_end)

    # Split masks
    train_mask = dates <= np.datetime64(train_end_dt)
    val_mask = (dates > np.datetime64(train_end_dt)) & (dates <= np.datetime64(val_end_dt))
    test_mask = dates > np.datetime64(val_end_dt)

    # Fit scaler on train only
    scaler_stats = fit_scaler(conditions[train_mask])

    # Apply scaler to all splits
    cond_train = apply_scaler(conditions[train_mask], scaler_stats)
    cond_val = apply_scaler(conditions[val_mask], scaler_stats)
    cond_test = apply_scaler(conditions[test_mask], scaler_stats)

    # Build datasets
    train_ds = CFMDataset(cond_train, proportions[train_mask])
    val_ds = CFMDataset(cond_val, proportions[val_mask])
    test_ds = CFMDataset(cond_test, proportions[test_mask])

    # Build loaders
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_stats
