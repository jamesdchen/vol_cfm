"""PyTorch dataset and dataloader construction for the vol-CFM pipeline."""

import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv, load_rv_source
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


def _build_source_pairs(
    harxhar_path: str,
    parquet_file: str,
    rv_column: str,
    context_days: int,
    intermediate_blocks: list[int] | None,
    intermediate_representation: str,
    intraday_summary_features: bool,
    date_start: str | None = None,
    date_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single RV source and build filtered CFM pairs.

    Parameters
    ----------
    harxhar_path : str
        Directory containing the parquet file.
    parquet_file : str
        Filename of the parquet file to read.
    rv_column : str
        Column name in the parquet file to use as RV.
    context_days : int
        Number of lagged daily RV values in condition.
    intermediate_blocks : list[int] or None
        Block sizes for sub-daily RV conditioning.
    intermediate_representation : str
        Encoding for intermediate RV: "sqrt", "proportion", or "both".
    intraday_summary_features : bool
        If True, append intraday summary stats for each lagged day.
    date_start : str or None
        If set, keep only dates strictly after this date (YYYY-MM-DD).
    date_end : str or None
        If set, keep only dates on or before this date (YYYY-MM-DD).

    Returns
    -------
    conditions : np.ndarray, shape (N, cond_dim)
    proportions : np.ndarray, shape (N, 48)
    dates : np.ndarray, shape (N,)
    """
    rv_df = load_rv_source(harxhar_path, parquet_file, rv_column)
    daily_df = compute_daily_rv(rv_df)
    conditions, proportions, dates = build_cfm_pairs(
        daily_df, context_days, intermediate_blocks,
        intermediate_representation, intraday_summary_features,
    )

    # Date filtering
    mask = np.ones(len(dates), dtype=bool)
    if date_start is not None:
        dt_start = datetime.date.fromisoformat(date_start)
        mask &= dates > np.datetime64(dt_start)
    if date_end is not None:
        dt_end = datetime.date.fromisoformat(date_end)
        mask &= dates <= np.datetime64(dt_end)

    return conditions[mask], proportions[mask], dates[mask]


def build_dataloaders(
    harxhar_path: str,
    context_days: int = 5,
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    batch_size: int = 256,
    seed: int = 42,
    intermediate_blocks: list[int] | None = None,
    intermediate_representation: str = "sqrt",
    intraday_summary_features: bool = False,
    train_source: str = "vwstock_stats.parquet",
    val_source: str = "ewstock_stats.parquet",
    test_source: str = "core_stats.parquet",
    source_columns: dict[str, str] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """End-to-end dataloader construction with multi-source data loading.

    Each split (train/val/test) loads from its own parquet source file
    and RV column, with date-based filtering applied per split.

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
    intraday_summary_features : bool
        If True, append intraday summary stats for each lagged day.
    train_source : str
        Parquet filename for training data.
    val_source : str
        Parquet filename for validation data.
    test_source : str
        Parquet filename for test data.
    source_columns : dict[str, str] or None
        Mapping from parquet filename to RV column name. If None, uses defaults.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Pin memory for faster GPU transfer.
    persistent_workers : bool
        Keep worker processes alive between batches.
    prefetch_factor : int
        Number of batches to prefetch per worker.

    Returns
    -------
    train_loader, val_loader, test_loader, scaler_stats
    """
    if source_columns is None:
        source_columns = {
            "vwstock_stats.parquet": "sumret2_vwstock",
            "ewstock_stats.parquet": "sumret2_ewstock",
            "core_stats.parquet": "sumret2",
        }

    # Build pairs from each source with date filtering
    train_cond, train_prop, _ = _build_source_pairs(
        harxhar_path, train_source, source_columns[train_source],
        context_days, intermediate_blocks, intermediate_representation,
        intraday_summary_features, date_end=train_end,
    )
    val_cond, val_prop, _ = _build_source_pairs(
        harxhar_path, val_source, source_columns[val_source],
        context_days, intermediate_blocks, intermediate_representation,
        intraday_summary_features, date_start=train_end, date_end=val_end,
    )
    test_cond, test_prop, _ = _build_source_pairs(
        harxhar_path, test_source, source_columns[test_source],
        context_days, intermediate_blocks, intermediate_representation,
        intraday_summary_features, date_start=val_end,
    )

    # Fit scaler on train only
    scaler_stats = fit_scaler(train_cond)
    cond_train = apply_scaler(train_cond, scaler_stats)
    cond_val = apply_scaler(val_cond, scaler_stats)
    cond_test = apply_scaler(test_cond, scaler_stats)

    # Build datasets
    train_ds = CFMDataset(cond_train, train_prop)
    val_ds = CFMDataset(cond_val, val_prop)
    test_ds = CFMDataset(cond_test, test_prop)

    # DataLoader with throughput optimization
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, **worker_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **worker_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **worker_kwargs)

    return train_loader, val_loader, test_loader, scaler_stats
