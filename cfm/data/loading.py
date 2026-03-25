"""Data loading and preprocessing for the vol-CFM pipeline.

Reads harxhar all30min parquet files, constructs clean RV series,
computes daily aggregates, and builds conditional flow matching pairs.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from cfm.config import PERIODS_PER_DAY, START_DATE


def _clean_30min_series(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared cleanup to a 30-min RV series.

    Builds a full 30-min grid from START_DATE to data end, drops weekends,
    forward-fills gaps, and removes remaining NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns [t, RV] with ``t`` as datetime.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns [t, RV].
    """
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").drop_duplicates(subset="t", keep="last").reset_index(drop=True)

    # Build full 30-min grid from START_DATE to data end
    grid = pd.date_range(start=START_DATE, end=df["t"].max(), freq="30min")
    grid_df = pd.DataFrame({"t": grid})

    df = grid_df.merge(df[["t", "RV"]], on="t", how="left")

    # Drop weekends: Friday after 20:00, all Saturday, Sunday before 18:30
    dow = df["t"].dt.dayofweek  # Monday=0, Sunday=6
    hour_min = df["t"].dt.hour * 60 + df["t"].dt.minute

    is_friday_late = (dow == 4) & (hour_min > 20 * 60)  # Friday after 20:00
    is_saturday = dow == 5
    is_sunday_early = (dow == 6) & (hour_min < 18 * 60 + 30)  # Sunday before 18:30

    weekend_mask = is_friday_late | is_saturday | is_sunday_early
    df = df[~weekend_mask].reset_index(drop=True)

    # Forward-fill and drop remaining NaN
    df["RV"] = df["RV"].ffill()
    df = df.dropna(subset=["RV"]).reset_index(drop=True)

    return df[["t", "RV"]]  # type: ignore[no-any-return]


def load_rv(path: str) -> pd.DataFrame:
    """Load and merge all parquet files, return clean 30-min RV series.

    Parameters
    ----------
    path : str
        Directory containing the 6 harxhar parquet files.

    Returns
    -------
    pd.DataFrame
        Columns [t, RV], 30-min frequency, weekends dropped, forward-filled.
    """
    parquet_files = sorted(glob.glob(str(Path(path) / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {path}")

    df = pd.read_parquet(parquet_files[0])
    for f in parquet_files[1:]:
        right = pd.read_parquet(f)
        df = pd.merge(df, right, on="endbartime", how="outer")

    df = df.rename(columns={"endbartime": "t", "sumret2": "RV"})

    return _clean_30min_series(df)


def load_rv_source(path: str, parquet_file: str, rv_column: str) -> pd.DataFrame:
    """Load a single parquet file and return a clean 30-min RV series.

    Parameters
    ----------
    path : str
        Directory containing the parquet file.
    parquet_file : str
        Filename of the parquet file to read.
    rv_column : str
        Column name in the parquet file to use as RV.

    Returns
    -------
    pd.DataFrame
        Columns [t, RV], 30-min frequency, weekends dropped, forward-filled.
    """
    df = pd.read_parquet(Path(path) / parquet_file)
    df = df.rename(columns={"endbartime": "t", rv_column: "RV"})

    return _clean_30min_series(df)


def compute_daily_rv(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30-min RV into daily realized volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Columns [t, RV] at 30-min frequency.

    Returns
    -------
    pd.DataFrame
        Columns [date, daily_rv, intraday_48]. Only days with exactly 48 bars.
    """
    df = df.copy()
    df["date"] = df["t"].dt.date

    daily = (
        df.groupby("date")
        .agg(
            daily_rv=("RV", "sum"),
            intraday_48=("RV", list),
            count=("RV", "size"),
        )
        .reset_index()
    )

    # Keep only days with exactly 48 bars
    daily = daily[daily["count"] == PERIODS_PER_DAY].drop(columns="count").reset_index(drop=True)
    daily["intraday_48"] = daily["intraday_48"].apply(lambda x: np.array(x, dtype=np.float64))

    return daily  # type: ignore[no-any-return]


def compute_block_rv(intraday_48: np.ndarray, block_size: int) -> np.ndarray:
    """Reshape (48,) intraday RV into blocks and sum within each block.

    Parameters
    ----------
    intraday_48 : np.ndarray, shape (48,)
        Individual 30-min RV bars for one day.
    block_size : int
        Number of 30-min bars per block. Must evenly divide 48.

    Returns
    -------
    np.ndarray, shape (48 // block_size,)
    """
    n_blocks = PERIODS_PER_DAY // block_size
    return intraday_48.reshape(n_blocks, block_size).sum(axis=1)  # type: ignore[no-any-return]


def compute_intraday_summary(intraday_48: np.ndarray) -> np.ndarray:
    """Compute summary statistics from 48 intraday RV bars.

    Parameters
    ----------
    intraday_48 : np.ndarray, shape (48,)
        Individual 30-min RV bars for one day.

    Returns
    -------
    np.ndarray, shape (4,)
        ``[sqrt(max), sqrt(min), sqrt(first_bar), sqrt(last_bar)]``
    """
    return np.array(
        [
            np.sqrt(intraday_48.max()),
            np.sqrt(intraday_48.min()),
            np.sqrt(intraday_48[0]),
            np.sqrt(intraday_48[-1]),
        ]
    )


def build_cfm_pairs(
    daily_df: pd.DataFrame,
    context_days: int = 5,
    intermediate_blocks: list[int] | None = None,
    intermediate_representation: str = "sqrt",
    intraday_summary_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (condition, target) pairs for conditional flow matching.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of compute_daily_rv with columns [date, daily_rv, intraday_48].
    context_days : int
        Number of lagged daily RV values in the condition vector.
    intermediate_blocks : list[int] or None
        Block sizes (in 30-min bars) for sub-daily RV conditioning.
        E.g., [12] gives 4 quarter-day (6h) blocks per lag day.
    intermediate_representation : str
        How to encode intermediate RV: "sqrt", "proportion", or "both".
    intraday_summary_features : bool
        If True, append ``compute_intraday_summary`` for each lagged day.

    Returns
    -------
    conditions : np.ndarray, shape (N, cond_dim)
        Daily sqrt(RV) features followed by intermediate RV features from lagged days.
    proportions : np.ndarray, shape (N, 48)
        Intraday RV as proportion of daily total.
    dates : np.ndarray, shape (N,)
        Date for each sample.
    """
    if intermediate_blocks is None:
        intermediate_blocks = []

    daily_rv = daily_df["daily_rv"].values
    intraday = np.stack(list(daily_df["intraday_48"].values))
    dates_arr = daily_df["date"].values

    sqrt_rv = np.sqrt(daily_rv)

    conditions_list = []
    proportions_list = []
    dates_list = []

    for i in range(context_days, len(daily_df)):
        # Skip days with zero daily RV (can't compute proportions)
        if daily_rv[i] == 0:
            continue

        # Daily sqrt(RV): [today, t-1, ..., t-context_days]
        cond_parts = [sqrt_rv[i - context_days : i + 1][::-1].copy()]

        # Intermediate features from lagged days only (t-1 through t-context_days)
        for j in range(1, context_days + 1):
            for bs in intermediate_blocks:
                block_rv = compute_block_rv(intraday[i - j], bs)
                if intermediate_representation in ("sqrt", "both"):
                    cond_parts.append(np.sqrt(block_rv))
                if intermediate_representation in ("proportion", "both"):
                    cond_parts.append(block_rv / daily_rv[i - j])

            if intraday_summary_features:
                cond_parts.append(compute_intraday_summary(intraday[i - j]))

        prop = intraday[i] / daily_rv[i]

        conditions_list.append(np.concatenate(cond_parts))
        proportions_list.append(prop)
        dates_list.append(dates_arr[i])

    conditions = np.array(conditions_list, dtype=np.float32)
    proportions = np.array(proportions_list, dtype=np.float32)
    dates = np.array(dates_list)

    return conditions, proportions, dates
