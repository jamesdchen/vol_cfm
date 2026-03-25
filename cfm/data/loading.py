"""Data loading and preprocessing for the vol-CFM pipeline.

Reads harxhar all30min parquet files, constructs clean RV series,
computes daily aggregates, and builds conditional flow matching pairs.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from cfm.config import PERIODS_PER_DAY, START_DATE


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

    return df[["t", "RV"]]


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
            intraday_48=("RV", lambda x: np.array(x.values, dtype=np.float64)),
            count=("RV", "size"),
        )
        .reset_index()
    )

    # Keep only days with exactly 48 bars
    daily = daily[daily["count"] == PERIODS_PER_DAY].drop(columns="count").reset_index(drop=True)

    return daily


def build_cfm_pairs(
    daily_df: pd.DataFrame, context_days: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (condition, target) pairs for conditional flow matching.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of compute_daily_rv with columns [date, daily_rv, intraday_48].
    context_days : int
        Number of lagged daily RV values in the condition vector.

    Returns
    -------
    conditions : np.ndarray, shape (N, context_days + 1)
        [sqrt(daily_rv_today), sqrt(daily_rv_{t-1}), ..., sqrt(daily_rv_{t-context_days})]
    proportions : np.ndarray, shape (N, 48)
        Intraday RV as proportion of daily total.
    dates : np.ndarray, shape (N,)
        Date for each sample.
    """
    daily_rv = daily_df["daily_rv"].values
    intraday = np.stack(daily_df["intraday_48"].values)
    dates_arr = daily_df["date"].values

    sqrt_rv = np.sqrt(daily_rv)

    conditions_list = []
    proportions_list = []
    dates_list = []

    for i in range(context_days, len(daily_df)):
        # condition: [sqrt(rv_today), sqrt(rv_{t-1}), ..., sqrt(rv_{t-context_days})]
        cond = sqrt_rv[i - context_days : i + 1][::-1].copy()  # today first, then lags
        prop = intraday[i] / daily_rv[i]

        conditions_list.append(cond)
        proportions_list.append(prop)
        dates_list.append(dates_arr[i])

    conditions = np.array(conditions_list, dtype=np.float32)
    proportions = np.array(proportions_list, dtype=np.float32)
    dates = np.array(dates_list)

    return conditions, proportions, dates
