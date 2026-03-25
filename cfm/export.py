"""Export generated intraday RV to parquet matching harxhar's all30min schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfm.config import PERIODS_PER_DAY

# 48 half-hour periods in a trading day: Sunday 18:30 -> Friday 17:00
# Each bar represents 30 minutes; timestamps mark the bar's end.
BAR_FREQ = pd.Timedelta(minutes=30)

# Market opens Sunday 18:00 CME (first bar ends 18:30) — but the convention
# in the harxhar dataset uses endbartime starting from the first bar of each
# calendar trading day.  We anchor at 00:00 of each date and step in 30-min
# increments to produce 48 bars (covers the 24-h futures session).
FIRST_BAR_OFFSET = pd.Timedelta(minutes=30)


def _trading_timestamps(date: np.datetime64) -> pd.DatetimeIndex:
    """Return 48 end-of-bar timestamps for a single trading day.

    Bars run from 00:30 to 24:00 (i.e. midnight of the next day) at
    30-minute intervals.  This matches the positional convention in the
    harxhar all30min parquet files where each day has exactly 48 rows.
    """
    start = pd.Timestamp(date) + FIRST_BAR_OFFSET
    return pd.date_range(start=start, periods=PERIODS_PER_DAY, freq="30min")


def export_for_harxhar(
    generated_paths: np.ndarray,
    dates: np.ndarray,
    output_path: str,
) -> None:
    """Export generated intraday RV as parquet matching harxhar's all30min schema.

    Creates a DataFrame with columns [endbartime, sumret2]:
    - For each day, expand 48 values into 48 rows with proper 30-min timestamps
    - endbartime: datetime at 30-min intervals within the trading day
    - sumret2: the generated RV value

    This can be loaded alongside real data in harxhar's pipeline.

    Parameters
    ----------
    generated_paths : np.ndarray, shape (N, 48)
        Absolute intraday RV values (not proportions).
    dates : np.ndarray, shape (N,)
        Date labels corresponding to each row.
    output_path : str
        Destination path for the parquet file.
    """
    if generated_paths.shape[0] != len(dates):
        raise ValueError(
            f"generated_paths has {generated_paths.shape[0]} rows but "
            f"dates has {len(dates)} entries"
        )
    if generated_paths.shape[1] != PERIODS_PER_DAY:
        raise ValueError(
            f"Expected {PERIODS_PER_DAY} columns, got {generated_paths.shape[1]}"
        )

    rows_endbartime = []
    rows_sumret2 = []

    for i, date in enumerate(dates):
        timestamps = _trading_timestamps(date)
        rows_endbartime.append(timestamps)
        rows_sumret2.append(generated_paths[i])

    df = pd.DataFrame(
        {
            "endbartime": np.concatenate(rows_endbartime),
            "sumret2": np.concatenate(rows_sumret2),
        }
    )

    df.to_parquet(output_path, index=False)
