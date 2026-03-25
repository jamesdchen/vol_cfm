#!/usr/bin/env python
"""Export the CFM dataset (conditions, proportions, dates) to a .pt file.

Usage:
    python scripts/export_dataset.py --harxhar-path data/all30min
    python scripts/export_dataset.py --output data/cfm_dataset.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv


def main():
    parser = argparse.ArgumentParser(description="Export CFM dataset to .pt file")
    parser.add_argument(
        "--harxhar-path",
        default="data/all30min",
        help="Path to all30min parquet directory",
    )
    parser.add_argument(
        "--output",
        default="data/cfm_dataset.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--context-days",
        type=int,
        default=5,
        help="Number of lagged daily RV values in condition",
    )
    args = parser.parse_args()

    print(f"Loading RV from {args.harxhar_path} ...")
    rv_df = load_rv(args.harxhar_path)
    print(f"  30-min bars: {len(rv_df):,}")
    print(f"  Date range:  {rv_df['t'].min()} -> {rv_df['t'].max()}")

    print("Computing daily RV ...")
    daily_df = compute_daily_rv(rv_df)
    print(f"  Trading days (48 bars each): {len(daily_df):,}")

    print(f"Building CFM pairs (context_days={args.context_days}) ...")
    conditions, proportions, dates = build_cfm_pairs(daily_df, args.context_days)
    print(f"  Samples: {len(conditions):,}")
    print(f"  Condition shape: {conditions.shape}")
    print(f"  Proportion shape: {proportions.shape}")
    print(f"  Date range: {dates[0]} -> {dates[-1]}")

    # Summary stats
    print("\nSummary statistics:")
    print(f"  Condition mean: {conditions.mean(axis=0)}")
    print(f"  Condition std:  {conditions.std(axis=0)}")
    print(f"  Proportion min: {proportions.min():.6f}")
    print(f"  Proportion max: {proportions.max():.6f}")
    print(f"  Proportion row sums (should be ~1): "
          f"mean={proportions.sum(axis=1).mean():.6f}, "
          f"std={proportions.sum(axis=1).std():.6f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "conditions": torch.tensor(conditions, dtype=torch.float32),
            "proportions": torch.tensor(proportions, dtype=torch.float32),
            "dates": dates,
            "context_days": args.context_days,
        },
        output_path,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
