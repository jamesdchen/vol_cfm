#!/usr/bin/env python
"""Export the CFM dataset (conditions, proportions, dates) to a .pt file.

Usage:
    python scripts/export_dataset.py --harxhar-path data/all30min
    python scripts/export_dataset.py --output data/cfm_dataset.pt
"""

import argparse
from pathlib import Path

import torch

from cfm.cli import setup_logging
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
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")
    args = parser.parse_args()

    logger = setup_logging(args)

    logger.info("Loading RV from %s ...", args.harxhar_path)
    rv_df = load_rv(args.harxhar_path)
    logger.info("  30-min bars: %s", f"{len(rv_df):,}")
    logger.info("  Date range:  %s -> %s", rv_df["t"].min(), rv_df["t"].max())

    logger.info("Computing daily RV ...")
    daily_df = compute_daily_rv(rv_df)
    logger.info("  Trading days (48 bars each): %s", f"{len(daily_df):,}")

    logger.info("Building CFM pairs (context_days=%d) ...", args.context_days)
    conditions, proportions, dates = build_cfm_pairs(daily_df, args.context_days)
    logger.info("  Samples: %s", f"{len(conditions):,}")
    logger.info("  Condition shape: %s", conditions.shape)
    logger.info("  Proportion shape: %s", proportions.shape)
    logger.info("  Date range: %s -> %s", dates[0], dates[-1])

    # Summary stats
    logger.info("")
    logger.info("Summary statistics:")
    logger.info("  Condition mean: %s", conditions.mean(axis=0))
    logger.info("  Condition std:  %s", conditions.std(axis=0))
    logger.info("  Proportion min: %.6f", proportions.min())
    logger.info("  Proportion max: %.6f", proportions.max())
    logger.info(
        "  Proportion row sums (should be ~1): mean=%.6f, std=%.6f",
        proportions.sum(axis=1).mean(),
        proportions.sum(axis=1).std(),
    )

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
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
