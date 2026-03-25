"""Generate synthetic intraday RV paths from a trained CFM checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best.pt
    python scripts/generate.py --checkpoint checkpoints/best.pt --split test --num-samples-per-day 5
"""

from __future__ import annotations

import argparse
import datetime
import logging

import numpy as np
import torch
import torch.nn.functional as F

from cfm.data.dataset import build_dataloaders
from cfm.logging import get_logger
from cfm.model import load_model_from_checkpoint
from cfm.model.sampler import CFMSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic intraday RV paths")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pt file")
    parser.add_argument("--harxhar-path", type=str, default="data/all30min")
    parser.add_argument("--output", type=str, default="generated_samples.pt")
    parser.add_argument("--num-samples-per-day", type=int, default=1, help="Synthetic paths per conditioning day")
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")
    return parser.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logger = get_logger("cfm", level)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_stats, config = load_model_from_checkpoint(args.checkpoint, device)

    # Build dataloaders to get the requested split
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        harxhar_path=args.harxhar_path,
        context_days=config.context_days,
        train_end=config.train_end,
        val_end=config.val_end,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = split_map[args.split]

    # Also need raw conditions (unscaled) to get daily_rv for denormalization
    # Re-build pairs to get dates and raw conditions
    from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv

    rv_df = load_rv(args.harxhar_path)
    daily_df = compute_daily_rv(rv_df)
    conditions_raw, proportions_raw, dates_raw = build_cfm_pairs(daily_df, config.context_days)

    train_end_dt = datetime.date.fromisoformat(config.train_end)
    val_end_dt = datetime.date.fromisoformat(config.val_end)

    if args.split == "train":
        mask = dates_raw <= np.datetime64(train_end_dt)
    elif args.split == "val":
        mask = (dates_raw > np.datetime64(train_end_dt)) & (dates_raw <= np.datetime64(val_end_dt))
    else:
        mask = dates_raw > np.datetime64(val_end_dt)

    split_conditions_raw = conditions_raw[mask]
    split_dates = dates_raw[mask]
    # daily_rv is the square of the first condition element (which is sqrt(daily_rv))
    split_daily_rv = split_conditions_raw[:, 0] ** 2  # undo sqrt

    # Create sampler
    sampler = CFMSampler(model, solver=args.solver, num_steps=args.num_steps)

    # Generate samples
    all_proportions = []
    all_absolute = []
    all_conditions = []

    idx = 0
    for proportions_batch, cond_batch in loader:
        cond_batch = cond_batch.to(device)
        B = cond_batch.shape[0]

        for _ in range(args.num_samples_per_day):
            raw = sampler.sample(cond_batch)
            props = F.softmax(raw, dim=-1)  # (B, 48)
            all_proportions.append(props.cpu())
            all_conditions.append(cond_batch.cpu())

            # Denormalize: multiply proportions by daily RV
            daily_rv_batch = torch.tensor(
                split_daily_rv[idx : idx + B], dtype=torch.float32
            )
            absolute = props.cpu() * daily_rv_batch.unsqueeze(-1)
            all_absolute.append(absolute)

        idx += B

    generated_proportions = torch.cat(all_proportions, dim=0)
    generated_absolute = torch.cat(all_absolute, dim=0)
    conditions_out = torch.cat(all_conditions, dim=0)

    # Expand dates for num_samples_per_day > 1
    dates_out = np.repeat(split_dates, args.num_samples_per_day)

    # Save
    torch.save(
        {
            "generated_proportions": generated_proportions,
            "generated_absolute": generated_absolute,
            "conditions": conditions_out,
            "dates": dates_out,
        },
        args.output,
    )

    logger.info("Split:                %s", args.split)
    logger.info("Conditioning days:    %d", len(split_dates))
    logger.info("Samples per day:      %d", args.num_samples_per_day)
    logger.info("Total generated:      %d", len(generated_proportions))
    logger.info("Proportions shape:    %s", tuple(generated_proportions.shape))
    logger.info("Absolute shape:       %s", tuple(generated_absolute.shape))
    logger.info("Saved to:             %s", args.output)


if __name__ == "__main__":
    main()
