"""Generate synthetic intraday RV paths from a trained CFM checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best.pt
    python scripts/generate.py --checkpoint checkpoints/best.pt --split all --num-samples-per-day 5
    python scripts/generate.py --checkpoint checkpoints/best.pt --output-dir samples --seed 0
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cfm.data.dataset import build_dataloaders
from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv
from cfm.logging import get_logger
from cfm.model import load_model_from_checkpoint
from cfm.model.sampler import CFMSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic intraday RV paths")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pt file")
    parser.add_argument("--harxhar-path", type=str, default="data/all30min")
    parser.add_argument("--output-dir", type=str, default="samples", help="Root output directory for chunks")
    parser.add_argument("--num-samples-per-day", type=int, default=1, help="Synthetic paths per conditioning day")
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config seed)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")
    return parser.parse_args()


def _split_mask(
    dates_raw: np.ndarray,
    split_name: str,
    train_end_dt: datetime.date,
    val_end_dt: datetime.date,
) -> np.ndarray:
    if split_name == "train":
        return dates_raw <= np.datetime64(train_end_dt)
    elif split_name == "val":
        return (dates_raw > np.datetime64(train_end_dt)) & (dates_raw <= np.datetime64(val_end_dt))
    else:
        return dates_raw > np.datetime64(val_end_dt)


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logger = get_logger("cfm", level)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler_stats, config = load_model_from_checkpoint(args.checkpoint, device)

    # Resolve seed
    seed = args.seed if args.seed is not None else config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build dataloaders
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        harxhar_path=args.harxhar_path,
        context_days=config.context_days,
        train_end=config.train_end,
        val_end=config.val_end,
        batch_size=config.batch_size,
        seed=seed,
        intermediate_blocks=getattr(config, "intermediate_blocks", []),
        intermediate_representation=getattr(config, "intermediate_representation", "sqrt"),
    )

    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Raw conditions for denormalization
    rv_df = load_rv(args.harxhar_path)
    daily_df = compute_daily_rv(rv_df)
    conditions_raw, _, dates_raw = build_cfm_pairs(
        daily_df,
        config.context_days,
        getattr(config, "intermediate_blocks", []),
        getattr(config, "intermediate_representation", "sqrt"),
    )

    train_end_dt = datetime.date.fromisoformat(config.train_end)
    val_end_dt = datetime.date.fromisoformat(config.val_end)

    # Create sampler
    sampler = CFMSampler(model, solver=args.solver, num_steps=args.num_steps)

    # Determine splits to generate
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    total_chunks = 0
    total_samples = 0

    for split_name in splits:
        loader = split_map[split_name]

        # Always use shuffle=False for generation (train loader shuffles by default,
        # which breaks the idx -> daily_rv mapping)
        if split_name == "train":
            loader = DataLoader(loader.dataset, batch_size=config.batch_size, shuffle=False)

        mask = _split_mask(dates_raw, split_name, train_end_dt, val_end_dt)
        split_conditions_raw = conditions_raw[mask]
        split_dates = dates_raw[mask]
        split_daily_rv = split_conditions_raw[:, 0] ** 2  # undo sqrt

        # Output directory for this split
        out_dir = Path(args.output_dir) / f"seed_{seed}" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        for chunk_idx, (proportions_batch, cond_batch) in enumerate(loader):
            cond_batch = cond_batch.to(device)
            B = cond_batch.shape[0]

            chunk_proportions = []
            chunk_absolute = []
            chunk_conditions = []

            for _ in range(args.num_samples_per_day):
                raw = sampler.sample(cond_batch)
                props = F.softmax(raw, dim=-1)  # (B, 48)
                chunk_proportions.append(props.cpu())
                chunk_conditions.append(cond_batch.cpu())

                daily_rv_batch = torch.tensor(
                    split_daily_rv[idx : idx + B], dtype=torch.float32
                )
                absolute = props.cpu() * daily_rv_batch.unsqueeze(-1)
                chunk_absolute.append(absolute)

            idx += B

            chunk_dates = np.repeat(split_dates[idx - B : idx], args.num_samples_per_day)

            chunk_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save(
                {
                    "generated_proportions": torch.cat(chunk_proportions, dim=0),
                    "generated_absolute": torch.cat(chunk_absolute, dim=0),
                    "conditions": torch.cat(chunk_conditions, dim=0),
                    "dates": chunk_dates,
                    "split": split_name,
                    "seed": seed,
                    "chunk_idx": chunk_idx,
                },
                chunk_path,
            )

            total_chunks += 1
            total_samples += len(chunk_dates)

        logger.info("Split %-5s: %d days, %d chunks → %s", split_name, len(split_dates), chunk_idx + 1, out_dir)

    logger.info("Seed:                 %d", seed)
    logger.info("Samples per day:      %d", args.num_samples_per_day)
    logger.info("Total chunks:         %d", total_chunks)
    logger.info("Total samples:        %d", total_samples)
    logger.info("Output dir:           %s", args.output_dir)


if __name__ == "__main__":
    main()
