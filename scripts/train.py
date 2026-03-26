"""Train CFM model.

Usage:
    python scripts/train.py
    python scripts/train.py --harxhar-path /path/to/all30min --epochs 100 --batch-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfm.cli import setup_logging
from cfm.config import CFMConfig
from cfm.training.trainer import CFMTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional flow matching model")

    parser.add_argument("--harxhar-path", type=str, default="data/all30min")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-end", type=str, default="2020-12-31")
    parser.add_argument("--val-end", type=str, default="2022-12-31")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument(
        "--bridge-interpolation",
        action="store_true",
        help="Enable coarse-to-fine bridge interpolation using block_granularities as waypoints.",
    )
    parser.add_argument(
        "--block-granularities",
        type=int,
        nargs="*",
        default=[12],
        help="Block sizes (in 30-min bars) for inpainting mask granularities. E.g., 12 for 6h blocks, 6 for 3h blocks.",
    )
    parser.add_argument(
        "--sparse-bar-prob",
        type=float,
        default=0.0,
        help="Probability of revealing individual sparse bars during training.",
    )
    parser.add_argument(
        "--mask-schedule",
        type=str,
        default="random_blocks",
        choices=["random_blocks", "all_blocks"],
        help="How to sample block masks during training.",
    )
    parser.add_argument(
        "--no-diurnal-prior",
        action="store_true",
        help="Disable diurnal prior (use standard Gaussian source).",
    )
    parser.add_argument(
        "--diurnal-prior-std",
        type=float,
        default=1.0,
        help="Standard deviation of the diurnal prior source distribution.",
    )
    parser.add_argument(
        "--train-source", type=str, default="vwstock_stats.parquet", help="Parquet file for training data"
    )
    parser.add_argument(
        "--val-source", type=str, default="ewstock_stats.parquet", help="Parquet file for validation data"
    )
    parser.add_argument("--test-source", type=str, default="core_stats.parquet", help="Parquet file for test data")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=auto-detect)")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logging(args)

    config = CFMConfig(
        harxhar_path=args.harxhar_path,
        train_end=args.train_end,
        val_end=args.val_end,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        sigma_min=args.sigma_min,
        block_granularities=args.block_granularities,
        sparse_bar_prob=args.sparse_bar_prob,
        mask_schedule=args.mask_schedule,
        diurnal_prior=not args.no_diurnal_prior,
        diurnal_prior_std=args.diurnal_prior_std,
        checkpoint_dir=args.checkpoint_dir,
        train_source=args.train_source,
        val_source=args.val_source,
        test_source=args.test_source,
        bridge_interpolation=args.bridge_interpolation,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    trainer = CFMTrainer(config)
    logger.info("Device: %s", trainer.device)
    logger.info("Train batches: %d", len(trainer.train_loader))
    logger.info("Val batches:   %d", len(trainer.val_loader))
    logger.info("Model params:  %s", f"{sum(p.numel() for p in trainer.model.parameters()):,}")

    trainer.fit()

    best_path = str(Path(config.checkpoint_dir) / "best.pt")
    logger.info("Training complete. Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
