"""Train CFM model.

Usage:
    python scripts/train.py
    python scripts/train.py --harxhar-path /path/to/all30min --epochs 100 --batch-size 512
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cfm.config import CFMConfig
from cfm.logging import get_logger
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
    parser.add_argument("--context-days", type=int, default=5)
    parser.add_argument("--train-end", type=str, default="2020-12-31")
    parser.add_argument("--val-end", type=str, default="2022-12-31")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument(
        "--intermediate-blocks", type=int, nargs="*", default=[],
        help="Block sizes (in 30-min bars) for intermediate RV conditioning. "
             "E.g., 12 for 6h blocks, 6 for 3h blocks. Empty = baseline.",
    )
    parser.add_argument(
        "--intermediate-representation", type=str, default="sqrt",
        choices=["sqrt", "proportion", "both"],
        help="How to represent intermediate RV features.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")

    return parser.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logger = get_logger("cfm", level)

    config = CFMConfig(
        harxhar_path=args.harxhar_path,
        context_days=args.context_days,
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
        intermediate_blocks=args.intermediate_blocks,
        intermediate_representation=args.intermediate_representation,
        checkpoint_dir=args.checkpoint_dir,
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
