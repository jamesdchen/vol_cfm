"""Train CFM model.

Usage:
    python scripts/train.py
    python scripts/train.py --harxhar-path /path/to/all30min --epochs 100 --batch-size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--context-days", type=int, default=5)
    parser.add_argument("--train-end", type=str, default="2020-12-31")
    parser.add_argument("--val-end", type=str, default="2022-12-31")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sigma-min", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

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
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = CFMTrainer(config)
    print(f"Device: {trainer.device}")
    print(f"Train batches: {len(trainer.train_loader)}")
    print(f"Val batches:   {len(trainer.val_loader)}")
    print(f"Model params:  {sum(p.numel() for p in trainer.model.parameters()):,}")
    print()

    trainer.fit()

    best_path = str(Path(config.checkpoint_dir) / "best.pt")
    print(f"\nTraining complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
