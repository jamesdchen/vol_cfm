"""Generate synthetic intraday RV paths from a trained CFM checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best.pt
    python scripts/generate.py --checkpoint checkpoints/best.pt --num-samples-per-day 5
    python scripts/generate.py --checkpoint checkpoints/best.pt --output-dir samples --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cfm.cli import (
    build_dataloaders_from_config,
    load_checkpoint_and_device,
    setup_logging,
)
from cfm.model.sampler import CFMSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic intraday RV paths")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pt file")
    parser.add_argument("--harxhar-path", type=str, default="data/all30min")
    parser.add_argument("--output-dir", type=str, default="samples", help="Root output directory for chunks")
    parser.add_argument("--num-samples-per-day", type=int, default=1, help="Synthetic paths per conditioning day")
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--block-granularities",
        type=int,
        nargs="*",
        default=None,
        help="Block granularities for inference mask. Defaults to checkpoint config.",
    )
    parser.add_argument(
        "--bridge-guidance-strength",
        type=float,
        default=0.0,
        help="Strength of waypoint guidance during sampling (0=disabled).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config seed)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logging(args)

    # Load checkpoint
    model, config, diurnal_mean, device = load_checkpoint_and_device(args.checkpoint)

    # Override block granularities if specified
    if args.block_granularities is not None:
        config.block_granularities = args.block_granularities

    # Resolve seed
    seed = args.seed if args.seed is not None else config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build dataloaders (test set)
    _, _, test_loader, metadata = build_dataloaders_from_config(args.harxhar_path, config)

    # Create sampler with diurnal prior
    diurnal_mean_tensor = None
    if diurnal_mean is not None:
        diurnal_mean_tensor = torch.tensor(diurnal_mean, dtype=torch.float32, device=device)

    bridge_blocks = (
        config.block_granularities if getattr(config, "bridge_interpolation", False) else None
    )
    sampler = CFMSampler(
        model,
        solver=args.solver,
        num_steps=args.num_steps,
        diurnal_mean=diurnal_mean_tensor,
        diurnal_std=config.diurnal_prior_std,
        bridge_blocks=bridge_blocks,
        bridge_guidance_strength=args.bridge_guidance_strength,
    )

    # Output directory
    out_dir = Path(args.output_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_samples = 0

    for chunk_idx, (_x_1_batch, cond_batch) in enumerate(test_loader):
        cond_batch = cond_batch.to(device)
        B = cond_batch.shape[0]

        chunk_proportions = []
        chunk_absolute = []
        chunk_conditions = []

        # Extract daily_rv from cond[:, -1]**2
        daily_rv_batch = cond_batch[:, -1] ** 2  # (B,)

        for _ in range(args.num_samples_per_day):
            props = sampler.sample_proportions(cond_batch)  # (B, 48)
            chunk_proportions.append(props.cpu())
            chunk_conditions.append(cond_batch.cpu())

            absolute = props.cpu() * daily_rv_batch.cpu().unsqueeze(-1)
            chunk_absolute.append(absolute)

        chunk_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save(
            {
                "generated_proportions": torch.cat(chunk_proportions, dim=0),
                "generated_absolute": torch.cat(chunk_absolute, dim=0),
                "conditions": torch.cat(chunk_conditions, dim=0),
                "split": "test",
                "seed": seed,
                "chunk_idx": chunk_idx,
            },
            chunk_path,
        )

        total_chunks += 1
        total_samples += B * args.num_samples_per_day

    logger.info("Test set: %d chunks", total_chunks)
    logger.info("Seed:                 %d", seed)
    logger.info("Samples per day:      %d", args.num_samples_per_day)
    logger.info("Total chunks:         %d", total_chunks)
    logger.info("Total samples:        %d", total_samples)
    logger.info("Output dir:           %s", out_dir)


if __name__ == "__main__":
    main()
