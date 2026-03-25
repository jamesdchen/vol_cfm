"""Evaluate a trained CFM model on test data with metrics and visualizations.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --output-dir eval_results --num-samples 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from cfm.cli import (
    build_dataloaders_from_config,
    get_split_mask,
    load_checkpoint_and_device,
    load_raw_pairs_from_config,
    save_figure,
    setup_logging,
)
from cfm.evaluation.metrics import (
    acf_comparison,
    evaluate_all,
)
from cfm.evaluation.visualize import (
    plot_acf,
    plot_diurnal_pattern,
    plot_marginal_distributions,
    plot_sample_paths,
)
from cfm.model.sampler import CFMSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained CFM model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--harxhar-path", type=str, default="data/all30min")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logging(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    model, scaler_stats, config, device = load_checkpoint_and_device(args.checkpoint)

    # Build test set
    train_loader, val_loader, test_loader, _ = build_dataloaders_from_config(args.harxhar_path, config)

    # Get raw test data for metrics
    conditions_raw, proportions_raw, dates_raw = load_raw_pairs_from_config(args.harxhar_path, config)

    test_mask = get_split_mask(dates_raw, config, "test")
    real_proportions = proportions_raw[test_mask]
    test_daily_rv = conditions_raw[test_mask, 0] ** 2  # undo sqrt

    # Generate samples for test conditions
    sampler = CFMSampler(model, solver=config.solver, num_steps=config.num_ode_steps)

    gen_proportions_list = []
    n_generated = 0

    for _proportions_batch, cond_batch in test_loader:
        if n_generated >= args.num_samples:
            break

        cond_batch = cond_batch.to(device)
        raw = sampler.sample(cond_batch)
        props = F.softmax(raw, dim=-1)
        gen_proportions_list.append(props.cpu().numpy())

        B = cond_batch.shape[0]
        n_generated += B

    gen_proportions = np.concatenate(gen_proportions_list, axis=0)[: args.num_samples]
    real_subset = real_proportions[: args.num_samples]
    daily_rv_subset = test_daily_rv[: args.num_samples]

    # Evaluate
    metrics = evaluate_all(real_subset, gen_proportions, daily_rv=daily_rv_subset)

    # Print metrics table
    logger.info("")
    logger.info("=" * 50)
    logger.info("  CFM Evaluation Metrics (test set)")
    logger.info("=" * 50)
    for key, value in metrics.items():
        logger.info("  %-30s %.6f", key, value)
    logger.info("=" * 50)

    # Generate visualizations
    fig1, _ = plot_sample_paths(real_subset, gen_proportions)
    save_figure(fig1, output_dir / "sample_paths.png", logger)

    fig2, _ = plot_diurnal_pattern(real_subset, gen_proportions)
    save_figure(fig2, output_dir / "diurnal_pattern.png", logger)

    fig3, _ = plot_marginal_distributions(real_subset, gen_proportions)
    save_figure(fig3, output_dir / "marginal_distributions.png", logger)

    acf_results = acf_comparison(real_subset, gen_proportions)
    fig4, _ = plot_acf(acf_results["real_acf"], acf_results["gen_acf"])
    save_figure(fig4, output_dir / "acf_comparison.png", logger)

    import matplotlib.pyplot as plt

    plt.close("all")

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved: %s", metrics_path)


if __name__ == "__main__":
    main()
