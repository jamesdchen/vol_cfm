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
import torch

from cfm.cli import (
    build_dataloaders_from_config,
    load_checkpoint_and_device,
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
    model, config, diurnal_mean, device = load_checkpoint_and_device(args.checkpoint)

    # Build test set
    _, _, test_loader, metadata = build_dataloaders_from_config(args.harxhar_path, config)

    # Create sampler with diurnal prior
    diurnal_mean_tensor = None
    if diurnal_mean is not None:
        diurnal_mean_tensor = torch.tensor(diurnal_mean, dtype=torch.float32, device=device)

    sampler = CFMSampler(
        model,
        solver=config.solver,
        num_steps=config.num_ode_steps,
        diurnal_mean=diurnal_mean_tensor,
        diurnal_std=config.diurnal_prior_std,
    )

    # Generate samples and collect real data from test loader
    gen_proportions_list = []
    real_proportions_list = []
    daily_rv_list = []
    n_generated = 0

    for x_1_batch, cond_batch in test_loader:
        if n_generated >= args.num_samples:
            break

        cond_batch = cond_batch.to(device)
        props = sampler.sample_proportions(cond_batch)
        gen_proportions_list.append(props.cpu().numpy())
        real_proportions_list.append(x_1_batch.numpy())

        # Extract daily_rv from cond[:, -1]**2
        daily_rv_batch = (cond_batch[:, -1] ** 2).cpu().numpy()
        daily_rv_list.append(daily_rv_batch)

        B = cond_batch.shape[0]
        n_generated += B

    gen_proportions = np.concatenate(gen_proportions_list, axis=0)[: args.num_samples]
    real_proportions = np.concatenate(real_proportions_list, axis=0)[: args.num_samples]
    daily_rv_subset = np.concatenate(daily_rv_list, axis=0)[: args.num_samples]

    # Evaluate
    metrics = evaluate_all(real_proportions, gen_proportions, daily_rv=daily_rv_subset)

    # Print metrics table
    logger.info("")
    logger.info("=" * 50)
    logger.info("  CFM Evaluation Metrics (test set)")
    logger.info("=" * 50)
    for key, value in metrics.items():
        logger.info("  %-30s %.6f", key, value)
    logger.info("=" * 50)

    # Generate visualizations
    fig1, _ = plot_sample_paths(real_proportions, gen_proportions)
    save_figure(fig1, output_dir / "sample_paths.png", logger)

    fig2, _ = plot_diurnal_pattern(real_proportions, gen_proportions)
    save_figure(fig2, output_dir / "diurnal_pattern.png", logger)

    fig3, _ = plot_marginal_distributions(real_proportions, gen_proportions)
    save_figure(fig3, output_dir / "marginal_distributions.png", logger)

    acf_results = acf_comparison(real_proportions, gen_proportions)
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
