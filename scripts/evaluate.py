"""Evaluate a trained CFM model on test data with metrics and visualizations.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --output-dir eval_results --num-samples 2000
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cfm.data.dataset import build_dataloaders
from cfm.data.loading import build_cfm_pairs, compute_daily_rv, load_rv
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
from cfm.model.vector_field import ConditionalVectorField


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained CFM model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--harxhar-path", type=str, default="/home1/jc_905/harxhar/all30min")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--num-samples", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_state = ckpt["model_state_dict"]
    scaler_stats = ckpt["scaler_stats"]
    config = ckpt["config"]

    # Reconstruct model
    model = ConditionalVectorField(
        output_dim=config.output_dim,
        cond_dim=config.cond_dim,
        hidden_dims=config.hidden_dims,
        time_embed_dim=config.time_embed_dim,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Build test set
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        harxhar_path=args.harxhar_path,
        context_days=config.context_days,
        train_end=config.train_end,
        val_end=config.val_end,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    # Get raw test data for metrics
    rv_df = load_rv(args.harxhar_path)
    daily_df = compute_daily_rv(rv_df)
    conditions_raw, proportions_raw, dates_raw = build_cfm_pairs(daily_df, config.context_days)

    val_end_dt = datetime.date.fromisoformat(config.val_end)
    test_mask = dates_raw > np.datetime64(val_end_dt)
    real_proportions = proportions_raw[test_mask]
    test_daily_rv = conditions_raw[test_mask, 0] ** 2  # undo sqrt

    # Generate samples for test conditions
    sampler = CFMSampler(model, solver=config.solver, num_steps=config.num_ode_steps)

    gen_proportions_list = []
    gen_absolute_list = []
    n_generated = 0

    for proportions_batch, cond_batch in test_loader:
        if n_generated >= args.num_samples:
            break

        cond_batch = cond_batch.to(device)
        raw = sampler.sample(cond_batch)
        props = F.softmax(raw, dim=-1)
        gen_proportions_list.append(props.cpu().numpy())

        B = cond_batch.shape[0]
        daily_rv_batch = test_daily_rv[n_generated : n_generated + B]
        absolute = props.cpu().numpy() * daily_rv_batch[:, None]
        gen_absolute_list.append(absolute)

        n_generated += B

    gen_proportions = np.concatenate(gen_proportions_list, axis=0)[: args.num_samples]
    gen_absolute = np.concatenate(gen_absolute_list, axis=0)[: args.num_samples]
    real_subset = real_proportions[: args.num_samples]
    daily_rv_subset = test_daily_rv[: args.num_samples]

    # Evaluate
    metrics = evaluate_all(real_subset, gen_proportions, daily_rv=daily_rv_subset)

    # Print metrics table
    print("\n" + "=" * 50)
    print("  CFM Evaluation Metrics (test set)")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key:<30s} {value:.6f}")
    print("=" * 50)

    # Generate visualizations
    fig1, _ = plot_sample_paths(real_subset, gen_proportions)
    fig1.savefig(str(output_dir / "sample_paths.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'sample_paths.png'}")

    fig2, _ = plot_diurnal_pattern(real_subset, gen_proportions)
    fig2.savefig(str(output_dir / "diurnal_pattern.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'diurnal_pattern.png'}")

    fig3, _ = plot_marginal_distributions(real_subset, gen_proportions)
    fig3.savefig(str(output_dir / "marginal_distributions.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'marginal_distributions.png'}")

    acf_results = acf_comparison(real_subset, gen_proportions)
    fig4, _ = plot_acf(acf_results["real_acf"], acf_results["gen_acf"])
    fig4.savefig(str(output_dir / "acf_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'acf_comparison.png'}")

    import matplotlib.pyplot as plt
    plt.close("all")

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
