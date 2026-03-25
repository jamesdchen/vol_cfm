"""CFM training loop: epoch-level train/validate/fit with checkpointing."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from cfm.config import CFMConfig
from cfm.data.dataset import build_dataloaders
from cfm.model.flow_matching import cfm_loss
from cfm.model.vector_field import ConditionalVectorField
from cfm.training.scheduler import CosineWarmupScheduler


class CFMTrainer:
    """End-to-end trainer for the conditional flow matching model.

    Parameters
    ----------
    config : CFMConfig
        Full experiment configuration.
    """

    def __init__(self, config: CFMConfig):
        self.config = config

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.scaler_stats = (
            build_dataloaders(
                harxhar_path=config.harxhar_path,
                context_days=config.context_days,
                train_end=config.train_end,
                val_end=config.val_end,
                batch_size=config.batch_size,
                seed=config.seed,
            )
        )

        # Model
        self.model = ConditionalVectorField(
            output_dim=config.output_dim,
            cond_dim=config.cond_dim,
            hidden_dims=config.hidden_dims,
            time_embed_dim=config.time_embed_dim,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = config.num_epochs * len(self.train_loader)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
        )

        # Tracking
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for proportions, conditions in self.train_loader:
            proportions = proportions.to(self.device)
            conditions = conditions.to(self.device)

            self.optimizer.zero_grad()
            loss = cfm_loss(self.model, proportions, conditions, self.config.sigma_min)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Compute mean validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for proportions, conditions in self.val_loader:
            proportions = proportions.to(self.device)
            conditions = conditions.to(self.device)

            loss = cfm_loss(self.model, proportions, conditions, self.config.sigma_min)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(self) -> None:
        """Full training loop with checkpointing."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            self.epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()
            lr = self.scheduler.get_lr()

            print(
                f"Epoch {epoch}/{self.config.num_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"lr={lr:.2e}"
            )

            # Best model checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(str(ckpt_dir / "best.pt"))

            # Periodic checkpoint
            if epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"))

        # Final checkpoint
        self.save_checkpoint(str(ckpt_dir / "final.pt"))

    def save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, scheduler state, and metadata."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
                "scaler_stats": self.scaler_stats,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint and restore all state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.scaler_stats = ckpt["scaler_stats"]
