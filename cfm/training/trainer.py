"""CFM training loop: epoch-level train/validate/fit with checkpointing."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from cfm.cli import get_device
from cfm.config import CFMConfig
from cfm.data.dataset import build_dataloaders
from cfm.logging import get_logger
from cfm.model.flow_matching import cfm_loss
from cfm.model.vector_field import ConditionalVectorField
from cfm.training.scheduler import CosineWarmupScheduler

logger = get_logger(__name__)


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
        self.device = get_device()

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.scaler_stats = build_dataloaders(
            harxhar_path=config.harxhar_path,
            context_days=config.context_days,
            train_end=config.train_end,
            val_end=config.val_end,
            batch_size=config.batch_size,
            seed=config.seed,
            intermediate_blocks=config.intermediate_blocks,
            intermediate_representation=config.intermediate_representation,
            intraday_summary_features=config.intraday_summary_features,
            train_source=config.train_source,
            val_source=config.val_source,
            test_source=config.test_source,
            source_columns=config.source_columns,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
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

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
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

        try:
            for epoch in range(1, self.config.num_epochs + 1):
                self.epoch = epoch

                train_loss = self.train_epoch()

                if math.isnan(train_loss):
                    logger.error("NaN training loss at epoch %d, saving checkpoint and stopping", epoch)
                    self.save_checkpoint(str(ckpt_dir / "nan_checkpoint.pt"))
                    return

                val_loss = self.validate()
                lr = self.scheduler.get_lr()

                logger.info(
                    "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                    epoch,
                    self.config.num_epochs,
                    train_loss,
                    val_loss,
                    lr,
                )

                # Best model checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(str(ckpt_dir / "best.pt"))

                # Periodic checkpoint
                if epoch % self.config.checkpoint_every == 0:
                    self.save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"))
        except KeyboardInterrupt:
            logger.warning("Training interrupted at epoch %d, saving checkpoint", self.epoch)
            self.save_checkpoint(str(ckpt_dir / "interrupted.pt"))
            return

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
