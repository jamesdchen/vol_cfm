"""CFM training loop: epoch-level train/validate/fit with checkpointing."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from cfm.cli import get_device
from cfm.config import CFMConfig
from cfm.data.dataset import build_inpainting_dataloaders
from cfm.logging import get_logger
from cfm.model.flow_matching import bridge_cfm_loss, cfm_loss
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
        self.train_loader, self.val_loader, self.test_loader, self.metadata = build_inpainting_dataloaders(
            config.harxhar_path, config
        )

        # Diurnal prior: convert to tensor on device if enabled, else None
        if config.diurnal_prior and self.metadata.get("diurnal_mean") is not None:
            self.diurnal_mean = torch.tensor(self.metadata["diurnal_mean"], dtype=torch.float32, device=self.device)
        else:
            self.diurnal_mean = None

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

    def _compute_loss(self, x_1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Dispatch to bridge or standard CFM loss based on config."""
        if self.config.bridge_interpolation:
            return bridge_cfm_loss(
                self.model,
                x_1,
                cond,
                intermediate_blocks=self.config.block_granularities,
                sigma_min=self.config.sigma_min,
            )
        return cfm_loss(
            self.model,
            x_1,
            cond,
            self.config.sigma_min,
            prior_mean=self.diurnal_mean,
            prior_std=self.config.diurnal_prior_std,
        )

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_1, cond in self.train_loader:
            x_1 = x_1.to(self.device)
            cond = cond.to(self.device)

            self.optimizer.zero_grad()
            loss = self._compute_loss(x_1, cond)
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

        for x_1, cond in self.val_loader:
            x_1 = x_1.to(self.device)
            cond = cond.to(self.device)

            loss = self._compute_loss(x_1, cond)
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
        diurnal_np = self.diurnal_mean.cpu().numpy() if self.diurnal_mean is not None else None
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
                "scaler_stats": {},  # kept for backward compat
                "diurnal_mean": diurnal_np,
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

        # Restore diurnal mean if present in checkpoint
        diurnal_np = ckpt.get("diurnal_mean", None)
        if diurnal_np is not None:
            self.diurnal_mean = torch.tensor(diurnal_np, dtype=torch.float32, device=self.device)
        else:
            self.diurnal_mean = None
