"""Cosine annealing learning rate scheduler with linear warmup."""

import math


class CosineWarmupScheduler:
    """Cosine annealing LR with linear warmup.

    During warmup (step < warmup_steps): LR = base_lr * step / warmup_steps.
    After warmup: cosine decay from base_lr to 0.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total number of training steps (warmup + cosine decay).
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        """Call after each optimizer step. Updates LR."""
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _compute_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(self.base_lr * step / self.warmup_steps)
        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 0:
            return float(self.base_lr)
        progress = (step - self.warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        return float(self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def get_lr(self) -> float:
        """Current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])
