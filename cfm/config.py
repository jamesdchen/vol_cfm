"""CFM configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

PERIODS_PER_DAY = 48
CONTEXT_DAYS = 5
OUTPUT_DIM = 48
START_DATE = "2005-01-01"
FRIDAY_CLOSE = "20:00"
SUNDAY_OPEN = "18:30"

@dataclass
class CFMConfig:
    # Data
    harxhar_path: str = "data/all30min"
    context_days: int = CONTEXT_DAYS
    output_dim: int = OUTPUT_DIM
    train_end: str = "2020-12-31"
    val_end: str = "2022-12-31"

    # Model (vector field network)
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    time_embed_dim: int = 32
    cond_dim: int = CONTEXT_DAYS + 1  # daily RV today + N days context
    sigma_min: float = 1e-4

    # Training
    batch_size: int = 256
    num_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 5.0
    seed: int = 42
    checkpoint_every: int = 10

    # Sampling
    num_ode_steps: int = 100
    solver: str = "dopri5"

    # Infra
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
