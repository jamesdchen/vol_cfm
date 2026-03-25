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

    # Intermediate conditioning
    intermediate_blocks: list[int] = field(default_factory=list)
    intermediate_representation: str = "sqrt"  # "sqrt" | "proportion" | "both"

    # Model (vector field network)
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    time_embed_dim: int = 32
    cond_dim: int = field(init=False)
    sigma_min: float = 1e-4

    def __post_init__(self) -> None:
        # Validate intermediate block sizes
        for bs in self.intermediate_blocks:
            if PERIODS_PER_DAY % bs != 0:
                raise ValueError(
                    f"Block size {bs} does not evenly divide {PERIODS_PER_DAY}. "
                    f"Valid sizes: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48."
                )
        if self.intermediate_representation not in ("sqrt", "proportion", "both"):
            raise ValueError(
                f"Invalid intermediate_representation: {self.intermediate_representation!r}. "
                f"Must be 'sqrt', 'proportion', or 'both'."
            )

        # Compute cond_dim: daily features + intermediate features from lagged days
        base = self.context_days + 1
        n_intermediate = sum(PERIODS_PER_DAY // bs for bs in self.intermediate_blocks)
        multiplier = 2 if self.intermediate_representation == "both" else 1
        self.cond_dim = base + n_intermediate * multiplier * self.context_days

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
