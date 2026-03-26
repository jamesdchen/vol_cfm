"""CFM configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

PERIODS_PER_DAY = 48
OUTPUT_DIM = 48
START_DATE = "2005-01-01"
FRIDAY_CLOSE = "20:00"
SUNDAY_OPEN = "18:30"


@dataclass
class CFMConfig:
    # Data
    harxhar_path: str = "data/all30min"
    output_dim: int = OUTPUT_DIM
    train_end: str = "2020-12-31"
    val_end: str = "2022-12-31"

    # Inpainting conditioning
    block_granularities: list[int] = field(default_factory=lambda: [12])
    sparse_bar_prob: float = 0.0
    mask_schedule: str = "random_blocks"  # "random_blocks" | "all_blocks"

    # Diurnal prior
    diurnal_prior: bool = True
    diurnal_prior_std: float = 1.0

    # Source-based split
    train_source: str = "vwstock_stats.parquet"
    val_source: str = "ewstock_stats.parquet"
    test_source: str = "core_stats.parquet"
    source_columns: dict[str, str] = field(
        default_factory=lambda: {
            "vwstock_stats.parquet": "sumret2_vwstock",
            "ewstock_stats.parquet": "sumret2_ewstock",
            "core_stats.parquet": "sumret2",
        }
    )

    # Model (vector field network)
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    time_embed_dim: int = 32
    cond_dim: int = field(init=False)
    sigma_min: float = 1e-4

    # Bridge interpolation (coarse-to-fine)
    bridge_interpolation: bool = False

    def __post_init__(self) -> None:
        for bs in self.block_granularities:
            if PERIODS_PER_DAY % bs != 0:
                raise ValueError(
                    f"Block size {bs} does not evenly divide {PERIODS_PER_DAY}. "
                    f"Valid sizes: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48."
                )
        if self.mask_schedule not in ("random_blocks", "all_blocks"):
            raise ValueError(f"Invalid mask_schedule: {self.mask_schedule!r}. Must be 'random_blocks' or 'all_blocks'.")
        if not 0.0 <= self.sparse_bar_prob <= 1.0:
            raise ValueError(f"sparse_bar_prob must be in [0, 1], got {self.sparse_bar_prob}")

        # Auto-detect num_workers
        if self.num_workers == 0:
            import os

            self.num_workers = min(os.cpu_count() or 4, 8)

        # cond_dim: mask(48) + known_values(48) + daily_rv(1)
        self.cond_dim = 2 * PERIODS_PER_DAY + 1

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

    # DataLoader throughput
    num_workers: int = 0  # 0 = auto-detect
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Infra
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
