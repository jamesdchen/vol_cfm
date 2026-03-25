"""CFM model components."""

from __future__ import annotations

import torch

from cfm.config import CFMConfig
from cfm.model.vector_field import ConditionalVectorField


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> tuple[ConditionalVectorField, dict, CFMConfig]:
    """Load model, scaler stats, and config from a training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pt`` checkpoint saved by :class:`CFMTrainer`.
    device : torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    model : ConditionalVectorField
        Model with loaded weights, in eval mode on *device*.
    scaler_stats : dict
        Per-column mean/std used during training.
    config : CFMConfig
        Experiment configuration stored in the checkpoint.

    Notes
    -----
    Uses ``weights_only=False`` because checkpoints contain a pickled
    :class:`CFMConfig` dataclass.  Only load checkpoints you trust.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Backward compat: old checkpoints lack intermediate block fields
    if not hasattr(config, "intermediate_blocks"):
        config.intermediate_blocks = []
    if not hasattr(config, "intermediate_representation"):
        config.intermediate_representation = "sqrt"
    if not hasattr(config, "cond_dim"):
        config.cond_dim = config.context_days + 1

    model = ConditionalVectorField(
        output_dim=config.output_dim,
        cond_dim=config.cond_dim,
        hidden_dims=config.hidden_dims,
        time_embed_dim=config.time_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["scaler_stats"], config
