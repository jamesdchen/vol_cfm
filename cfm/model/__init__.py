"""CFM model components."""

from __future__ import annotations

import numpy as np
import torch

from cfm.config import CFMConfig
from cfm.model.vector_field import ConditionalVectorField


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> tuple[ConditionalVectorField, CFMConfig, np.ndarray | None]:
    """Load model, config, and diurnal mean from a training checkpoint.

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
    config : CFMConfig
        Experiment configuration stored in the checkpoint.
    diurnal_mean : numpy.ndarray | None
        Per-period mean proportions (48,) used as the sampling prior,
        or None if the checkpoint was created without a diurnal prior.

    Notes
    -----
    Uses ``weights_only=False`` because checkpoints contain a pickled
    :class:`CFMConfig` dataclass.  Only load checkpoints you trust.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Backward compat: old checkpoints lack inpainting config fields
    if not hasattr(config, "block_granularities"):
        config.block_granularities = [12]
    if not hasattr(config, "sparse_bar_prob"):
        config.sparse_bar_prob = 0.0
    if not hasattr(config, "mask_schedule"):
        config.mask_schedule = "random_blocks"
    if not hasattr(config, "diurnal_prior"):
        config.diurnal_prior = True
    if not hasattr(config, "diurnal_prior_std"):
        config.diurnal_prior_std = 1.0
    if not hasattr(config, "cond_dim"):
        config.cond_dim = 2 * 48 + 1
    if not hasattr(config, "bridge_interpolation"):
        config.bridge_interpolation = False
    if not hasattr(config, "log_time_beta"):
        config.log_time_beta = 0.0

    model = ConditionalVectorField(
        output_dim=config.output_dim,
        cond_dim=config.cond_dim,
        hidden_dims=config.hidden_dims,
        time_embed_dim=config.time_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diurnal_mean = ckpt.get("diurnal_mean", None)

    return model, config, diurnal_mean
