"""Conditional vector field network for OT-CFM volatility generation."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion time t.

    Maps scalar time values to a higher-dimensional representation using
    geometric frequency bands (half sin, half cos), following the standard
    transformer positional encoding scheme.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed scalar time values.

        Args:
            t: Time values, shape (B,).

        Returns:
            Time embeddings, shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # If dim is odd, pad with a zero column
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ConditionalVectorField(nn.Module):
    """Neural network parameterising the conditional vector field v(x_t, t, cond).

    Architecture:
        concat(x_t, time_embed(t), cond)
        → [Linear → LayerNorm → SiLU → Dropout] × len(hidden_dims)
        → Linear → residual add with x_t

    Used inside the OT-CFM loss and ODE sampler to predict the velocity
    that transports noise x_0 to data x_1.
    """

    def __init__(
        self,
        output_dim: int = 48,
        cond_dim: int = 6,
        hidden_dims: list[int] | None = None,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        input_dim = output_dim + time_embed_dim + cond_dim

        layers: list[nn.Module] = []
        in_features = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = h

        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, output_dim)

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """Predict the velocity field at (x_t, t) given conditioning.

        Args:
            x_t:  Noisy / interpolated state, shape (B, 48).
            t:    Diffusion time, shape (B,).
            cond: Conditioning scalars, shape (B, cond_dim).

        Returns:
            Predicted velocity, shape (B, 48).
        """
        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        h = torch.cat([x_t, t_emb, cond], dim=-1)  # (B, input_dim)
        h = self.net(h)
        out = self.head(h)  # (B, output_dim)
        return out + x_t  # residual connection
