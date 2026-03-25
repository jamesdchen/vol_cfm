"""Tests for cfm.model.vector_field."""

import torch
import pytest

from cfm.model.vector_field import ConditionalVectorField, SinusoidalTimeEmbedding


def test_vector_field_output_shape():
    B, output_dim, cond_dim = 16, 48, 6
    model = ConditionalVectorField(
        output_dim=output_dim, cond_dim=cond_dim, hidden_dims=[64, 64]
    )
    model.eval()

    x_t = torch.randn(B, output_dim)
    t = torch.rand(B)
    cond = torch.randn(B, cond_dim)

    out = model(x_t, t, cond)
    assert out.shape == (B, output_dim)


def test_vector_field_residual():
    """Verify the network contributes beyond the residual identity."""
    B, output_dim, cond_dim = 8, 48, 6
    model = ConditionalVectorField(
        output_dim=output_dim, cond_dim=cond_dim, hidden_dims=[64, 64]
    )
    model.eval()

    x_t = torch.zeros(B, output_dim)
    t = torch.full((B,), 0.5)
    cond = torch.randn(B, cond_dim)

    out = model(x_t, t, cond)
    # With x_t=0, residual adds 0, so output == head(net(input))
    # This should be non-zero because cond and t embed are non-zero
    assert out.abs().sum().item() > 0


def test_time_embedding_shape():
    embed_dim = 32
    B = 8
    emb = SinusoidalTimeEmbedding(embed_dim)

    t = torch.rand(B)
    out = emb(t)
    assert out.shape == (B, embed_dim)
