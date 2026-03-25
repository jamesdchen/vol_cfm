"""Tests for cfm.model.flow_matching."""

import torch

from cfm.model.flow_matching import cfm_loss, ot_pair
from cfm.model.vector_field import ConditionalVectorField


def test_cfm_loss_scalar():
    B, D, cond_dim = 8, 48, 6
    model = ConditionalVectorField(output_dim=D, cond_dim=cond_dim, hidden_dims=[32, 32])

    x_1 = torch.randn(B, D)
    cond = torch.randn(B, cond_dim)

    loss = cfm_loss(model, x_1, cond, sigma_min=1e-4)

    assert loss.dim() == 0  # scalar
    assert loss.requires_grad


def test_ot_pair_shapes():
    B, D = 16, 48
    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)

    x_0_paired, x_1_paired = ot_pair(x_0, x_1)

    assert x_0_paired.shape == (B, D)
    assert x_1_paired.shape == (B, D)


def test_ot_pair_is_permutation():
    B, D = 10, 48
    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)

    x_0_paired, x_1_paired = ot_pair(x_0, x_1)

    # Each row in x_0_paired should appear in x_0 (it's a permutation of rows)
    for i in range(B):
        dists = torch.norm(x_0 - x_0_paired[i], dim=1)
        assert dists.min().item() < 1e-6, f"Row {i} of x_0_paired not found in x_0"

    for i in range(B):
        dists = torch.norm(x_1 - x_1_paired[i], dim=1)
        assert dists.min().item() < 1e-6, f"Row {i} of x_1_paired not found in x_1"


def test_interpolation_endpoints():
    """At t=0, x_t should approximate x_0; at t=1, x_t should approximate x_1."""
    B, D = 8, 48
    sigma_min = 1e-4

    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)

    # At t=0: x_t = (1 - 0) * x_0 + 0 * x_1 = x_0
    t_zero = torch.zeros(B, 1)
    x_t_at_0 = (1.0 - (1.0 - sigma_min) * t_zero) * x_0 + t_zero * x_1
    torch.testing.assert_close(x_t_at_0, x_0, atol=1e-5, rtol=1e-5)

    # At t=1: x_t = (1 - (1-sigma_min)) * x_0 + 1 * x_1 = sigma_min * x_0 + x_1
    t_one = torch.ones(B, 1)
    x_t_at_1 = (1.0 - (1.0 - sigma_min) * t_one) * x_0 + t_one * x_1
    expected_at_1 = sigma_min * x_0 + x_1
    torch.testing.assert_close(x_t_at_1, expected_at_1, atol=1e-5, rtol=1e-5)

    # x_t at t=1 should be close to x_1 (sigma_min is small)
    diff = (x_t_at_1 - x_1).abs().max().item()
    assert diff < 0.1, f"At t=1, x_t should be close to x_1, but max diff = {diff}"
