"""OT-CFM loss: Optimal Transport Conditional Flow Matching.

Implements the OT-paired CFM objective from Lipman et al. (2022) and
Tong et al. (2023).  Mini-batch OT pairing via the Hungarian algorithm
aligns source (noise) and target (data) samples before computing the
conditional flow matching loss.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from cfm.model.vector_field import ConditionalVectorField


def ot_pair(x_0: Tensor, x_1: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mini-batch optimal transport pairing.

    Solves a linear assignment on the pairwise L2 cost matrix so that
    each noise sample x_0[i] is matched to the closest data sample x_1[j]
    (in the OT sense).

    Args:
        x_0: Source (noise) samples, shape (B, D).
        x_1: Target (data) samples, shape (B, D).

    Returns:
        Tuple of (x_0_paired, x_1_paired), each shape (B, D).
    """
    with torch.no_grad():
        cost = torch.cdist(x_0, x_1, p=2).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

    return x_0[row_ind], x_1[col_ind]


def cfm_loss(
    vector_field: ConditionalVectorField,
    x_1: Tensor,
    cond: Tensor,
    sigma_min: float = 1e-4,
) -> Tensor:
    """Compute the OT-CFM training loss.

    Samples noise x_0, pairs it with data x_1 via OT, builds the linear
    interpolant x_t, and regresses the predicted velocity toward the
    analytic conditional velocity u_t.

    Args:
        vector_field: The neural vector field network.
        x_1:   Target (data) samples, shape (B, 48).
        cond:  Conditioning scalars, shape (B, cond_dim).
        sigma_min: Minimum noise scale (controls path straightness).

    Returns:
        Scalar MSE loss.
    """
    assert x_1.ndim == 2 and cond.ndim == 2, f"Expected 2D tensors, got x_1={x_1.shape}, cond={cond.shape}"
    B, D = x_1.shape
    device = x_1.device

    # Sample noise and time
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, device=device)

    # OT pairing
    x_0, x_1 = ot_pair(x_0, x_1)

    # Linear interpolant: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
    t_col = t[:, None]
    x_t = (1.0 - (1.0 - sigma_min) * t_col) * x_0 + t_col * x_1

    # Analytic conditional velocity
    u_t = x_1 - (1.0 - sigma_min) * x_0

    # Predicted velocity
    v_t = vector_field(x_t, t, cond)

    return ((v_t - u_t) ** 2).mean()


# ---------------------------------------------------------------------------
# Brownian-bridge coarse-to-fine interpolation
# ---------------------------------------------------------------------------


def compute_block_waypoint(x_1: Tensor, block_size: int) -> Tensor:
    """Compute a piecewise-constant waypoint from ground truth *x_1*.

    Aggregates *x_1* into blocks of *block_size*, divides each block sum
    by *block_size*, and repeats each value *block_size* times.  The result
    is a piecewise-constant vector whose total sum equals that of *x_1*.

    Args:
        x_1: Ground truth data, shape ``(B, D)`` where ``D`` is divisible
            by *block_size*.
        block_size: Number of bars per block.

    Returns:
        Piecewise-constant waypoint, shape ``(B, D)``.
    """
    B, D = x_1.shape
    n_blocks = D // block_size
    # (B, n_blocks, block_size) -> sum -> (B, n_blocks)
    block_sums = x_1.reshape(B, n_blocks, block_size).sum(dim=2)
    block_means = block_sums / block_size
    return block_means.unsqueeze(2).expand(B, n_blocks, block_size).reshape(B, D)


def compute_bridge_schedule(
    intermediate_blocks: list[int],
    schedule: str = "uniform",
) -> tuple[list[int], list[float]]:
    """Compute sorted block sizes and time boundaries for bridge interpolation.

    Args:
        intermediate_blocks: Block sizes (e.g. ``[12, 6]``).  Sorted
            internally so that the coarsest (largest *block_size*) comes
            first.
        schedule: ``"uniform"`` divides ``[0, 1]`` into equal-length
            segments.

    Returns:
        ``(sorted_blocks, time_boundaries)`` where *sorted_blocks* is
        descending and *time_boundaries* has ``len(sorted_blocks) + 2``
        entries spanning ``[0, 1]``.
    """
    sorted_blocks = sorted(intermediate_blocks, reverse=True)
    n = len(sorted_blocks)
    if schedule == "uniform":
        time_boundaries = [i / (n + 1) for i in range(n + 2)]
    else:
        raise ValueError(f"Unknown bridge schedule: {schedule!r}")
    return sorted_blocks, time_boundaries


def bridge_cfm_loss(
    vector_field: ConditionalVectorField,
    x_1: Tensor,
    cond: Tensor,
    intermediate_blocks: list[int],
    sigma_min: float = 1e-4,
    bridge_time_schedule: str = "uniform",
) -> Tensor:
    """OT-CFM loss with coarse-to-fine bridge interpolation.

    Replaces the single linear interpolant with a piecewise linear path
    that passes through block-level waypoints at designated time
    boundaries.  Each waypoint is derived from *x_1* (today's ground
    truth) — no autoregression or lagged-day dependency is introduced.

    Args:
        vector_field: Neural vector field network.
        x_1: Target data, shape ``(B, 48)``.
        cond: Conditioning, shape ``(B, cond_dim)``.
        intermediate_blocks: Block sizes for waypoints (e.g. ``[12, 6]``).
        sigma_min: Minimum noise scale (applied to first segment only).
        bridge_time_schedule: Time allocation strategy.

    Returns:
        Scalar MSE loss.
    """
    assert x_1.ndim == 2 and cond.ndim == 2
    B, D = x_1.shape
    device = x_1.device

    # Sample noise and time
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, device=device)

    # OT pairing
    x_0, x_1 = ot_pair(x_0, x_1)

    # Build schedule and ordered waypoints
    sorted_blocks, time_bounds = compute_bridge_schedule(
        intermediate_blocks, bridge_time_schedule,
    )

    # waypoints: [x_0, coarsest_waypoint, ..., finest_waypoint, x_1]
    waypoints: list[Tensor] = [x_0]
    for bs in sorted_blocks:
        waypoints.append(compute_block_waypoint(x_1, bs))
    waypoints.append(x_1)

    # Determine segment membership and compute interpolant + velocity
    t_col = t[:, None]  # (B, 1)
    x_t = torch.zeros_like(x_1)
    u_t = torch.zeros_like(x_1)

    n_segments = len(time_bounds) - 1
    for seg_idx in range(n_segments):
        t_start = time_bounds[seg_idx]
        t_end = time_bounds[seg_idx + 1]
        seg_len = t_end - t_start

        # Mask: which batch elements fall in this segment
        if seg_idx == n_segments - 1:
            mask = (t >= t_start) & (t <= t_end)
        else:
            mask = (t >= t_start) & (t < t_end)

        if not mask.any():
            continue

        mask_col = mask[:, None]

        # Local time within segment, rescaled to [0, 1]
        s = ((t - t_start) / seg_len)[:, None]

        w_start = waypoints[seg_idx]
        w_end = waypoints[seg_idx + 1]

        if seg_idx == 0:
            # First segment (noise → coarsest waypoint): apply sigma_min
            local_x_t = (1.0 - (1.0 - sigma_min) * s) * w_start + s * w_end
            local_u_t = (w_end - (1.0 - sigma_min) * w_start) / seg_len
        else:
            # Later segments (waypoint → waypoint): deterministic
            local_x_t = (1.0 - s) * w_start + s * w_end
            local_u_t = (w_end - w_start) / seg_len

        x_t = torch.where(mask_col, local_x_t, x_t)
        u_t = torch.where(mask_col, local_u_t, u_t)

    # Predicted velocity
    v_t = vector_field(x_t, t, cond)

    return ((v_t - u_t) ** 2).mean()
