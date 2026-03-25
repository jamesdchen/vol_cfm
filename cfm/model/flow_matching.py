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
