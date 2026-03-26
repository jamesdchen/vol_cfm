"""ODE-based sampler for the trained conditional flow matching model.

Integrates the learned vector field from t=0 (prior) to t=1 (data) using
either torchdiffeq's adaptive ODE solver or a simple Euler fallback.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from cfm.model.flow_matching import compute_bridge_schedule, log_time_grid
from cfm.model.vector_field import ConditionalVectorField

try:
    from torchdiffeq import odeint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


def _block_project(x: Tensor, block_size: int) -> Tensor:
    """Project *x* onto piecewise-constant block structure."""
    B, D = x.shape
    n_blocks = D // block_size
    block_means = x.reshape(B, n_blocks, block_size).mean(dim=2)
    return block_means.unsqueeze(2).expand(B, n_blocks, block_size).reshape(B, D)


class CFMSampler:
    """Sample from the learned OT-CFM generative model.

    Attributes:
        vector_field: Trained conditional vector field network.
        solver: ODE solver method for torchdiffeq (default ``"dopri5"``).
        num_steps: Number of Euler steps when torchdiffeq is unavailable.
        diurnal_mean: Mean of the source distribution, shape (48,). None for N(0,I).
        diurnal_std: Std of the source distribution.
        bridge_blocks: If set, enables segmented ODE integration with
            optional waypoint guidance at time boundaries.
        bridge_guidance_strength: Interpolation weight toward block-projected
            state at each waypoint boundary (0 = disabled).
    """

    def __init__(
        self,
        vector_field: ConditionalVectorField,
        solver: str = "dopri5",
        num_steps: int = 100,
        diurnal_mean: Tensor | None = None,
        diurnal_std: float = 1.0,
        bridge_blocks: list[int] | None = None,
        bridge_guidance_strength: float = 0.0,
        log_time_beta: float = 0.0,
    ) -> None:
        self.vector_field = vector_field
        self.solver = solver
        self.num_steps = num_steps
        self.diurnal_mean = diurnal_mean
        self.diurnal_std = diurnal_std
        self.bridge_blocks = bridge_blocks
        self.bridge_guidance_strength = bridge_guidance_strength
        self.log_time_beta = log_time_beta

        if bridge_blocks:
            self._sorted_blocks, self._time_bounds = compute_bridge_schedule(bridge_blocks)
        else:
            self._sorted_blocks = []
            self._time_bounds = [0.0, 1.0]

    @torch.no_grad()
    def sample(self, cond: Tensor) -> Tensor:
        """Generate raw samples by integrating the vector field ODE.

        When *bridge_blocks* is configured the integration is split into
        segments at the bridge time boundaries.  Between segments an
        optional waypoint projection nudges the state toward block-level
        structure.

        Args:
            cond: Conditioning tensor, shape (B, 97).

        Returns:
            Generated samples at t=1, shape (B, 48).
        """
        self.vector_field.eval()
        B = cond.shape[0]
        device = cond.device
        output_dim = self.vector_field.output_dim

        # Sample from prior
        if self.diurnal_mean is not None:
            mean = self.diurnal_mean.to(device)
            x_0 = mean.unsqueeze(0) + self.diurnal_std * torch.randn(B, output_dim, device=device)
        else:
            x_0 = torch.randn(B, output_dim, device=device)

        def ode_fn(t_scalar: Tensor, x: Tensor) -> Tensor:
            t_batch = t_scalar.expand(B)
            return self.vector_field(x, t_batch, cond)

        if self.bridge_blocks:
            # Segmented integration with optional waypoint guidance
            x = x_0
            n_segments = len(self._time_bounds) - 1

            for seg_idx in range(n_segments):
                t_start = self._time_bounds[seg_idx]
                t_end = self._time_bounds[seg_idx + 1]

                if HAS_TORCHDIFFEQ:
                    t_span = torch.tensor([t_start, t_end], device=device)
                    traj = odeint(ode_fn, x, t_span, method=self.solver)
                    x = traj[-1]
                else:
                    seg_steps = max(1, int(self.num_steps * (t_end - t_start)))
                    if self.log_time_beta > 0.0:
                        seg_grid = log_time_grid(seg_steps, self.log_time_beta, device)
                        seg_grid = t_start + (t_end - t_start) * seg_grid
                    else:
                        seg_grid = torch.linspace(t_start, t_end, seg_steps + 1, device=device)
                    for i in range(seg_steps):
                        t_val = seg_grid[i].item()
                        dt = (seg_grid[i + 1] - seg_grid[i]).item()
                        t_batch = torch.full((B,), t_val, device=device)
                        x = x + dt * self.vector_field(x, t_batch, cond)

                # Apply waypoint guidance between segments (not after last)
                if seg_idx < n_segments - 1 and self.bridge_guidance_strength > 0:
                    bs = self._sorted_blocks[seg_idx]
                    projected = _block_project(x, bs)
                    alpha = self.bridge_guidance_strength
                    x = (1.0 - alpha) * x + alpha * projected

            x_1 = x
        elif HAS_TORCHDIFFEQ:
            t_span = torch.tensor([0.0, 1.0], device=device)
            trajectory = odeint(ode_fn, x_0, t_span, method=self.solver)
            x_1 = trajectory[-1]
        else:
            # Use log-time grid when beta > 0 so the Euler solver spends
            # more steps near t=1 (matching the training distribution).
            if self.log_time_beta > 0.0:
                t_grid = log_time_grid(self.num_steps, self.log_time_beta, device)
            else:
                t_grid = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device)
            x = x_0
            for i in range(self.num_steps):
                t_val = t_grid[i].item()
                dt = (t_grid[i + 1] - t_grid[i]).item()
                t_batch = torch.full((B,), t_val, device=device)
                x = x + dt * self.vector_field(x, t_batch, cond)
            x_1 = x

        return x_1

    @torch.no_grad()
    def sample_proportions(self, cond: Tensor) -> Tensor:
        """Generate valid proportion vectors (positive, sum to 1).

        Args:
            cond: Conditioning tensor, shape (B, 97).

        Returns:
            Proportion vectors, shape (B, 48).
        """
        raw = self.sample(cond)
        return F.softmax(raw, dim=-1)
