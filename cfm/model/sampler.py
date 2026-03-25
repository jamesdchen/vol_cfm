"""ODE-based sampler for the trained conditional flow matching model.

Integrates the learned vector field from t=0 (noise) to t=1 (data) using
either torchdiffeq's adaptive ODE solver or a simple Euler fallback.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from cfm.model.flow_matching import compute_bridge_schedule
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
        bridge_blocks: list[int] | None = None,
        bridge_guidance_strength: float = 0.0,
    ) -> None:
        self.vector_field = vector_field
        self.solver = solver
        self.num_steps = num_steps
        self.bridge_blocks = bridge_blocks
        self.bridge_guidance_strength = bridge_guidance_strength

        if bridge_blocks:
            self._sorted_blocks, self._time_bounds = compute_bridge_schedule(bridge_blocks)
        else:
            self._sorted_blocks = []
            self._time_bounds = [0.0, 1.0]

    @torch.no_grad()
    def sample(self, cond: Tensor, num_samples: int = 1) -> Tensor:
        """Generate samples by integrating the vector field ODE.

        When *bridge_blocks* is configured the integration is split into
        segments at the bridge time boundaries.  Between segments an
        optional waypoint projection nudges the state toward block-level
        structure.

        Args:
            cond: Conditioning tensor, shape (B, cond_dim).
            num_samples: Unused (B is inferred from *cond*); kept for API
                compatibility.

        Returns:
            Generated samples at t=1, shape (B, 48).
        """
        self.vector_field.eval()
        B = cond.shape[0]
        device = cond.device
        output_dim = self.vector_field.output_dim

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
                    dt = (t_end - t_start) / seg_steps
                    for i in range(seg_steps):
                        t_val = t_start + i * dt
                        t_batch = torch.full((B,), t_val, device=device)
                        x = x + dt * self.vector_field(x, t_batch, cond)

                # Apply waypoint guidance between segments (not after last)
                if (
                    seg_idx < n_segments - 1
                    and self.bridge_guidance_strength > 0
                ):
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
            dt = 1.0 / self.num_steps
            x = x_0
            for i in range(self.num_steps):
                t_val = i * dt
                t_batch = torch.full((B,), t_val, device=device)
                x = x + dt * self.vector_field(x, t_batch, cond)
            x_1 = x

        return x_1

    @torch.no_grad()
    def sample_consistent(self, daily_rv: Tensor, context: Tensor) -> Tensor:
        """Generate intraday RV that sums to the observed daily RV.

        Builds the conditioning vector, samples raw network output, then
        maps to a valid simplex (softmax) and rescales by daily RV.

        Args:
            daily_rv: Scalar daily realized volatility per sample, shape (B,).
            context:  Recent context features, shape (B, cond_dim - 1).

        Returns:
            Intraday realized volatility, shape (B, 48), summing to
            *daily_rv* per row.
        """
        cond = torch.cat([daily_rv.unsqueeze(-1).sqrt(), context], dim=-1)
        raw = self.sample(cond)
        proportions = F.softmax(raw, dim=-1)  # (B, 48), positive & sums to 1
        intraday = proportions * daily_rv.unsqueeze(-1)
        return intraday
