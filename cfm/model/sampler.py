"""ODE-based sampler for the trained conditional flow matching model.

Integrates the learned vector field from t=0 (noise) to t=1 (data) using
either torchdiffeq's adaptive ODE solver or a simple Euler fallback.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from cfm.model.vector_field import ConditionalVectorField

try:
    from torchdiffeq import odeint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class CFMSampler:
    """Sample from the learned OT-CFM generative model.

    Attributes:
        vector_field: Trained conditional vector field network.
        solver: ODE solver method for torchdiffeq (default ``"dopri5"``).
        num_steps: Number of Euler steps when torchdiffeq is unavailable.
    """

    def __init__(
        self,
        vector_field: ConditionalVectorField,
        solver: str = "dopri5",
        num_steps: int = 100,
    ) -> None:
        self.vector_field = vector_field
        self.solver = solver
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(self, cond: Tensor, num_samples: int = 1) -> Tensor:
        """Generate samples by integrating the vector field ODE.

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

        if HAS_TORCHDIFFEQ:
            # Define dynamics for torchdiffeq: f(t, x) -> dx/dt
            def ode_fn(t_scalar: Tensor, x: Tensor) -> Tensor:
                t_batch = t_scalar.expand(B)
                return self.vector_field(x, t_batch, cond)

            t_span = torch.tensor([0.0, 1.0], device=device)
            trajectory = odeint(ode_fn, x_0, t_span, method=self.solver)
            x_1 = trajectory[-1]  # state at t=1
        else:
            # Euler integration fallback
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
