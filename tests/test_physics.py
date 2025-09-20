from __future__ import annotations

import math
import torch

from src.physics import pendulum_deriv, energy, rollout_rk4


def test_pendulum_deriv_shape_and_values():
    torch.manual_seed(0)
    x = torch.randn(32, 2, dtype=torch.float32)
    x[:, 0].clamp_(-math.pi, math.pi)  # q in [-pi, pi]

    xdot = pendulum_deriv(x, eps=0.0)
    assert xdot.shape == x.shape
    assert torch.isfinite(xdot).all()

    # small angle; approximately simple harmonic
    small = torch.zeros(1, 2)
    small[0, 0] = 1e-4  # tiny angle
    small_xdot = pendulum_deriv(small)
    # dp/dt ≈ - m g L * q  (with m=L=1)
    assert torch.allclose(
        small_xdot[0, 1], torch.tensor(-9.81e-4), rtol=1e-3, atol=1e-6
    )


def test_rollout_shapes_and_derivs_flag():
    torch.manual_seed(1)
    B, steps, dt = 5, 20, 0.05
    x0 = torch.randn(B, 2, dtype=torch.float32)

    states = rollout_rk4(pendulum_deriv, x0, dt=dt, steps=steps, eps=0.0)
    if isinstance(states, tuple):
        states = states[0]
    assert states.shape == (steps + 1, B, 2)

    states2, derivs2 = rollout_rk4(
        pendulum_deriv, x0, dt=dt, steps=steps, return_derivs=True, eps=0.0
    )
    assert states2.shape == (steps + 1, B, 2)
    assert derivs2.shape == (steps + 1, B, 2)
    assert torch.isfinite(states2).all() and torch.isfinite(derivs2).all()


def test_energy_conservation_rk4_eps0_long_rollout():
    """
    Acceptance-style check:
      With ε=0, RK4 energy drift should be <1% over 200 steps (dt=0.05).
    """
    torch.manual_seed(42)
    B = 128
    steps = 200
    dt = 0.05

    # Sample initial states across a broad range
    q0 = (2 * math.pi) * torch.rand(B, 1, dtype=torch.float64) - math.pi
    p0 = 4.0 * torch.rand(B, 1, dtype=torch.float64) - 2.0
    x0 = torch.cat([q0, p0], dim=1)

    states = rollout_rk4(pendulum_deriv, x0, dt=dt, steps=steps, eps=0.0)
    if isinstance(states, tuple):
        states = states[0]
    E = energy(states)  # [T+1, B]

    E0 = E[0]  # [B]
    # Filter out near-zero energies where relative error is ill-defined
    mask = E0 > 0.05
    assert mask.any(), "Expected at least some trajectories with non-trivial energy."

    # Relative drift over time for each trajectory
    rel = (E[:, mask] - E0[mask]) / E0[mask]
    max_rel = rel.abs().max().item()
    assert max_rel < 0.01, f"Energy drift {max_rel:.4f} exceeded 1%."

    # no NaNs/infs anywhere
    assert torch.isfinite(E).all()
