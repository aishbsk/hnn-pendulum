from __future__ import annotations

from typing import Callable, Tuple
import torch

Tensor = torch.Tensor

__all__ = [
    "pendulum_deriv",
    "energy",
    "rk4_step",
    "rollout_rk4",
]


def pendulum_deriv(
    x: Tensor,
    eps: float | Tensor = 0.0,
    g: float | Tensor = 9.81,
    L: float | Tensor = 1.0,
    m: float | Tensor = 1.0,
) -> Tensor:
    if x.shape[-1] != 2:
        raise ValueError(f"Expected dimension 2 for [q, p], got {x.shape}")
    device, dtype = x.device, x.dtype
    eps = torch.as_tensor(eps, device=device, dtype=dtype)
    g = torch.as_tensor(g, device=device, dtype=dtype)
    L = torch.as_tensor(L, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)

    q = x[..., 0:1]
    p = x[..., 1:2]

    dqdt = p / (m * (L**2))
    dpdt = -m * g * L * torch.sin(q) - eps * p
    return torch.cat([dqdt, dpdt], dim=-1)


def energy(
    x: Tensor,
    g: float | Tensor = 9.81,
    L: float | Tensor = 1.0,
    m: float | Tensor = 1.0,
) -> Tensor:
    """
    Mechanical energy H(q, p) = T(p) + V(q) of the pendulum.

    T = p^2 / (2 m L^2)
    V = mgL (1 - cos q)
    """
    if x.shape[-1] != 2:
        raise ValueError(f"Expected dimension 2 for [q, p], got {x.shape}")
    device, dtype = x.device, x.dtype
    g = torch.as_tensor(g, device=device, dtype=dtype)
    L = torch.as_tensor(L, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)

    q = x[..., 0:1]
    p = x[..., 1:2]

    T = (p**2) / (2.0 * m * L**2)
    V = m * g * L * (1.0 - torch.cos(q))

    return (T + V).squeeze(-1)


def rk4_step(
    f: Callable[..., Tensor],
    x: Tensor,
    dt: float | Tensor,
    **f_kwargs,
) -> Tensor:
    device, dtype = x.device, x.dtype
    dt = torch.as_tensor(dt, device=device, dtype=dtype)

    k1 = f(x, **f_kwargs)
    k2 = f(x + 0.5 * dt * k1, **f_kwargs)
    k3 = f(x + 0.5 * dt * k2, **f_kwargs)
    k4 = f(x + dt * k3, **f_kwargs)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def rollout_rk4(
    f: Callable[..., Tensor],
    x0: Tensor,
    dt: float,
    steps: int,
    *,
    return_derivs: bool = False,
    **f_kwargs,
) -> Tuple[Tensor, Tensor] | Tensor:
    # Rolls out a trajectory with RK4 for a vector field
    if steps < 0:
        raise ValueError("Steps must be >= 0")
    batch_shape = x0.shape[:-1]
    D = x0.shape[-1]
    device, dtype = x0.device, x0.dtype

    # Initialize state tensor
    states = torch.empty((steps + 1, *batch_shape, D), device=device, dtype=dtype)
    states[0] = x0

    if return_derivs:
        derivs = torch.empty_like(states)
        derivs[0] = f(x0, **f_kwargs)
        x = x0
        for t in range(1, steps + 1):
            x = rk4_step(f, x, dt, **f_kwargs)
            states[t] = x
            derivs[t] = f(x, **f_kwargs)
        return states, derivs
    else:
        x = x0
        for t in range(1, steps + 1):
            x = rk4_step(f, x, dt, **f_kwargs)
            states[t] = x
        return states
