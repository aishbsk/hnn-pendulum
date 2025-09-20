from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Tuple, Dict

import numpy as np
import torch

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import after adding to sys.path
from src.physics import pendulum_deriv, rollout_rk4


def sample_initial_states(
    n: int,
    q_range: Tuple[float, float] = (-math.pi, math.pi),
    p_range: Tuple[float, float] = (-2.0, -2.0),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    q = (q_range[1] - q_range[0]) * torch.rand(
        n, 1, device=device, dtype=dtype
    ) + q_range[0]
    p = (p_range[1] - p_range[0]) * torch.rand(
        n, 1, device=device, dtype=dtype
    ) + p_range[0]
    x0 = torch.cat([q, p], dim=1)
    return x0


@torch.no_grad()
def make_split(
    n_traj: int,
    steps: int,
    dt: float,
    eps: float,
    sigma: float,
    seed: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, np.ndarray]:
    """
    Generate a split (train or test).

    Returns a dict with:
      - x:      [n_traj, T, 2] observed states (noise added if sigma>0)
      - xdot:   [n_traj, T, 2] analytic derivatives evaluated at observed states
      - t:      [T] time stamps
      - dt:     scalar step (float)
      - meta:   JSON string with parameters
    """
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    # Time axis
    T = steps
    # rollout_rk4 returns (steps+1) states; use steps-1 so final length is T
    # but we want length=steps exactly per spec (e.g., 200). We'll do steps-1 inside rollout and keep T.
    rollout_steps = T - 1
    t = torch.arange(T, device=device, dtype=dtype) * dt

    # Sample initial conditions
    x0 = sample_initial_states(n_traj, device=device, dtype=dtype)  # [N,2]

    # Clean rollout (no measurement noise injected here)
    states = rollout_rk4(
        pendulum_deriv, x0, dt=dt, steps=rollout_steps, eps=eps, return_derivs=False
    )  # [T, N, 2]
    states = states.transpose(0, 1).contiguous()  # [N, T, 2]

    # Observation noise
    if sigma > 0:
        noise = sigma * torch.randn_like(states)
        x_obs = states + noise
    else:
        x_obs = states

    # Supervision targets: derivatives from the analytic system at the *observed* states
    # (This gives a clean and a noisy variant depending on sigma.)
    xdot = pendulum_deriv(x_obs, eps=eps)

    # Package as numpy
    out = {
        "x": x_obs.cpu().numpy(),
        "xdot": xdot.cpu().numpy(),
        "t": t.cpu().numpy(),
        "dt": np.array(dt, dtype=np.float32),
        "meta": np.array(
            json.dumps(
                {
                    "n_traj": int(n_traj),
                    "steps": int(steps),
                    "dt": float(dt),
                    "eps": float(eps),
                    "sigma": float(sigma),
                    "seed": int(seed),
                    "dtype": str(dtype).replace("torch.", ""),
                },
                indent=2,
            )
        ),
    }
    return out


def save_npz(path: str, payload: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pendulum datasets (train/test) as .npz files."
    )
    p.add_argument(
        "--n-train", type=int, default=200, help="number of training trajectories"
    )
    p.add_argument("--n-test", type=int, default=50, help="number of test trajectories")
    p.add_argument(
        "--length",
        type=int,
        default=200,
        help="trajectory length T (number of time steps)",
    )
    p.add_argument("--dt", type=float, default=0.05, help="time step")
    p.add_argument(
        "--eps", type=float, default=0.0, help="linear damping coefficient ε"
    )
    p.add_argument(
        "--noise",
        type=str,
        default="0.0,0.01",
        help="comma-separated list of measurement noise stddevs (e.g., '0.0,0.01')",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument(
        "--outdir", type=str, default="data", help="output directory for .npz files"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    dtype = torch.float32

    # Parse noise list
    noise_levels = [float(s.strip()) for s in args.noise.split(",") if s.strip() != ""]
    if len(noise_levels) == 0:
        noise_levels = [0.0]

    for sigma in noise_levels:
        # Train split
        train = make_split(
            n_traj=args.n_train,
            steps=args.length,
            dt=args.dt,
            eps=args.eps,
            sigma=sigma,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        train_path = os.path.join(args.outdir, f"pendulum_train_sigma{sigma:.2f}.npz")
        save_npz(train_path, train)

        # Test split (different seed to avoid leakage)
        test = make_split(
            n_traj=args.n_test,
            steps=args.length,
            dt=args.dt,
            eps=args.eps,
            sigma=sigma,
            seed=args.seed + 1,
            device=device,
            dtype=dtype,
        )
        test_path = os.path.join(args.outdir, f"pendulum_test_sigma{sigma:.2f}.npz")
        save_npz(test_path, test)

        print(
            f"Wrote:\n  {train_path}  (x: {train['x'].shape}, xdot: {train['xdot'].shape})\n"
            f"  {test_path}   (x: {test['x'].shape}, xdot: {test['xdot'].shape})"
        )


if __name__ == "__main__":
    main()
