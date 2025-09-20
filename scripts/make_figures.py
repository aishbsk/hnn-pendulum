from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def angle_wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def energy_np(
    x: np.ndarray, g: float = 9.81, L: float = 1.0, m: float = 1.0
) -> np.ndarray:
    """
    Mechanical energy for pendulum states.
    x: [N, T, 2] with [..., 0]=q, [..., 1]=p
    Returns: [N, T] energies.
    """
    q = x[..., 0]
    p = x[..., 1]
    T = (p**2) / (2.0 * m * L * L)
    V = m * g * L * (1.0 - np.cos(q))
    return T + V


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Returns (x, xdot, t, dt, meta_json_str)
      x:    [N, T, 2]
      xdot: [N, T, 2]
      t:    [T]
      dt:   float
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find file: {path}")
    with np.load(path, allow_pickle=False) as z:
        x = z["x"]
        xdot = z["xdot"]
        t = z["t"]
        dt = float(z["dt"].item()) if np.ndim(z["dt"]) else float(z["dt"])
        meta = z["meta"].item() if np.ndim(z["meta"]) else str(z["meta"])
    return x, xdot, t, dt, meta


def plot_q_overlay(
    x: np.ndarray, out_path: str, n_show: int = 5, title: str = "q(t) overlay (data)"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N, T, _ = x.shape
    idx = np.arange(min(n_show, N))
    t = np.arange(T)

    plt.figure(figsize=(7, 4.2), dpi=140)
    for i in idx:
        q = angle_wrap(x[i, :, 0])
        plt.plot(t, q, lw=1)
    plt.xlabel("Step")
    plt.ylabel("q (rad)")
    plt.title(title + f" — first {len(idx)} trajs")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_p_overlay(
    x: np.ndarray, out_path: str, n_show: int = 5, title: str = "p(t) overlay (data)"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N, T, _ = x.shape
    idx = np.arange(min(n_show, N))
    t = np.arange(T)

    plt.figure(figsize=(7, 4.2), dpi=140)
    for i in idx:
        p = x[i, :, 1]
        plt.plot(t, p, lw=1)
    plt.xlabel("Step")
    plt.ylabel("p")
    plt.title(title + f" — first {len(idx)} trajs")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_phase_space(
    x: np.ndarray, out_path: str, n_show: int = 8, title: str = "Phase space (q vs p)"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N, T, _ = x.shape
    idx = np.arange(min(n_show, N))

    plt.figure(figsize=(5.5, 5.5), dpi=140)
    for i in idx:
        q = angle_wrap(x[i, :, 0])
        p = x[i, :, 1]
        plt.plot(q, p, lw=1)
    plt.xlabel("q (rad)")
    plt.ylabel("p")
    plt.title(title + f" — first {len(idx)} trajs")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_energy(
    x: np.ndarray,
    out_path: str,
    n_show: int = 5,
    title: str = "Energy over time (data)",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N, T, _ = x.shape
    idx = np.arange(min(n_show, N))
    t = np.arange(T)

    E = energy_np(x)  # [N, T]

    plt.figure(figsize=(7, 4.2), dpi=140)
    for i in idx:
        plt.plot(t, E[i], lw=1)
    plt.xlabel("Step")
    plt.ylabel("H(q,p)")
    plt.title(title + f" — first {len(idx)} trajs")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize pendulum .npz dataset.")
    ap.add_argument(
        "--data-file",
        type=str,
        default="data/pendulum_train_sigma0.00.npz",
        help="Path to a dataset .npz (train or test).",
    )
    ap.add_argument(
        "--outdir", type=str, default="reports", help="Where to write figures."
    )
    ap.add_argument(
        "--n-show", type=int, default=5, help="How many trajectories to overlay."
    )
    args = ap.parse_args()

    x, xdot, t, dt, meta = load_npz(args.data_file)
    N, T, D = x.shape
    print(f"Loaded {args.data_file}")
    print(f"Shapes: x={x.shape}, xdot={xdot.shape}, t={t.shape}, dt={dt}")
    print(f"Meta: {meta}")

    plot_q_overlay(
        x, os.path.join(args.outdir, "data_q_overlay.png"), n_show=args.n_show
    )
    plot_p_overlay(
        x, os.path.join(args.outdir, "data_p_overlay.png"), n_show=args.n_show
    )
    plot_phase_space(
        x, os.path.join(args.outdir, "data_phase_space.png"), n_show=max(8, args.n_show)
    )
    plot_energy(x, os.path.join(args.outdir, "data_energy.png"), n_show=args.n_show)

    print("Saved figures to:", args.outdir)


if __name__ == "__main__":
    main()
