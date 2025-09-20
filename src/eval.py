from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.physics import pendulum_deriv, energy, rollout_rk4
from src.data import default_split_paths
from src.models.hnn import HNN, HNNConfig
from src.models.mlp import MLPVectorField, MLPConfig


Tensor = torch.Tensor


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def angle_wrap(x: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi]."""
    two_pi = torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype)
    pi = torch.tensor(math.pi, device=x.device, dtype=x.dtype)
    return torch.remainder(x + pi, two_pi) - pi


def state_diff(a: Tensor, b: Tensor) -> Tensor:
    """
    Difference between states a and b with angle wrap for q.
    a,b: (..., 2)
    returns: (..., 2) where [dq_wrapped, dp]
    """
    dq = angle_wrap(a[..., 0] - b[..., 0]).unsqueeze(-1)
    dp = (a[..., 1] - b[..., 1]).unsqueeze(-1)
    return torch.cat([dq, dp], dim=-1)


def load_test_initial_states(
    data_dir: str,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    """
    Load test split and return (x0, dt).
    x0 shape: [N, 2]  (first state of each trajectory)
    """
    _, test_path = default_split_paths(data_dir, sigma)
    with np.load(test_path, allow_pickle=False) as z:
        x = torch.from_numpy(z["x"]).to(device=device, dtype=dtype)  # [N, T, 2]
        dt = float(z["dt"])
    x0 = x[:, 0, :]
    return x0, dt


def build_model(
    name: str, hidden: int, layers: int, angle_embed: bool, device: torch.device
):
    if name == "hnn":
        model = HNN(HNNConfig(hidden=hidden, layers=layers, angle_embed=angle_embed))
    elif name == "mlp":
        model = MLPVectorField(
            MLPConfig(hidden=hidden, layers=layers, angle_embed=angle_embed)
        )
    else:
        raise ValueError("model must be 'hnn' or 'mlp'")
    return model.to(device)


@torch.no_grad()
def rollout_model(model: torch.nn.Module, x0: Tensor, dt: float, steps: int) -> Tensor:
    """
    Roll out a model with RK4 given initial states x0.
    Returns states of shape [steps+1, N, 2].
    """
    model.eval()
    with torch.enable_grad():
        result = rollout_rk4(lambda x: model(x), x0, dt=dt, steps=steps)
    if isinstance(result, tuple):
        return result[0]
    return result


def metrics_rollout(
    pred: Tensor,  # [T+1, N, 2]
    gt: Tensor,  # [T+1, N, 2]
) -> Dict[str, float]:
    """
    Compute rollout metrics: MSE over (q,p) with angular wrap on q,
    plus ADE/FDE on q only (angles).
    """
    diff = state_diff(pred, gt)  # [T+1, N, 2]
    mse = (diff**2).mean().item()

    ang_diff = angle_wrap(pred[..., 0] - gt[..., 0])
    ade = ang_diff.abs().mean().item()
    fde = ang_diff[-1].abs().mean().item()
    return {"rollout_mse": mse, "ade_q": ade, "fde_q": fde}


def energy_drift_stats(states: Tensor) -> Tuple[Tensor, float, float]:
    """
    Compute relative energy drift time series and summary stats.
    Returns (rel_series [T+1,N], mean_abs_final, max_abs_over_time).
    """
    E = energy(states)  # [T+1, N]
    E0 = E[0]  # [N]
    # Avoid divide-by-zero: mask tiny energies
    mask = E0 > 1e-6
    rel = torch.zeros_like(E)
    rel[:, mask] = (E[:, mask] - E0[mask]) / E0[mask]
    mean_abs_final = rel[-1, mask].abs().mean().item() if mask.any() else float("nan")
    max_abs_over_time = rel[:, mask].abs().max().item() if mask.any() else float("nan")
    return rel, mean_abs_final, max_abs_over_time


def plot_energy_rel(
    rel_list: list[Tensor],
    labels: list[str],
    out_path: str,
    title: str = "Relative Energy Drift over Time",
):
    """
    Plot median ± IQR of relative energy drift (per model).
    rel_list entries: [T+1, N]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4.2), dpi=140)
    for rel, lab in zip(rel_list, labels):
        # median and IQR across N
        med = rel.median(dim=1).values.cpu().numpy()  # [T+1]
        q25 = rel.quantile(0.25, dim=1).cpu().numpy()  # [T+1]
        q75 = rel.quantile(0.75, dim=1).cpu().numpy()  # [T+1]
        t = np.arange(rel.shape[0])
        plt.plot(t, med, label=lab)
        plt.fill_between(t, q25, q75, alpha=0.2)
    plt.axhline(0.0, lw=1, ls="--")
    plt.xlabel("Step")
    plt.ylabel("Relative drift (E_t - E_0) / E_0")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_q_overlay(
    pred: Tensor,
    gt: Tensor,
    out_path: str,
    n_show: int = 5,
    title: str = "q(t) overlay",
):
    """
    Overlay a few angle trajectories for qualitative comparison.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Tp1, N, _ = pred.shape
    idx = torch.arange(min(n_show, N))
    t = np.arange(Tp1)

    plt.figure(figsize=(7, 4.2), dpi=140)
    for i in idx:
        q_pred = angle_wrap(pred[:, i, 0]).cpu().numpy()
        q_gt = angle_wrap(gt[:, i, 0]).cpu().numpy()
        plt.plot(t, q_gt, lw=2, alpha=0.8)
        plt.plot(t, q_pred, lw=1, ls="--", alpha=0.9)
    plt.xlabel("Step")
    plt.ylabel("q (rad)")
    plt.title(title + f" (first {len(idx)} trajs)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def default_ckpt_path(model: str, sigma: float, runs_dir: str = "runs") -> str:
    return os.path.join(runs_dir, f"{model}_sigma{sigma:.2f}.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate long-horizon rollouts, energy drift, and plots."
    )
    # Data / rollout
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument(
        "--dt", type=float, default=None, help="override dt; default uses dataset dt"
    )
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    # Model A 
    p.add_argument("--model", choices=["hnn", "mlp"], default="hnn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--angle-embed", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    # Model B (baseline for comparison on plot)
    p.add_argument(
        "--compare-mlp",
        action="store_true",
        help="also evaluate an MLP for comparison plot",
    )
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--mlp-angle-embed", action="store_true")
    p.add_argument("--mlp-ckpt", type=str, default=None)
    # Device
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--outdir", type=str, default="reports")
    return p.parse_args()


def load_ckpt(model: torch.nn.Module, ckpt_path: str) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32

    # Data initial states
    x0, dt_data = load_test_initial_states(args.data_dir, args.sigma, device, dtype)
    dt = float(args.dt) if args.dt is not None else dt_data
    steps = int(args.horizon)

    # Ground truth rollout
    gt_result = rollout_rk4(
        pendulum_deriv, x0, dt=dt, steps=steps, eps=0.0
    )  # [T+1, N, 2] or tuple
    gt = gt_result[0] if isinstance(gt_result, tuple) else gt_result

    # Primary model
    model = build_model(args.model, args.hidden, args.layers, args.angle_embed, device)
    ckpt = args.ckpt or default_ckpt_path(args.model, args.sigma)
    load_ckpt(model, ckpt)
    pred = rollout_model(model, x0, dt=dt, steps=steps)

    # Metrics for primary
    m = metrics_rollout(pred, gt)
    rel, mean_abs_final, max_abs_over = energy_drift_stats(pred)
    m.update(
        {
            "energy_drift_final_abs_mean": mean_abs_final,
            "energy_drift_abs_max": max_abs_over,
        }
    )

    # MLP Comparison model
    rel_list = [rel]
    labels = [args.model.upper()]
    if args.compare_mlp:
        mlp = build_model(
            "mlp", args.mlp_hidden, args.mlp_layers, args.mlp_angle_embed, device
        )
        mlp_ckpt = args.mlp_ckpt or default_ckpt_path("mlp", args.sigma)
        load_ckpt(mlp, mlp_ckpt)
        pred_mlp = rollout_model(mlp, x0, dt=dt, steps=steps)
        rel_mlp, mean_abs_final_mlp, max_abs_over_mlp = energy_drift_stats(pred_mlp)
        m["mlp_energy_drift_final_abs_mean"] = mean_abs_final_mlp
        m["mlp_energy_drift_abs_max"] = max_abs_over_mlp
        
        plot_q_overlay(
            pred_mlp,
            gt,
            os.path.join(args.outdir, "q_overlay_mlp.png"),
            title="q(t) overlay: MLP vs Ground Truth",
        )

        rel_list.append(rel_mlp)
        labels.append("MLP")

        # Rollout MSE for MLP
        mm = metrics_rollout(pred_mlp, gt)
        m["mlp_rollout_mse"] = mm["rollout_mse"]
        m["mlp_ade_q"] = mm["ade_q"]
        m["mlp_fde_q"] = mm["fde_q"]

    # Plots for primary
    os.makedirs(args.outdir, exist_ok=True)
    plot_energy_rel(rel_list, labels, os.path.join(args.outdir, "energy_rel.png"))
    plot_q_overlay(
        pred,
        gt,
        os.path.join(args.outdir, "q_overlay.png"),
        title=f"q(t) overlay: {args.model.upper()} vs Ground Truth",
    )

    # Save numeric summary
    summary = {
        "model": args.model,
        "sigma": args.sigma,
        "dt": dt,
        "horizon": steps,
        **m,
    }
    with open(os.path.join(args.outdir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(json.dumps(summary, indent=2))

    # Soft acceptance hints
    if args.model == "hnn":
        if summary["energy_drift_final_abs_mean"] <= 0.05:
            print("[OK] Energy drift ≤ 5% over horizon.")
        else:
            print(
                "[WARN] Energy drift > 5%; consider longer training or symplectic integrator for rollouts."
            )
        if summary["rollout_mse"] <= 0.02:
            print("[OK] Rollout MSE ≤ 0.02 over 10s horizon.")
        else:
            print("[WARN] Rollout MSE > 0.02; consider tuning or angle embedding.")


if __name__ == "__main__":
    main()
