from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import make_deriv_loaders
from src.models.hnn import HNN, HNNConfig
from src.models.mlp import MLPVectorField, MLPConfig


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    model_name: str, hidden: int, layers: int, angle_embed: bool, device: torch.device
) -> nn.Module:
    if model_name == "hnn":
        model = HNN(HNNConfig(hidden=hidden, layers=layers, angle_embed=angle_embed))
    elif model_name == "mlp":
        model = MLPVectorField(
            MLPConfig(hidden=hidden, layers=layers, angle_embed=angle_embed)
        )
    else:
        raise ValueError(f"Unknown model '{model_name}, options are hnn or mlp.")
    return model.to(device=device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.1
    count = 0
    for x, xdot in loader:
        with torch.set_grad_enabled(True):
            pred = model(x)
        loss = loss_fn(pred, xdot)
        bs = x.shape[0]
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, xdot in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, xdot)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = x.shape[0]
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


def default_ckpt_path(model: str, sigma: float, outdir: str = "runs") -> str:
    os.makedirs(outdir, exist_ok=True)
    tag = f"{model}_sigma{sigma:.2f}"
    return os.path.join(outdir, f"{tag}.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised derivative training for HNN/MLP on pendulum."
    )
    # Data
    p.add_argument(
        "--data-dir", type=str, default="data", help="directory with .npz files"
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="noise level used for data file selection",
    )
    # Model
    p.add_argument("--model", type=str, choices=["hnn", "mlp"], default="hnn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument(
        "--angle-embed", action="store_true", help="use [sin q, cos q, p] inputs"
    )
    # Train
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    # Device
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--ckpt", type=str, default=None, help="override checkpoint path")
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32

    # Data
    train_loader, test_loader = make_deriv_loaders(
        data_dir=args.data_dir,
        sigma=args.sigma,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        device=device,
        dtype=dtype,
    )

    # Model + Optimizer
    model = build_model(args.model, args.hidden, args.layers, args.angle_embed, device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    # Checkpoint path + save config
    ckpt_path = args.ckpt or default_ckpt_path(args.model, args.sigma)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    config_path = ckpt_path.replace(".pt", ".json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "data_dir": args.data_dir,
                "sigma": args.sigma,
                "model": args.model,
                "hidden": args.hidden,
                "layers": args.layers,
                "angle_embed": bool(args.angle_embed),
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch": args.batch,
                "num_workers": args.num_workers,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "device": args.device,
                "ckpt": ckpt_path,
                "param_count": count_params(model),
            },
            f,
            indent=2,
        )

    # Initial evaluation
    best_test = evaluate(model, test_loader, loss_fn)
    print(f"[init] params={count_params(model):,}  test_mse={best_test:.6e}")

    # Train
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model, train_loader, opt, loss_fn, grad_clip=args.grad_clip
        )
        test_mse = evaluate(model, test_loader, loss_fn)

        if test_mse < best_test:
            best_test = test_mse
            torch.save(
                {"model": args.model, "state_dict": model.state_dict()}, ckpt_path
            )

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[epoch {epoch:4d}] "
                f"train_mse={train_mse:.6e}  test_mse={test_mse:.6e}  best={best_test:.6e}"
            )

    elapsed = time.time() - start_time
    print(
        f"Done. Best test MSE={best_test:.6e}  ckpt='{ckpt_path}'  elapsed={elapsed:.1f}s"
    )

    if args.sigma == 0.0 and args.model == "hnn":
        target = 1e-3
        if best_test <= target:
            print(f"[OK] Derivative MSE target met (≤ {target:g}).")
        else:
            print(
                f"[WARN] Derivative MSE target not met (>{target:g}). Consider more epochs/tuning."
            )


if __name__ == "__main__":
    main()
