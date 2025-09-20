from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


__all__ = [
    "DerivDataset",
    "default_split_paths",
    "make_deriv_loaders",
]


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        x = z["x"]  # [N, T, 2]
        xdot = z["xdot"]  # [N, T, 2]
        t = z["t"]  # [T]
        dt = float(z["dt"].item()) if np.ndim(z["dt"]) else float(z["dt"])
        meta = z["meta"].item() if np.ndim(z["meta"]) else str(z["meta"])
    return {"x": x, "xdot": xdot, "t": t, "dt": dt, "meta": meta}


@dataclass(frozen=True)
class DerivDatasetConfig:
    """Configuration for DerivDataset."""

    paths: Sequence[str]
    device: torch.device | None = None
    dtype: torch.dtype = torch.float32
    flatten: bool = True  # if True: return per-time-step samples; else per-trajectory
    max_samples: int | None = None  # in flattened mode: cap total (N*T)
    max_traj: int | None = None  # in non-flattened mode: cap number of trajectories


class DerivDataset(Dataset):
    """
    Supervised (x -> xdot) dataset loaded from one or more .npz files.

    Each .npz must contain:
      - x:    [N, T, 2] states (q, p)
      - xdot: [N, T, 2] analytic derivatives at those states
      - t:    [T] time stamps
      - dt:   scalar
      - meta: string (JSON)

    In flattened mode (default), samples are (x[i], xdot[i]) with shape [2],
    where i indexes over N*T across all files.

    In non-flattened mode, samples are sequences (x_seq, xdot_seq) with shape [T, 2].
    """

    def __init__(self, cfg: DerivDatasetConfig):
        super().__init__()
        self.cfg = cfg
        device = cfg.device or torch.device("cpu")

        # Load and concatenate along trajectory axis
        xs, xds, ts = [], [], []
        self.dts: List[float] = []
        self.metas: List[str] = []

        for p in cfg.paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Dataset file not found: {p}")
            d = _load_npz(p)
            if d["x"].ndim != 3 or d["x"].shape != d["xdot"].shape:
                raise ValueError(
                    f"Inconsistent shapes in {p}: x {d['x'].shape}, xdot {d['xdot'].shape}"
                )
            xs.append(torch.from_numpy(d["x"]))
            xds.append(torch.from_numpy(d["xdot"]))
            ts.append(torch.from_numpy(d["t"]))
            self.dts.append(float(d["dt"]))
            self.metas.append(str(d["meta"]))

        # Ensure consistent T across files
        T_set = {int(t.numel()) for t in ts}
        if len(T_set) != 1:
            raise ValueError(f"All files must share the same T; got {sorted(T_set)}")
        self.T: int = T_set.pop()

        # Concatenate trajectories
        self.x = torch.cat(xs, dim=0).to(
            device=device, dtype=cfg.dtype
        )  # [N_total, T, 2]
        self.xdot = torch.cat(xds, dim=0).to(
            device=device, dtype=cfg.dtype
        )  # [N_total, T, 2]

        if cfg.flatten:
            # Merge (N, T) -> (N*T,)
            N_total = self.x.shape[0]
            X = self.x.reshape(N_total * self.T, 2)
            Y = self.xdot.reshape(N_total * self.T, 2)

            if cfg.max_samples is not None and cfg.max_samples < X.shape[0]:
                X = X[: cfg.max_samples]
                Y = Y[: cfg.max_samples]

            self.X = X
            self.Y = Y
            self.length = X.shape[0]
        else:
            # Keep per-trajectory sequences
            if cfg.max_traj is not None and cfg.max_traj < self.x.shape[0]:
                self.x = self.x[: cfg.max_traj]
                self.xdot = self.xdot[: cfg.max_traj]
            self.length = self.x.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.flatten:
            return self.X[idx], self.Y[idx]  # [2], [2]
        else:
            return self.x[idx], self.xdot[idx]  # [T,2], [T,2]


def default_split_paths(data_dir: str = "data", sigma: float = 0.0) -> tuple[str, str]:
    """
    Convenience helper returning (train_path, test_path) for a given sigma.
    Matches the naming from scripts/gen_data.py.
    """
    train = os.path.join(data_dir, f"pendulum_train_sigma{sigma:.2f}.npz")
    test = os.path.join(data_dir, f"pendulum_test_sigma{sigma:.2f}.npz")
    return train, test


def make_deriv_loaders(
    data_dir: str = "data",
    sigma: float = 0.0,
    *,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders for supervised derivative fitting.

    Args:
        data_dir: Folder with the .npz splits.
        sigma: Noise level used when generating files.
        batch_size: Batch size for loaders.
        shuffle: Shuffle train set.
        num_workers: DataLoader workers (set >0 for speed if desired).
        pin_memory: Pin memory (useful when training on GPU).
        device, dtype: Move tensors to device/dtype upon load.

    Returns:
        (train_loader, test_loader)
    """
    train_path, test_path = default_split_paths(data_dir, sigma)

    train_ds = DerivDataset(
        DerivDatasetConfig(
            paths=[train_path],
            device=device,
            dtype=dtype,
            flatten=True,
        )
    )
    test_ds = DerivDataset(
        DerivDatasetConfig(
            paths=[test_path],
            device=device,
            dtype=dtype,
            flatten=True,
        )
    )

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        drop_last=False,
    )
    return train_loader, test_loader
