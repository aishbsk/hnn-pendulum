from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


import torch
import torch.nn as nn

Tensor = torch.Tensor

__all__ = ["HNN", "HNNConfig"]


@dataclass(frozen=True)
class HNNConfig:
    hidden: int = 64
    layers: int = 2
    angle_embed: bool = False


class HNN(nn.Module):
    """
    Hamiltonian Neural Network for single pendulum

    Forward:
    x -> f_theta(x) = [ dq/dt, dp/dt ] via Hamilton's equations:
        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q
    """

    def __init__(self, cfg: Optional[HNNConfig] = None):
        super().__init__()
        self.cfg = cfg or HNNConfig()

        in_dim = 3 if self.cfg.angle_embed else 2

        layers: list[nn.Module] = []
        h = self.cfg.hidden
        d = in_dim
        for _ in range(self.cfg.layers):
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

        # Lightweight init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _phi(self, x: Tensor) -> Tensor:
        """Optional input embedding. x: (...,2) -> (...,in_dim)."""
        if not self.cfg.angle_embed:
            return x
        q = x[..., 0:1]
        p = x[..., 1:2]
        return torch.cat([torch.sin(q), torch.cos(q), p], dim=-1)

    def hamiltonian(self, x: Tensor) -> Tensor:
        """
        Compute H_theta(x) per-sample (no batch reduction).

        Args:
            x: (..., 2) with [q, p]

        Returns:
            H: (...,) scalar energy estimate per sample.
        """
        z = self._phi(x)
        H = self.net(z).squeeze(-1)
        return H

    def forward(self, x: Tensor) -> Tensor:
        """
        Vector field f_theta(x) = [dqdt, dpdt].

        Args:
            x: (batch, 2) or (..., 2)

        Returns:
            dxdt: same leading shape as x, last dim=2
        """
        if x.shape[-1] != 2:
            raise ValueError(f"Expected last dim 2 for [q,p], got {x.shape}")

        # Store original grad state to restore it later
        grad_enabled = torch.is_grad_enabled()

        # Always enable gradients for autograd.grad, even in eval mode
        with torch.enable_grad():
            # Clone and prepare input for gradient computation
            x_with_grad = x.clone().detach().to(x.device)
            x_with_grad.requires_grad_(True)

            # Compute Hamiltonian
            H = self.hamiltonian(x_with_grad)
            H_sum = H.sum()

            # Compute gradients of Hamiltonian w.r.t. inputs
            # create_graph only needed during training
            create_graph = grad_enabled and self.training
            dHdx = torch.autograd.grad(H_sum, x_with_grad, create_graph=create_graph)[0]

        # Apply Hamilton's equations
        dqdt = dHdx[..., 1:2]  # ∂H/∂p
        dpdt = -dHdx[..., 0:1]  # -∂H/∂q
        return torch.cat([dqdt, dpdt], dim=-1)
