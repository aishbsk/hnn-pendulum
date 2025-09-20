from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

Tensor = torch.Tensor

__all__ = ["MLPVectorField", "MLPConfig"]


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for a plain MLP vector field x->dx/dt."""

    hidden: int = 64
    layers: int = 2
    angle_embed: bool = False


class MLPVectorField(nn.Module):
    """Plain MLP baseline to predict derivates"""

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.cfg = cfg or MLPConfig()

        in_dim = 3 if self.cfg.angle_embed else 2
        h = self.cfg.hidden

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(self.cfg.layers):
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _phi(self, x: Tensor) -> Tensor:
        if not self.cfg.angle_embed:
            return x
        q = x[..., 0:1]
        p = x[..., 1:2]
        return torch.cat([torch.sin(q), torch.cos(q), p], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError(f"Expected last dim 2 for [q,p], {x.shape}")
        z = self._phi(x)
        dxdt = self.net(z)
        return dxdt
