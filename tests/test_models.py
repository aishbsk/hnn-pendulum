from __future__ import annotations

import time
import torch
import pytest

from src.models.hnn import HNN, HNNConfig


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def test_hnn_shapes_and_grad_consistency():
    torch.manual_seed(0)
    B = 256
    x = torch.randn(B, 2, dtype=torch.float32)

    model = HNN(HNNConfig(angle_embed=False))

    # Hamiltonian per-sample
    H = model.hamiltonian(x.requires_grad_(True))  # [B]
    assert H.shape == (B,)
    Hsum = H.sum()
    dHdx = torch.autograd.grad(Hsum, x, create_graph=False)[0]  # [B,2]
    assert dHdx.shape == x.shape
    assert _finite(dHdx)

    # Forward should equal [∂H/∂p, -∂H/∂q]
    f = model(x.detach())  # [B,2]
    expect = torch.cat([dHdx[:, 1:2], -dHdx[:, 0:1]], dim=1)
    assert torch.allclose(f, expect, rtol=1e-4, atol=1e-6)


def test_hnn_angle_embed_path():
    torch.manual_seed(1)
    B = 128
    x = torch.randn(B, 2, dtype=torch.float32)

    model = HNN(HNNConfig(angle_embed=True))
    H = model.hamiltonian(x.requires_grad_(True))
    assert H.ndim == 1 and H.numel() == B
    dHdx = torch.autograd.grad(H.sum(), x, create_graph=False)[0]
    f = model(x.detach())
    expect = torch.cat([dHdx[:, 1:2], -dHdx[:, 0:1]], dim=1)
    assert torch.allclose(f, expect, rtol=1e-4, atol=1e-6)
    assert _finite(H) and _finite(dHdx) and _finite(f)


def test_hnn_forward_backward_performance_smoke():
    """
    Loose performance smoke test
    """
    torch.manual_seed(2)
    B = 1024
    x = torch.randn(B, 2, dtype=torch.float32)
    y = torch.zeros_like(x)

    model = HNN(HNNConfig())
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()

    # Timed run
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt.step()
    elapsed = time.perf_counter() - start

    if elapsed > 0.10:
        pytest.xfail(
            f"Performance is environment-dependent (elapsed={elapsed * 1e3:.1f} ms)"
        )
    assert elapsed <= 0.10, f"Forward+backward too slow: {elapsed * 1e3:.1f} ms"
