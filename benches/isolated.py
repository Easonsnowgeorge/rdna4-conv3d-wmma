"""Isolate FWD / dW / dX against F.conv3d on the UNet trunk shapes."""
from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from rdna4_conv3d import wmma_dw, wmma_dx, wmma_fwd

SHAPES = [
    ("e1  1->24  16x32x32 k3", 2, 1, 24, 16, 32, 32, 3, 1),
    ("e1 24->24  16x32x32 k3", 2, 24, 24, 16, 32, 32, 3, 1),
    ("e2 48->48  8x16x16  k3", 2, 48, 48, 8, 16, 16, 3, 1),
    ("out 24->1  16x32x32 k1", 2, 24, 1, 16, 32, 32, 1, 0),
]


def _errs(a, b):
    da = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-3)
    return da.max().item(), da.mean().item(), (da / denom).max().item()


def _time(fn, repeats=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3


def main():
    if not torch.cuda.is_available():
        print("skip: no GPU", flush=True)
        return 0
    print(f"{'shape':<28} {'miF':>8} {'wmF':>8} {'xF':>6}  Ymax  dWmax  dXmax", flush=True)
    for name, n, cin, cout, d, h, w, k, pad in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(n, cin, d, h, w, device="cuda", dtype=torch.float16)
        wt = torch.randn(cout, cin, k, k, k, device="cuda", dtype=torch.float16)
        b = torch.randn(cout, device="cuda", dtype=torch.float16)
        y_mi = F.conv3d(x, wt, b, 1, pad)
        go = torch.randn_like(y_mi)
        y_wm = wmma_fwd(x, wt, b, 1, pad)
        dw_wm = wmma_dw(x, wt, go, 1, pad)
        dx_wm = wmma_dx(x, wt, go, 1, pad)
        x1 = x.detach().requires_grad_(True)
        w1 = wt.detach().requires_grad_(True)
        y1 = F.conv3d(x1, w1, None, 1, pad)
        y1.backward(go)
        y_e, dw_e, dx_e = _errs(y_wm, y_mi), _errs(dw_wm, w1.grad), _errs(dx_wm, x1.grad)
        mi_f = _time(lambda: F.conv3d(x, wt, b, 1, pad))
        wm_f = _time(lambda: wmma_fwd(x, wt, b, 1, pad))
        print(
            f"{name:<28} {mi_f:8.2f} {wm_f:8.2f} {mi_f / wm_f:6.2f}  "
            f"{y_e[0]:.1e} {dw_e[0]:.1e} {dx_e[0]:.1e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
