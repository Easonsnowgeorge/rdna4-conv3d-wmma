"""Tiny correctness + one UNet-shaped forward timing. Needs a GPU."""
from __future__ import annotations

import sys
import time
import traceback

import torch
import torch.nn.functional as F

from rdna4_conv3d import conv3d_wmma


def _check(n, cin, cout, d, h, w, k=3, pad=1):
    torch.manual_seed(0)
    xr = torch.randn(n, cin, d, h, w, device="cuda", dtype=torch.float32)
    wr = torch.randn(cout, cin, k, k, k, device="cuda", dtype=torch.float32)
    br = torch.randn(cout, device="cuda", dtype=torch.float32)
    y_fp32 = F.conv3d(xr, wr, br, 1, pad)
    x = xr.half().detach().requires_grad_(True)
    wt = wr.half().detach().requires_grad_(True)
    b = br.half().detach().requires_grad_(True)
    y = conv3d_wmma(x, wt, b, 1, pad)
    x2 = xr.half().detach().requires_grad_(True)
    w2 = wr.half().detach().requires_grad_(True)
    b2 = br.half().detach().requires_grad_(True)
    y2 = F.conv3d(x2, w2, b2, 1, pad)
    gy = torch.randn_like(y2)
    y2.backward(gy)
    y.backward(gy)
    print(
        "n{} c{}->{} {}x{}x{} k{}: vsfp32 {:.3e} vsfp16 {:.3e} dX {:.3e} dW {:.3e} dB {:.3e}".format(
            n, cin, cout, d, h, w, k,
            (y.float() - y_fp32).abs().max().item(),
            (y.float() - y2.float()).abs().max().item(),
            (x.grad.float() - x2.grad.float()).abs().max().item(),
            (wt.grad.float() - w2.grad.float()).abs().max().item(),
            (b.grad.float() - b2.grad.float()).abs().max().item(),
        ),
        flush=True,
    )


def _bench_fwd(n, cin, cout, d, h, w, k=3, pad=1, repeats=10):
    x = torch.randn(n, cin, d, h, w, device="cuda", dtype=torch.float16)
    wt = torch.randn(cout, cin, k, k, k, device="cuda", dtype=torch.float16)
    b = torch.randn(cout, device="cuda", dtype=torch.float16)

    def time_fn(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats * 1e3

    mi = time_fn(lambda: F.conv3d(x, wt, b, 1, pad))
    wm = time_fn(lambda: conv3d_wmma(x, wt, b, 1, pad))
    print(
        "fwd ms  n{} {}->{} {}x{}x{}: stock {:.2f}  WMMA {:.2f}  {:.2f}x".format(
            n, cin, cout, d, h, w, mi, wm, mi / wm if wm else 0.0
        ),
        flush=True,
    )


def main():
    if not torch.cuda.is_available():
        print("SMOKE_SKIP: no GPU", flush=True)
        return 0
    p = torch.cuda.get_device_properties(0)
    print(
        "device: {} | {} | {:.1f} GB".format(
            p.name, getattr(p, "gcnArchName", "?"), p.total_memory / 1e9
        ),
        flush=True,
    )
    try:
        _check(2, 5, 7, 9, 11, 13, k=3, pad=1)
        _check(2, 24, 1, 8, 16, 16, k=1, pad=0)
        _check(1, 24, 24, 16, 32, 32, k=3, pad=1)
        _bench_fwd(1, 24, 24, 16, 32, 32)
        print("SMOKE_OK", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("SMOKE_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
