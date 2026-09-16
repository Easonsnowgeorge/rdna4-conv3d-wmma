"""One AMP training step of TinyUNet3D: stock Conv3d vs Conv3dWmma."""
from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

from rdna4_conv3d import Conv3dWmma, ConvTranspose3dWmma


def _block(cin, cout, conv):
    return nn.Sequential(
        conv(cin, cout, 3, padding=1),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        conv(cout, cout, 3, padding=1),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, conv, tr, base=16):
        super().__init__()
        self.e1 = _block(1, base, conv)
        self.down = nn.MaxPool3d(2)
        self.e2 = _block(base, base * 2, conv)
        self.up = tr(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base, conv)
        self.out = conv(base, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.down(a))
        u = self.up(b)
        return self.out(self.d1(torch.cat([u, a], dim=1)))


def _step(net, x, repeats=8):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    def go():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            y = net(x)
            loss = loss_fn(y, torch.zeros_like(y))
        loss.backward()
        opt.step()
        return float(loss.detach())

    for _ in range(3):
        go()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last = 0.0
    for _ in range(repeats):
        last = go()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3, last


def main():
    if not torch.cuda.is_available():
        print("skip: no GPU", flush=True)
        return 0
    x = torch.randn(2, 1, 16, 32, 32, device="cuda")
    stock = TinyUNet(nn.Conv3d, nn.ConvTranspose3d).cuda()
    wmma = TinyUNet(Conv3dWmma, ConvTranspose3dWmma).cuda()
    wmma.load_state_dict(stock.state_dict())
    ms_s, loss_s = _step(stock, x)
    ms_w, loss_w = _step(wmma, x)
    print(
        f"stock {ms_s:.1f} ms/step loss={loss_s:.4f} | "
        f"WMMA {ms_w:.1f} ms/step loss={loss_w:.4f} | "
        f"{ms_s / ms_w:.2f}x",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
