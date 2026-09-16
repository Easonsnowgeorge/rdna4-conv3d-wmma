"""Minimal 3D UNet using Conv3dWmma. For illustration, not a trained model."""
import torch
import torch.nn as nn

from rdna4_conv3d import Conv3dWmma, ConvTranspose3dWmma, swap_conv3d


def conv_block(cin, cout):
    return nn.Sequential(
        Conv3dWmma(cin, cout, 3, padding=1),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        Conv3dWmma(cout, cout, 3, padding=1),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class TinyUNet3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.e1 = conv_block(1, base)
        self.e2 = conv_block(base, base * 2)
        self.pool = nn.MaxPool3d(2)
        self.up = ConvTranspose3dWmma(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.out = Conv3dWmma(base, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        u = self.up(b)
        cat = torch.cat([u, a], dim=1)
        return self.out(self.d1(cat))


def from_stock_unet(stock: nn.Module) -> nn.Module:
    n = swap_conv3d(stock)
    print(f"swapped {n} Conv3d modules")
    return stock


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TinyUNet3D().to(device)
    x = torch.randn(1, 1, 16, 32, 32, device=device)
    with torch.autocast("cuda", enabled=device == "cuda"):
        y = net(x)
    print("out", tuple(y.shape), y.dtype)
