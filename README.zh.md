# rdna4-conv3d-wmma

AMD RDNA4（`gfx1201`，含 Radeon AI PRO R9700 / RX 9070 XT）上的 `nn.Conv3d` 替换件。

MIOpen 没有 gfx12 的 3D find-db，默认卷积走启发式核，用不上 16×16×16 WMMA。本库把 k=1 / k=3、stride 1 的卷积映射到这些矩阵核上。

R9700 上 UNet3D pool2 b4 AMP：MIOpen **~287 ms/sample** → WMMA **61 / 48 ms/sample**（base 24/32），同机同脚本快过 RTX 3070。

```python
from rdna4_conv3d import Conv3dWmma, swap_conv3d
swap_conv3d(model)
```

几何不支持时回落 `F.conv3d`。完整说明见 [README.md](README.md)，测量过程见 [docs/DESIGN.md](docs/DESIGN.md)。

MIT。与任何竞赛仓库无关。
