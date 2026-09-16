"""Drop-in 3D convolution for AMD RDNA4 (gfx1200/gfx1201) via Triton WMMA.

On gfx120x, ``nn.Conv3d`` through MIOpen often misses the matrix cores because
there is no 3D find-db. This package maps k=1 and k=3 stride-1 convolutions
onto 16x16x16 Wave32 WMMA tiles.

Unsupported geometries and non-RDNA4 GPUs fall back to ``torch.nn.functional.conv3d``.
"""

from rdna4_conv3d.conv3d_wmma import (
    Conv3dWmma,
    ConvTranspose3dWmma,
    conv3d_plan,
    conv3d_wmma,
    swap_conv3d,
    wmma_dw,
    wmma_dx,
    wmma_fwd,
    wmma_ok_dw,
    wmma_ok_dx,
    wmma_ok_fwd,
)

__version__ = "0.1.0"
__all__ = [
    "Conv3dWmma",
    "ConvTranspose3dWmma",
    "conv3d_plan",
    "conv3d_wmma",
    "swap_conv3d",
    "wmma_dw",
    "wmma_dx",
    "wmma_fwd",
    "wmma_ok_dw",
    "wmma_ok_dx",
    "wmma_ok_fwd",
]
