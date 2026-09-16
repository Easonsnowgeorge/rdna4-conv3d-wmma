# rdna4-conv3d-wmma

Drop-in `nn.Conv3d` / `nn.ConvTranspose3d` for **AMD RDNA4** (`gfx1200` / `gfx1201`, including Radeon AI PRO R9700 and RX 9070 XT).

MIOpen ships no 3D find-db for gfx12, so stock PyTorch `Conv3d` falls through to heuristic kernels and never uses the 16×16×16 WMMA matrix cores. This package maps k=1 and k=3 stride-1 convolutions onto those cores via Triton `tl.dot`.

On an R9700, a UNet3D (`base=24/32`, pool2 `64×128×128`, batch 4, AMP, full fwd+bwd+Adam) went from **~287 ms/sample (MIOpen)** to **61 / 48 ms/sample**, and ended faster than an RTX 3070 on the same box and script.

## Install

```bash
# ROCm PyTorch + Triton already installed
pip install -e .
```

Requires: Python ≥ 3.10, `torch` with ROCm, Triton 3.x. Non-RDNA4 GPUs run the stock PyTorch path.

## Usage

```python
import torch.nn as nn
from rdna4_conv3d import Conv3dWmma, swap_conv3d

# Drop-in layer
conv = Conv3dWmma(24, 24, 3, padding=1)

# Or patch an existing model (copies weights)
n = swap_conv3d(model)
print(f"swapped {n} Conv3d modules")
```

Unsupported geometries (`groups != 1`, stride ≠ 1, dilation ≠ 1, kernel not 1 or 3) fall back to `F.conv3d`.

```python
from rdna4_conv3d import conv3d_plan
print(conv3d_plan(24, 24, 3))   # {'fwd': 'wmma', 'dw': 'wmma', 'dx': 'wmma'} on gfx1201
```

Environment:

| Variable | Values | Default |
|---|---|---|
| `RDNA4_CONV3D_BACKEND` | `hybrid` / `wmma` / `miopen` / `hip` | `hybrid` |
| `BIOHUB_CONV3D_BACKEND` | same | fallback alias |

`hip` tries an optional rocWMMA fprop for 3×3×3 s1 only; backward still uses Triton.

## What is in scope

- Forward, `dW`, `dX` for `Conv3d` k=1 and k=3, stride 1, dilation 1, groups 1
- `ConvTranspose3d` k=2 s=2 pad=0 (UNet upsample)
- fp16 / bf16 compute, fp32 accumulate
- Autocast: custom autograd Functions bypass ATen, so the package casts **outside** the Function the same way `nn.Conv3d` does. Forgetting this sent BN/ReLU/Pool back to fp32 and cost ~70 ms/step plus 2× activation memory.

Out of scope (falls back to MIOpen): other kernels, grouped conv, stride 2 downsampling convs.

## Measured (R9700, UNet3D pool2 b4 AMP, exclusive GPU)

| | ms/step | ms/sample | vs MIOpen |
|---|---:|---:|---|
| MIOpen base=24 | ~1170 | ~287 | 1.0× |
| WMMA base=24 | **243** | **61** | **4.8×** |
| WMMA base=32 | **191** | **48** | **8.1×** |
| RTX 3070 + channels_last, base=24 | 258 | 64 | — |

40 AMP steps: `max|Δloss|` ≈ 3–4×10⁻⁴ vs MIOpen (usable for training, not bitwise identical). Weight gradients can miss a tight `assert_close` because of large-K reduction order; loss tracks.

## Tests and benches

```bash
pytest                      # CPU / non-RDNA4: plan + fallback. gfx1201: numerical checks
python benches/smoke.py     # tiny correctness + one UNet-shaped fwd timing (needs GPU)
python benches/isolated.py  # FWD / dW / dX vs F.conv3d on the UNet trunk shapes
python benches/unet_e2e.py  # small UNet3D step vs nn.Conv3d
```

## Design notes

See [docs/DESIGN.md](docs/DESIGN.md) for the measurement log: why channels_last and conv2d-decomposition failed, why isolated kernel timings cannot drive dispatch, and why a custom `autograd.Function` must be autocast-aware.

## Status

Alpha. Tuned on gfx1201 (R9700) for 3D UNet training. Not a general MIOpen replacement.

## License

MIT. Independent of any Kaggle / competition codebase.
