# Design and measurement log

Numbers below were taken on one machine: AMD Radeon AI PRO R9700 (gfx1201, 64 CU, 32 GB) and an RTX 3070 8 GB, ROCm 7.2, PyTorch 2.14, Triton 3.8. Protocol unless noted: UNet3D `base=24`, pool2 input `64×128×128`, batch 4, AMP, full forward + backward + Adam.

This is an experience report, not a claim that every 3D conv on every RDNA4 SKU will see the same speedup.

## 1. The bottleneck is solver selection

MIOpen has 3D find-db entries for several CDNA / older RDNA arches, not for gfx12xx. Stock `nn.Conv3d` saturates the card on a *bad* kernel:

| | ms/sample |
|---|---:|
| R9700 MIOpen, pool2 b4 | 261–290 (b4 and b8 identical → compute-bound) |
| RTX 3070 + channels_last, same script | 64 |

Cheap workarounds that did nothing:

| Attempt | Result |
|---|---|
| `channels_last_3d` | 1.00× (ROCm ignores it for conv3d) |
| Decompose 3×3×3 into three conv2d | 1.06×, memory 5.5 → 9.5 GB |

## 2. Implicit GEMM on WMMA

Map the convolution onto GEMM tiles that match gfx1201 WMMA (16×16×16, Wave32, fp16/bf16 in, fp32 acc).

- Activations: NDHWC
- Weights: `(Kd·Kh·Kw·Cin, Cout)` with Cin fastest inside each tap
- K order: a `BLOCK_K` chunk that stays inside one tap is a contiguous channel vector

Forward 27-tap unroll along Cin is correct. The same unroll on backward, when K is the spatial product (~millions), became 27 serial giant loops and made the net *slower* than MIOpen (e1 `1→24` combined bwd 42 → 698 ms). Weight gradient is `M=Cout`, `N=Cin×27`, `K=spatial`, split-K into a private fp32 buffer, no atomics.

PyTorch 2.14 AMP requires `setup_context` on the custom Function. Saving tensors in `forward(ctx, ...)` left `saved_tensors` empty.

## 3. Isolated timings are not dispatch evidence

Skinny dX (`1→24`, `24→1`) loses to MIOpen in isolation (~0.2–0.6×) and *wins* inside the network:

| | ms/step | peak GB |
|---|---:|---:|
| base=24 skinny dX → MIOpen | 344 | 7.57 |
| base=24 all dX → WMMA | **315** | **6.41** |

Those two MIOpen layers cost ~31 ms in-network (`Col2Im3dU` 20 + rocBLAS 11) versus 2.4 ms isolated — a 13× underestimate. The im2col/workspace path looks free against an empty allocator and a warm cache.

The same trap hit `ConvTranspose3d` k=2 s=2: isolated three legs saved ~15 ms; in the net they saved 171 ms/step.

**Rule: settle dispatch end-to-end.** Isolated kernel time is not evidence about behaviour under a real step's memory pressure.

## 4. Tiles are shape-dependent

A single global wgrad config cannot serve both hot layers and the already-fast e2 `48→48` (forcing the skinny config there cost 60%).

- `16 < Cout ≤ 32`: `BLOCK_K=32`, `warps=8`
- otherwise: `BLOCK_K=64`, `warps=4`
- `Cout=1` is `M≤32` but must *not* take 8 warps
- `BLOCK_M=16` for Cout=24 was a regression (rolled back)
- `BLOCK_K=128/256` was catastrophic on the hot shapes

`base=32` (channels 32/64/128/256, WMMA-aligned) was 12% *faster* than `base=24` at 1.78× capacity. MIOpen got 32% slower at base=32.

## 5. Custom Functions must be autocast-aware

Autocast intercepts at the ATen op. A Function that calls Triton directly never sees it. Early code did `fp32 → fp16 → fp32` inside `forward` and returned fp32, so BN / ReLU / pool ran at double the bytes.

Deleting BatchNorm did **not** remove the cast kernels. BN itself was ~11 ms, not the ~62 ms the profile first suggested.

Fix: cast `x` / `weight` / `bias` to the autocast dtype *outside* the Function and leave the output in that dtype — the same contract as `nn.Conv3d`.

| | before | after |
|---|---:|---:|
| base=24 | 314 ms/step, 6.41 GB | **243 / 3.31 GB** |
| base=32 | 283 ms/step, 8.55 GB | **191 / 4.42 GB** |

That is the point at which the R9700 overtook the 3070 (1.06× / 1.33×). Peak memory halved because AMP saved fp16 activations.

## 6. Training recipe (this model, not this library)

The library only replaces convolution. A TemporalUNet + detection head on this card additionally needed:

- UNet under bf16 autocast (no fp32 WMMA kernel; raw fp16 with no loss scale NaN'd the detection loss)
- 1×1 detection head in fp32 (head in bf16: grad 1e6–1e8, loss stuck, tens of thousands of peaks)
- per-voxel temporal attention in fp32 (`MultiheadAttention` in bf16 raises `hipErrorInvalidValue` on gfx1201)

Those patches belong in the application, not here.

## 7. Still open

- `d1 48→24` dX (~42 ms) is the largest remaining single kernel
- `e1 24→24` dW (~33 ms) is tiling-limited; next step is an LDS-resident `go` tile
- stride-2 convs still go to MIOpen
- no comparison yet against Composable Kernel implicit GEMM once gfx1201 solvers land in MIOpen
