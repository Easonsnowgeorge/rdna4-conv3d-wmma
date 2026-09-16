"""Implicit-GEMM conv3d for RDNA4 (gfx1201) via Triton tl.dot -> WMMA.

Maps a 3D convolution onto GEMM tiles that match the gfx1201 WMMA shape
(16x16x16, Wave32, fp16/bf16 in, fp32 acc). MIOpen's gfx12 conv3d path does
not use those matrix cores; this one is written so that every inner product
is a tl.dot of 16-aligned tiles.

Layouts inside the kernels (packed once per call):
  input/output : NDHWC  (channels_last_3d)
  weight       : (Kd*Kh*Kw*Cin, Cout) with Cin fastest inside each tap

K-ordering: k = ((kd*KH + kh)*KW + kw)*Cin + c, so a BLOCK_K chunk that
stays inside one tap is a contiguous channel vector — the only gather is
the spatial neighbour, not the channel axis.

Hybrid backend (per-op, not a full MIOpen replacement):
  RDNA4_CONV3D_BACKEND=hybrid (default) | wmma | miopen | hip
  BIOHUB_CONV3D_BACKEND is still read as a fallback alias.
  FWD/dW/dX: k=1 or k=3 s=1 dil=1 groups=1 on gfx120x → WMMA
  else: MIOpen (unsupported geom, non-gfx1201, groups≠1)

Set RDNA4_CONV3D_BACKEND=hip to try the rocWMMA HIP fprop (3x3x3 s1 only);
backward still goes through Triton.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


def _is_rdna4() -> bool:
    if not torch.cuda.is_available():
        return False
    name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return "gfx120" in name


def _t3(v):
    if isinstance(v, int):
        return (v, v, v)
    return tuple(int(x) for x in v)


# ---------------------------------------------------------------------------
# Shape dispatcher. Measured R9700 pool2 b4 fp16, MIOpen → WMMA (ms):
#   FWD  1→24 5.0→1.9 (2.6x)  24→24 60→10 (5.8x)  48→48 15→2.5 (6.1x)
#        96→48 30→3.8 (7.7x)  48→24 118→11 (11x)  24→1 k1 2.0→1.1 (1.8x)
#   dW   all 12 shapes faster; 24→24 full-res 80→39 (2.0x) is the weak dW
#   dX   ALL shapes go WMMA. The skinny ones still LOSE in isolation
#        (1→24 1.5→6.9 = 0.2x, 24→1 1.9→3.2 = 0.6x) but win e2e by 29–41 ms and
#        1.2 GB, because MIOpen's im2col/workspace costs ~31 ms inside a real step
#        vs 2.4 ms measured alone. Settle dispatch e2e, never in isolation.
#        e2 2.2+2.2+2.2=6.6ms — do not chase.
# ---------------------------------------------------------------------------
def _backend_name() -> str:
    return (
        os.environ.get("RDNA4_CONV3D_BACKEND")
        or os.environ.get("BIOHUB_CONV3D_BACKEND")
        or "hybrid"
    ).lower()


_BACKEND = _backend_name()


def _wmma_hw() -> bool:
    if _BACKEND in ("miopen", "torch"):
        return False
    if not _HAS_TRITON:
        return False
    if _BACKEND == "wmma":
        return True
    return _is_rdna4()


def _geom_ok(kernel_size, stride, dilation, groups) -> bool:
    if int(groups) != 1:
        return False
    ks, st, dl = _t3(kernel_size), _t3(stride), _t3(dilation)
    if dl != (1, 1, 1) or st != (1, 1, 1):
        return False
    return ks in ((1, 1, 1), (3, 3, 3))


def wmma_ok_fwd(cin, cout, kernel_size, stride=1, dilation=1, groups=1) -> bool:
    return _wmma_hw() and _geom_ok(kernel_size, stride, dilation, groups)


def wmma_ok_dw(cin, cout, kernel_size, stride=1, dilation=1, groups=1) -> bool:
    return wmma_ok_fwd(cin, cout, kernel_size, stride, dilation, groups)


def wmma_ok_dx(cin, cout, kernel_size, stride=1, dilation=1, groups=1) -> bool:
    """Every dX geometry we support goes to WMMA, skinny ones included.

    This used to be `cin >= 16 and cout >= 16`, sending e1 1->24 and out 24->1 to
    MIOpen, because ISOLATED micro-benchmarks said MIOpen won them by ~6.8 ms
    (1.56 vs 6.86 and 1.94 vs 3.46). Measured e2e on 09-04 09:50, that is wrong:

        base=24  skinny dX -> MIOpen  344 ms/step  7.57 GB
                 all dX    -> WMMA    315 ms/step  6.41 GB
        base=32  skinny dX -> MIOpen  323 ms/step 10.09 GB
                 all dX    -> WMMA    282 ms/step  8.55 GB

    Faster AND smaller, by 29-41 ms. The profile shows why: in the full step those
    two MIOpen layers cost ~31 ms (Col2Im3dU 20.25 + a rocBLAS Cijk GEMM 11.18),
    not the 2.4 ms they measure alone -- MIOpen's im2col/workspace path degrades
    badly under a real step's memory pressure. This is the same trap the
    ConvTranspose fusion already hit once: isolated it looked worth ~15 ms, in the
    net it was worth 171.

    LESSON: dispatch decisions must be settled e2e. An isolated kernel timing is
    not evidence about behaviour inside the network.
    """
    return wmma_ok_fwd(cin, cout, kernel_size, stride, dilation, groups)


def conv3d_plan(cin, cout, kernel_size, stride=1, dilation=1, groups=1) -> dict:
    """Which backend runs each leg. Values are 'wmma' or 'miopen'."""
    ks, st, dl = kernel_size, stride, dilation
    g = groups
    return {
        "fwd": "wmma" if wmma_ok_fwd(cin, cout, ks, st, dl, g) else "miopen",
        "dw": "wmma" if wmma_ok_dw(cin, cout, ks, st, dl, g) else "miopen",
        "dx": "wmma" if wmma_ok_dx(cin, cout, ks, st, dl, g) else "miopen",
    }


# ---------------------------------------------------------------------------
# Tile picker. gfx1201: Wave32, 64 KB LDS, 64 CUs, WMMA 16x16x16.
# LDS for A+B with 2 stages: (BM*BK + BK*BN) * 2B * 2  must stay under 64 KB.
# ---------------------------------------------------------------------------
def _pick_tiles(M: int, N: int, K: int, kind: str = "fwd") -> Tuple[int, int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages.

    kind:
      fwd/dgrad — M is spatial, N is channels (Cout or Cin), K is the other channel
      wgrad     — M is Cout, N is Cin, K is spatial (split-K GEMM)
    """
    stages = 2
    if kind == "wgrad":
        # Skinny-M (Cout), fat-N (Cin*kvol), huge-K (spatial). Feed WMMA on K.
        # bm=16 for M<=24 was measured slower (e1 dW 39→52, d1 dW 25→46).
        bm = 16 if M <= 16 else (32 if M <= 32 else 64)
        bn = 64 if N >= 64 else (32 if N >= 32 else 16)
        # The wgrad optimum is shape-dependent and splits on M=Cout (swept
        # 09-04 09:35, one knob at a time, ranked on the two hot shapes):
        #   Cout<=32  bk=32 warps=8 -- e1 24->24 dW 40.28 -> 32.84 (2.0x -> 2.5x),
        #             d1 48->24 dW 25.20 -> 24.31.
        #   Cout>=48  bk=64 warps=4 -- forcing the skinny config on e2 48->48
        #             costs 60% (dW 2.21 -> 3.53), the exact e2 regression the
        #             standing rule forbids. A single global config cannot serve
        #             both, which is why this is keyed on Cout rather than tuned.
        # num_stages=3 and target_blocks 256/1024 were swept and lost; bk=128 and
        # 256 are catastrophic on the hot shapes (e1 dW 92 and 304 ms).
        # Lower bound matters: Cout=1 (the out 24->1 layer) also has M<=32 but a
        # 1-row GEMM does not want 8 warps -- it measured 2.98 -> 3.83 ms. Only the
        # genuinely skinny-but-not-degenerate M band takes the tuned config.
        if 16 < M <= 32:
            return bm, bn, 32, 8, stages
        return bm, bn, 64, 4, stages
    if N <= 16:
        bn = 16
    elif N <= 32:
        bn = 32
    elif N <= 64 and M >= 65536:
        bn = 64
    elif N <= 48:
        bn = 32
    else:
        bn = 64
    if kind == "dgrad":
        # Same tiling as FWD once N=Cin is wide enough for WMMA (special
        # kernels handle Cin==1). BLOCK_M=128 on large spatial matches FWD.
        if M >= 65536:
            bm, warps = 128, 8
        elif M >= 64:
            bm, warps = 64, 4
        else:
            bm, warps = 32, 4
        bk = _pick_bk(K)
        return bm, bn, bk, warps, stages
    if M >= 65536:
        bm, warps = 128, 8
    elif M >= 64:
        bm, warps = 64, 4
    else:
        bm, warps = 32, 4
    bk = _pick_bk(K)
    return bm, bn, bk, warps, stages


def _pick_bk(K: int) -> int:
    """BLOCK_K minimising the padded K, cdiv(K,bk)*bk; larger bk breaks ties.

    The old rule (bk = 16 if K < 32 else 32) is a straddle test, not a fit test,
    and it is wrong for K=48: it issues 64 K-slots to cover 48 channels, so a
    third of every MAC is padding. bk=16 covers 48 exactly. K=24 ties at 32 slots
    either way, so take bk=32 and save a loop iteration. 96 and 192 are unchanged.
    Ties fall back to the OLD rule rather than to the larger bk. Measured: taking
    bk=32 for K=24 (same 32 slots, one fewer loop iteration -- the plausible-looking
    choice) costs 15%: e1 24->24 FWD 10.20 -> 11.76 ms and its dX 10.23 -> 11.73.
    With bk=16 the first of the two K-iterations is fully dense; with bk=32 every
    single dot carries a 24/32 mask. Only K=48 actually changes here.

    Capped at 32 on purpose -- bk=64 also ties at 96/192 but changes LDS pressure,
    which is a separate, unmeasured knob.
    """
    old_rule = 16 if K < 32 else 32
    best = old_rule
    for bk in (16, 32):
        if -(-K // bk) * bk < -(-K // best) * best:
            best = bk
    return best


def _split_k(M, N, K, bm, bn, bk, target_blocks=512):
    tiles = max(1, -(-M // bm) * -(-N // bn))
    split = max(1, (target_blocks + tiles - 1) // tiles)
    max_split = max(1, K // max(bk * 32, 1))
    return int(max(1, min(split, max_split, 32)))


def _out_size(L, k, stride, pad, dil):
    return (L + 2 * pad - dil * (k - 1) - 1) // stride + 1


# ===========================================================================
# Triton kernels
# ===========================================================================
if _HAS_TRITON:

    @triton.jit
    def _fwd_kernel(
        x_ptr, w_ptr, y_ptr, bias_ptr,
        N, D, H, W, Cin, Cout, OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
        stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
        stride_wk, stride_wn,
        stride_yn, stride_yd, stride_yh, stride_yw, stride_yc,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Y[m, n] = sum_k im2col(X)[m, k] @ W_packed[k, n], m = n_b*OD*OH*OW."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        ow = offs_m % OW
        t = offs_m // OW
        oh = t % OH
        t = t // OH
        od = t % OD
        n = t // OD

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, GEMM_K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < GEMM_K
            c = offs_k % Cin
            tap = offs_k // Cin
            kw = tap % KW
            t2 = tap // KW
            kh = t2 % KH
            kd = t2 // KH

            id_ = od[:, None] * stride_d + kd[None, :] * dil_d - pad_d
            ih_ = oh[:, None] * stride_h + kh[None, :] * dil_h - pad_h
            iw_ = ow[:, None] * stride_w + kw[None, :] * dil_w - pad_w
            inb = (id_ >= 0) & (id_ < D) & (ih_ >= 0) & (ih_ < H) & (iw_ >= 0) & (iw_ < W)
            a_mask = mask_m[:, None] & mask_k[None, :] & inb
            a_off = (
                n[:, None].to(tl.int64) * stride_xn
                + id_.to(tl.int64) * stride_xd
                + ih_.to(tl.int64) * stride_xh
                + iw_.to(tl.int64) * stride_xw
                + c[None, :].to(tl.int64) * stride_xc
            )
            a = tl.load(x_ptr + a_off, mask=a_mask, other=0.0)

            b_off = offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
            b = tl.load(w_ptr + b_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = acc.to(y_ptr.dtype.element_ty)
        y_off = (
            n[:, None].to(tl.int64) * stride_yn
            + od[:, None].to(tl.int64) * stride_yd
            + oh[:, None].to(tl.int64) * stride_yh
            + ow[:, None].to(tl.int64) * stride_yw
            + offs_n[None, :].to(tl.int64) * stride_yc
        )
        tl.store(y_ptr + y_off, y, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _fwd_k1_kernel(
        x_ptr, w_ptr, y_ptr, bias_ptr,
        GEMM_M, GEMM_N, GEMM_K,
        stride_xm, stride_xk, stride_wk, stride_wn, stride_ym, stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """1x1x1: Y = X @ W, X is (M, Cin) NDHWC flattened, W is (Cin, Cout)."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        a_ptrs = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :].to(tl.int64) * stride_xk
        b_ptrs = w_ptr + offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, GEMM_K, BLOCK_K):
            k = k0 + offs_k
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (k[None, :] < GEMM_K), other=0.0)
            b = tl.load(b_ptrs, mask=(k[:, None] < GEMM_K) & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_xk
            b_ptrs += BLOCK_K * stride_wk
        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = acc.to(y_ptr.dtype.element_ty)
        tl.store(
            y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :].to(tl.int64) * stride_yn,
            y, mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fwd_k3_kernel(
        x_ptr, w_ptr, y_ptr, bias_ptr,
        N, D, H, W, Cin, Cout, OD, OH, OW,
        pad_d, pad_h, pad_w,
        GEMM_M, GEMM_N,
        stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
        stride_wk, stride_wn,
        stride_yn, stride_yd, stride_yh, stride_yw, stride_yc,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """3x3x3 stride-1: 27 taps unrolled, K walks Cin (contiguous in NDHWC)."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        ow = offs_m % OW
        t = offs_m // OW
        oh = t % OH
        t = t // OH
        od = t % OD
        n = t // OD

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Python-level range(27) is unrolled by the Triton compiler.
        for tap in range(27):
            kd = tap // 9
            kh = (tap // 3) % 3
            kw = tap % 3
            id_ = od + kd - pad_d
            ih_ = oh + kh - pad_h
            iw_ = ow + kw - pad_w
            inb = (id_ >= 0) & (id_ < D) & (ih_ >= 0) & (ih_ < H) & (iw_ >= 0) & (iw_ < W)
            for ki in range(0, tl.cdiv(Cin, BLOCK_K)):
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < Cin
                a_mask = mask_m[:, None] & mask_k[None, :] & inb[:, None]
                a_off = (
                    n[:, None].to(tl.int64) * stride_xn
                    + id_[:, None].to(tl.int64) * stride_xd
                    + ih_[:, None].to(tl.int64) * stride_xh
                    + iw_[:, None].to(tl.int64) * stride_xw
                    + offs_k[None, :].to(tl.int64) * stride_xc
                )
                a = tl.load(x_ptr + a_off, mask=a_mask, other=0.0)
                k_row = tap * Cin + offs_k
                b_off = k_row[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
                b = tl.load(w_ptr + b_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                acc = tl.dot(a, b, acc)

        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = acc.to(y_ptr.dtype.element_ty)
        y_off = (
            n[:, None].to(tl.int64) * stride_yn
            + od[:, None].to(tl.int64) * stride_yd
            + oh[:, None].to(tl.int64) * stride_yh
            + ow[:, None].to(tl.int64) * stride_yw
            + offs_n[None, :].to(tl.int64) * stride_yc
        )
        tl.store(y_ptr + y_off, y, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _dgrad_kernel(
        go_ptr, w_ptr, gx_ptr,
        N, D, H, W, Cin, Cout, OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
        stride_gon, stride_god, stride_goh, stride_gow, stride_goc,
        stride_wk, stride_wn,
        stride_gxn, stride_gxd, stride_gxh, stride_gxw, stride_gxc,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """dX[m, cin] = sum_{cout,tap} dY[out_pos, cout] * W[k, cout], m = n*D*H*W.

        k-index of W_packed is (tap, cin) with cin fastest: k = tap*Cin + cin.
        For dgrad, GEMM_N = Cin and we walk Cout*kvol as K, loading W as
        W_packed[tap*Cin + cin, cout] i.e. (GEMM_N, Cout) per tap — handled by
        treating packed W as (kvol*Cin, Cout) and gathering the right row.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cin
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        iw = offs_m % W
        t = offs_m // W
        ih = t % H
        t = t // H
        id_ = t % D
        n = t // D

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, GEMM_K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)  # over Cout * kvol, cout fastest
            mask_k = offs_k < GEMM_K
            cout = offs_k % Cout
            tap = offs_k // Cout
            kw = tap % KW
            t2 = tap // KW
            kh = t2 % KH
            kd = t2 // KH

            # id = od*stride + kd*dil - pad  =>  od*stride = id - kd*dil + pad
            numer_d = id_[:, None] + pad_d - kd[None, :] * dil_d
            numer_h = ih[:, None] + pad_h - kh[None, :] * dil_h
            numer_w = iw[:, None] + pad_w - kw[None, :] * dil_w
            match = (
                (numer_d % stride_d == 0)
                & (numer_h % stride_h == 0)
                & (numer_w % stride_w == 0)
            )
            od = numer_d // stride_d
            oh = numer_h // stride_h
            ow = numer_w // stride_w
            inb = match & (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
            a_mask = mask_m[:, None] & mask_k[None, :] & inb
            a_off = (
                n[:, None].to(tl.int64) * stride_gon
                + od.to(tl.int64) * stride_god
                + oh.to(tl.int64) * stride_goh
                + ow.to(tl.int64) * stride_gow
                + cout[None, :].to(tl.int64) * stride_goc
            )
            a = tl.load(go_ptr + a_off, mask=a_mask, other=0.0)

            # B[k, n] = W_packed[tap*Cin + cin, cout] = W_packed[tap*Cin + offs_n, cout]
            # packed layout (tap, cin, cout) -> k_row = tap*Cin + cin, n_col = cout
            # we need B (cout_k, cin) = packed[tap*Cin + cin, cout].T
            w_row = tap[:, None] * Cin + offs_n[None, :]
            b_off = w_row.to(tl.int64) * stride_wk + cout[:, None].to(tl.int64) * stride_wn
            b = tl.load(w_ptr + b_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        gx = acc.to(gx_ptr.dtype.element_ty)
        gx_off = (
            n[:, None].to(tl.int64) * stride_gxn
            + id_[:, None].to(tl.int64) * stride_gxd
            + ih[:, None].to(tl.int64) * stride_gxh
            + iw[:, None].to(tl.int64) * stride_gxw
            + offs_n[None, :].to(tl.int64) * stride_gxc
        )
        tl.store(gx_ptr + gx_off, gx, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _wgrad_kernel(
        x_ptr, go_ptr, dw_ptr,
        N, D, H, W, Cin, Cout, OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
        stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
        stride_gon, stride_god, stride_goh, stride_gow, stride_goc,
        stride_split, stride_wk, stride_wn,
        SPLIT_K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """dW = X_col^T @ dY, implicit. M=Cout, N=Cin*kvol, K=spatial.

        Taps live in N (parallel tiles), not unrolled over K. Split-K writes a
        private fp32 buffer; the host sums splits. No atomics.
        """
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # cout
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # tap*Cin + cin
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        cin = offs_n % Cin
        tap = offs_n // Cin
        kw = tap % KW
        t2 = tap // KW
        kh = t2 % KH
        kd = t2 // KH

        chunk = tl.cdiv(GEMM_K, SPLIT_K)
        k_start = pid_k * chunk
        k_end = min(k_start + chunk, GEMM_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(k_start, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)  # spatial
            mask_k = (offs_k < k_end) & (offs_k < GEMM_K)
            ow = offs_k % OW
            t = offs_k // OW
            oh = t % OH
            t = t // OH
            od = t % OD
            n = t // OD

            # A is (BLOCK_M, BLOCK_K) = (cout, spatial). go is NDHWC: (spatial, cout).
            go_off = (
                n[None, :].to(tl.int64) * stride_gon
                + od[None, :].to(tl.int64) * stride_god
                + oh[None, :].to(tl.int64) * stride_goh
                + ow[None, :].to(tl.int64) * stride_gow
                + offs_m[:, None].to(tl.int64) * stride_goc
            )
            a = tl.load(go_ptr + go_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            id_ = od[:, None] * stride_d + kd[None, :] * dil_d - pad_d
            ih_ = oh[:, None] * stride_h + kh[None, :] * dil_h - pad_h
            iw_ = ow[:, None] * stride_w + kw[None, :] * dil_w - pad_w
            inb = (id_ >= 0) & (id_ < D) & (ih_ >= 0) & (ih_ < H) & (iw_ >= 0) & (iw_ < W)
            b_mask = mask_k[:, None] & mask_n[None, :] & inb
            b_off = (
                n[:, None].to(tl.int64) * stride_xn
                + id_.to(tl.int64) * stride_xd
                + ih_.to(tl.int64) * stride_xh
                + iw_.to(tl.int64) * stride_xw
                + cin[None, :].to(tl.int64) * stride_xc
            )
            b = tl.load(x_ptr + b_off, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)

        acc_t = tl.trans(acc)
        dw_off_t = (
            pid_k * stride_split
            + offs_n[:, None].to(tl.int64) * stride_wk
            + offs_m[None, :].to(tl.int64) * stride_wn
        )
        tl.store(dw_ptr + dw_off_t, acc_t, mask=mask_n[:, None] & mask_m[None, :])

    @triton.jit(do_not_specialize=["NTAPS"])
    def _dgrad_k3_kernel(
        go_ptr, w_ptr, gx_ptr,
        N, D, H, W, Cin, Cout, OD, OH, OW,
        pad_d, pad_h, pad_w,
        GEMM_M, GEMM_N,
        stride_gon, stride_god, stride_goh, stride_gow, stride_goc,
        stride_wk, stride_wn,
        stride_gxn, stride_gxd, stride_gxh, stride_gxw, stride_gxc,
        NTAPS,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """3x3x3 stride-1 dX: taps in a runtime loop (not unrolled) to save VGPR."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        iw = offs_m % W
        t = offs_m // W
        ih = t % H
        t = t // H
        id_ = t % D
        n = t // D

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for tap in range(0, NTAPS):
            kd = tap // 9
            kh = (tap // 3) % 3
            kw = tap % 3
            od = id_ - kd + pad_d
            oh = ih - kh + pad_h
            ow = iw - kw + pad_w
            inb = (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
            for ki in range(0, tl.cdiv(Cout, BLOCK_K)):
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < Cout
                a_mask = mask_m[:, None] & mask_k[None, :] & inb[:, None]
                a_off = (
                    n[:, None].to(tl.int64) * stride_gon
                    + od[:, None].to(tl.int64) * stride_god
                    + oh[:, None].to(tl.int64) * stride_goh
                    + ow[:, None].to(tl.int64) * stride_gow
                    + offs_k[None, :].to(tl.int64) * stride_goc
                )
                a = tl.load(go_ptr + a_off, mask=a_mask, other=0.0)
                cin_rows = tap * Cin + offs_n
                b_off = cin_rows[None, :].to(tl.int64) * stride_wk + offs_k[:, None].to(tl.int64) * stride_wn
                b = tl.load(w_ptr + b_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                acc = tl.dot(a, b, acc)

        gx = acc.to(gx_ptr.dtype.element_ty)
        gx_off = (
            n[:, None].to(tl.int64) * stride_gxn
            + id_[:, None].to(tl.int64) * stride_gxd
            + ih[:, None].to(tl.int64) * stride_gxh
            + iw[:, None].to(tl.int64) * stride_gxw
            + offs_n[None, :].to(tl.int64) * stride_gxc
        )
        tl.store(gx_ptr + gx_off, gx, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _dgrad_k3_n1_kernel(
        go_ptr, w_ptr, gx_ptr,
        N, D, H, W, Cout, OD, OH, OW,
        pad_d, pad_h, pad_w,
        GEMM_M,
        stride_gon, stride_god, stride_goh, stride_gow, stride_goc,
        stride_wk, stride_wn,
        stride_gxn, stride_gxd, stride_gxh, stride_gxw, stride_gxc,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Cin=1, 3x3x3 dX: no WMMA on N. acc[m] += dY[out, :] · W[tap, :]."""
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < GEMM_M
        iw = offs_m % W
        t = offs_m // W
        ih = t % H
        t = t // H
        id_ = t % D
        n = t // D
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for tap in range(27):
            kd = tap // 9
            kh = (tap // 3) % 3
            kw = tap % 3
            od = id_ - kd + pad_d
            oh = ih - kh + pad_h
            ow = iw - kw + pad_w
            inb = (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
            for ki in range(0, tl.cdiv(Cout, BLOCK_K)):
                offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < Cout
                a_off = (
                    n[:, None].to(tl.int64) * stride_gon
                    + od[:, None].to(tl.int64) * stride_god
                    + oh[:, None].to(tl.int64) * stride_goh
                    + ow[:, None].to(tl.int64) * stride_gow
                    + offs_k[None, :].to(tl.int64) * stride_goc
                )
                a = tl.load(go_ptr + a_off, mask=mask_m[:, None] & mask_k[None, :] & inb[:, None], other=0.0)
                b = tl.load(w_ptr + tap * stride_wk + offs_k * stride_wn, mask=mask_k, other=0.0)
                acc += tl.sum(a * b[None, :], 1)
        gx = acc.to(gx_ptr.dtype.element_ty)
        gx_off = (
            n.to(tl.int64) * stride_gxn
            + id_.to(tl.int64) * stride_gxd
            + ih.to(tl.int64) * stride_gxh
            + iw.to(tl.int64) * stride_gxw
        )
        tl.store(gx_ptr + gx_off, gx, mask=mask_m)

    @triton.jit
    def _dgrad_k1_scale_kernel(
        go_ptr, w_ptr, gx_ptr,
        GEMM_M, Cin,
        stride_gm, stride_gc,
        stride_wn, stride_wk,
        stride_xm, stride_xn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """k1 Cout=1: dX[m, cin] = dY[m] * W[cin]."""
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(Cin, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < Cin
        g = tl.load(go_ptr + offs_m.to(tl.int64) * stride_gm, mask=mask_m, other=0.0)
        wv = tl.load(w_ptr + offs_n.to(tl.int64) * stride_wn, mask=mask_n, other=0.0)
        out = g[:, None] * wv[None, :]
        tl.store(
            gx_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_n[None, :].to(tl.int64) * stride_xn,
            out.to(gx_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    # _wgrad_k3_kernel lived here and is deleted on purpose. It put the 27 taps in
    # an OUTER loop, so `go` and `x` were re-read once per tap (54 full passes for
    # Cin=24 with bn=16) -- pure bandwidth. dW must keep taps in N (M=Cout,
    # N=Cin*27, K=spatial, split-K into private fp32 buffers), which is what the
    # live _wgrad_kernel does. Do not reintroduce the tap-outer form.
    @triton.jit
    def _tr_k2s2_fwd_kernel(
        x_ptr, w_ptr, y_ptr, bias_ptr,
        Din, Hin, Win, Cout,
        GEMM_M, GEMM_N, GEMM_K,
        stride_xm, stride_xk, stride_wk, stride_wn,
        stride_yn, stride_yd, stride_yh, stride_yw, stride_yc,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """ConvTranspose k=2 s=2 FWD. One GEMM: Y_8 = X @ W, W is (Cin, 8*Cout).

        Each workgroup owns an input-spatial tile of X (coalesced load) and
        writes the matching 2x2x2 output block. No compact-then-scatter.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        a_ptrs = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :].to(tl.int64) * stride_xk
        b_ptrs = w_ptr + offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, GEMM_K, BLOCK_K):
            k = k0 + offs_k
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (k[None, :] < GEMM_K), other=0.0)
            b = tl.load(b_ptrs, mask=(k[:, None] < GEMM_K) & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_xk
            b_ptrs += BLOCK_K * stride_wk

        tap = offs_n // Cout
        co = offs_n % Cout
        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + co, mask=mask_n, other=0.0).to(tl.float32)

        iw = offs_m % Win
        t = offs_m // Win
        ih = t % Hin
        t = t // Hin
        id_ = t % Din
        n = t // Din
        kd = tap // 4
        kh = (tap // 2) % 2
        kw = tap % 2
        od = id_[:, None] * 2 + kd[None, :]
        oh = ih[:, None] * 2 + kh[None, :]
        ow = iw[:, None] * 2 + kw[None, :]
        y = acc.to(y_ptr.dtype.element_ty)
        y_off = (
            n[:, None].to(tl.int64) * stride_yn
            + od.to(tl.int64) * stride_yd
            + oh.to(tl.int64) * stride_yh
            + ow.to(tl.int64) * stride_yw
            + co[None, :].to(tl.int64) * stride_yc
        )
        tl.store(y_ptr + y_off, y, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _tr_k2s2_dx_kernel(
        go_ptr, w_ptr, gx_ptr,
        Din, Hin, Win, Cout,
        GEMM_M, GEMM_N, GEMM_K,
        stride_gon, stride_god, stride_goh, stride_gow, stride_goc,
        stride_wk, stride_wn,
        stride_gxn, stride_gxd, stride_gxh, stride_gxw, stride_gxc,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """ConvTranspose k=2 s=2 dX. Each WG owns a dX tile: gather 8 dY taps, one store.

        dX[m, cin] = dY_8[m, 8*Cout] @ W[8*Cout, cin].
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        iw = offs_m % Win
        t = offs_m // Win
        ih = t % Hin
        t = t // Hin
        id_ = t % Din
        n = t // Din

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, GEMM_K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < GEMM_K
            tap = offs_k // Cout
            co = offs_k % Cout
            kd = tap // 4
            kh = (tap // 2) % 2
            kw = tap % 2
            od = id_[:, None] * 2 + kd[None, :]
            oh = ih[:, None] * 2 + kh[None, :]
            ow = iw[:, None] * 2 + kw[None, :]
            a_off = (
                n[:, None].to(tl.int64) * stride_gon
                + od.to(tl.int64) * stride_god
                + oh.to(tl.int64) * stride_goh
                + ow.to(tl.int64) * stride_gow
                + co[None, :].to(tl.int64) * stride_goc
            )
            a = tl.load(go_ptr + a_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_off = offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :].to(tl.int64) * stride_wn
            b = tl.load(w_ptr + b_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        gx = acc.to(gx_ptr.dtype.element_ty)
        gx_off = (
            n[:, None].to(tl.int64) * stride_gxn
            + id_[:, None].to(tl.int64) * stride_gxd
            + ih[:, None].to(tl.int64) * stride_gxh
            + iw[:, None].to(tl.int64) * stride_gxw
            + offs_n[None, :].to(tl.int64) * stride_gxc
        )
        tl.store(gx_ptr + gx_off, gx, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _tr_k2s2_dw_kernel(
        x_ptr, go_ptr, dw_ptr,
        Din, Hin, Win, Cin, Cout,
        GEMM_M, GEMM_N, GEMM_K,
        stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
        stride_gon, stride_god, stride_goh, stride_gow, stride_goc,
        stride_split, stride_wk, stride_wn,
        SPLIT_K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """ConvTranspose k=2 s=2 dW. M=8*Cout, N=Cin, K=input spatial. Split-K, no atomics.

        dw[tap*Cout+c, cin] += dY[n,c,2id+kd,2ih+kh,2iw+kw] * X[n,cin,id,ih,iw]
        """
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
        num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < GEMM_M
        mask_n = offs_n < GEMM_N

        tap = offs_m // Cout
        co = offs_m % Cout
        kd = tap // 4
        kh = (tap // 2) % 2
        kw = tap % 2

        chunk = tl.cdiv(GEMM_K, SPLIT_K)
        k_start = pid_k * chunk
        k_end = min(k_start + chunk, GEMM_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = (offs_k < k_end) & (offs_k < GEMM_K)
            iw = offs_k % Win
            t = offs_k // Win
            ih = t % Hin
            t = t // Hin
            id_ = t % Din
            n = t // Din
            od = id_[None, :] * 2 + kd[:, None]
            oh = ih[None, :] * 2 + kh[:, None]
            ow = iw[None, :] * 2 + kw[:, None]
            go_off = (
                n[None, :].to(tl.int64) * stride_gon
                + od.to(tl.int64) * stride_god
                + oh.to(tl.int64) * stride_goh
                + ow.to(tl.int64) * stride_gow
                + co[:, None].to(tl.int64) * stride_goc
            )
            a = tl.load(go_ptr + go_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            x_off = (
                n[:, None].to(tl.int64) * stride_xn
                + id_[:, None].to(tl.int64) * stride_xd
                + ih[:, None].to(tl.int64) * stride_xh
                + iw[:, None].to(tl.int64) * stride_xw
                + offs_n[None, :].to(tl.int64) * stride_xc
            )
            b = tl.load(x_ptr + x_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        acc_t = tl.trans(acc)
        dw_off = (
            pid_k * stride_split
            + offs_n[:, None].to(tl.int64) * stride_wk
            + offs_m[None, :].to(tl.int64) * stride_wn
        )
        tl.store(dw_ptr + dw_off, acc_t, mask=mask_n[:, None] & mask_m[None, :])


# ===========================================================================
# Launch wrappers
# ===========================================================================
def _require_triton():
    if not _HAS_TRITON:
        raise RuntimeError("triton is required for conv3d_wmma")


def _pack_weight(weight: torch.Tensor) -> torch.Tensor:
    """OIDHW (Cout,Cin,Kd,Kh,Kw) -> (Kd*Kh*Kw*Cin, Cout), Cin fastest per tap."""
    cout, cin, kd, kh, kw = weight.shape
    return weight.permute(2, 3, 4, 1, 0).contiguous().view(kd * kh * kw * cin, cout)


def _unpack_weight(packed: torch.Tensor, cout: int, cin: int, kd: int, kh: int, kw: int) -> torch.Tensor:
    return packed.view(kd, kh, kw, cin, cout).permute(4, 3, 0, 1, 2).contiguous()


def _empty_ndhwc(n, c, d, h, w, device, dtype) -> torch.Tensor:
    """dX buffer. Every live dgrad kernel stores its whole tile unconditionally
    (masked only at the tensor bounds), so this must NOT be zeroed: the old
    zero_() cost a 402 MB memset per d1 backward and the is_k3 branch threw the
    buffer away entirely."""
    return torch.empty((n, c, d, h, w), device=device, dtype=dtype,
                       memory_format=torch.channels_last_3d)


def _as_ndhwc(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        return x
    return x.contiguous(memory_format=torch.channels_last_3d)


def _launch_fwd(x, w_packed, bias, stride, padding, dilation, kernel_size, out_shape, dtype):
    n, cin, d, h, w = x.shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    od, oh, ow = out_shape
    cout = w_packed.shape[1]
    x_cl = _as_ndhwc(x)
    y_cl = torch.empty((n, cout, od, oh, ow), device=x.device, dtype=dtype,
                       memory_format=torch.channels_last_3d)
    gemm_m = n * od * oh * ow
    gemm_n = cout
    gemm_k = kd * kh * kw * cin
    is_k1 = kd == kh == kw == 1 and sd == sh == sw == 1 and pd == ph == pw == 0
    is_k3 = (kd, kh, kw) == (3, 3, 3) and (sd, sh, sw) == (1, 1, 1) and (dd, dh, dw) == (1, 1, 1)
    inner_k = cin if (is_k1 or is_k3) else gemm_k
    bm, bn, bk, warps, stages = _pick_tiles(gemm_m, gemm_n, inner_k, "fwd")
    grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(gemm_n, bn),)
    has_bias = bias is not None
    bias_arg = bias if has_bias else w_packed  # dummy ptr, unused

    if is_k1:
        _fwd_k1_kernel[grid](
            x_cl, w_packed, y_cl, bias_arg,
            gemm_m, gemm_n, gemm_k,
            cin, 1, cout, 1, cout, 1,
            HAS_BIAS=has_bias,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=warps, num_stages=stages,
        )
    elif is_k3:
        _fwd_k3_kernel[grid](
            x_cl, w_packed, y_cl, bias_arg,
            n, d, h, w, cin, cout, od, oh, ow,
            pd, ph, pw,
            gemm_m, gemm_n,
            x_cl.stride(0), x_cl.stride(2), x_cl.stride(3), x_cl.stride(4), x_cl.stride(1),
            w_packed.stride(0), w_packed.stride(1),
            y_cl.stride(0), y_cl.stride(2), y_cl.stride(3), y_cl.stride(4), y_cl.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=warps, num_stages=stages,
        )
    else:
        _fwd_kernel[grid](
            x_cl, w_packed, y_cl, bias_arg,
            n, d, h, w, cin, cout, od, oh, ow,
            kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw,
            gemm_m, gemm_n, gemm_k,
            x_cl.stride(0), x_cl.stride(2), x_cl.stride(3), x_cl.stride(4), x_cl.stride(1),
            w_packed.stride(0), w_packed.stride(1),
            y_cl.stride(0), y_cl.stride(2), y_cl.stride(3), y_cl.stride(4), y_cl.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=warps, num_stages=stages,
        )
    return y_cl


def _launch_dgrad(go, w_packed, x_shape, stride, padding, dilation, kernel_size,
                  go_cl=None):
    n, cin, d, h, w = x_shape
    _, cout, od, oh, ow = go.shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    # backward() hands us go_cl so dgrad and wgrad share one NDHWC conversion.
    if go_cl is None:
        go_cl = _as_ndhwc(go)
    gemm_m = n * d * h * w
    gemm_n = cin
    is_k1 = kd == kh == kw == 1 and sd == sh == sw == 1 and pd == ph == pw == 0
    is_k3 = (kd, kh, kw) == (3, 3, 3) and (sd, sh, sw) == (1, 1, 1) and (dd, dh, dw) == (1, 1, 1)

    if is_k1 and cout == 1:
        gx_cl = _empty_ndhwc(n, cin, d, h, w, go.device, go.dtype)
        bm, bn = 256, 32
        grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(cin, bn),)
        _dgrad_k1_scale_kernel[grid](
            go_cl, w_packed, gx_cl,
            gemm_m, cin,
            cout, 1,
            w_packed.stride(0), w_packed.stride(1),
            cin, 1,
            BLOCK_M=bm, BLOCK_N=bn,
            num_warps=4, num_stages=2,
        )
        return gx_cl

    if is_k1:
        gx_cl = _empty_ndhwc(n, cin, d, h, w, go.device, go.dtype)
        bm, bn, bk, warps, stages = _pick_tiles(gemm_m, cin, cout, "dgrad")
        grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(cin, bn),)
        # dX = dY @ W^T. packed is (Cin, Cout); B[k,n]=packed[n,k] => stride_wk=1, stride_wn=Cout.
        _fwd_k1_kernel[grid](
            go_cl, w_packed, gx_cl, w_packed,
            gemm_m, cin, cout,
            cout, 1, 1, cout, cin, 1,
            HAS_BIAS=False,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            num_warps=warps, num_stages=stages,
        )
        return gx_cl

    if is_k3:
        # dX = conv3d(dY, rot180(W)^T) with the same pad. Reuses the FWD WMMA kernel.
        w_oidhw = _unpack_weight(w_packed, cout, cin, 3, 3, 3)
        w_dx = w_oidhw.flip(2, 3, 4).transpose(0, 1).contiguous()
        return _launch_fwd(
            go_cl, _pack_weight(w_dx), None,
            (1, 1, 1), (pd, ph, pw), (1, 1, 1),
            (3, 3, 3), (d, h, w), go.dtype,
        )

    gemm_k = cout * kd * kh * kw
    gx_cl = _empty_ndhwc(n, cin, d, h, w, go.device, go.dtype)
    bm, bn, bk, warps, stages = _pick_tiles(gemm_m, gemm_n, gemm_k, "dgrad")
    grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(gemm_n, bn),)
    _dgrad_kernel[grid](
        go_cl, w_packed, gx_cl,
        n, d, h, w, cin, cout, od, oh, ow,
        kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw,
        gemm_m, gemm_n, gemm_k,
        go_cl.stride(0), go_cl.stride(2), go_cl.stride(3), go_cl.stride(4), go_cl.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        gx_cl.stride(0), gx_cl.stride(2), gx_cl.stride(3), gx_cl.stride(4), gx_cl.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return gx_cl


def _launch_wgrad(x, go, kernel_size, stride, padding, dilation,
                  x_cl=None, go_cl=None):
    n, cin, d, h, w = x.shape
    _, cout, od, oh, ow = go.shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    if x_cl is None:
        x_cl = _as_ndhwc(x)
    if go_cl is None:
        go_cl = _as_ndhwc(go)
    gemm_k = n * od * oh * ow
    gemm_m = cout
    gemm_n = kd * kh * kw * cin
    bm, bn, bk, warps, stages = _pick_tiles(gemm_m, gemm_n, gemm_k, "wgrad")
    split_k = _split_k(gemm_m, gemm_n, gemm_k, bm, bn, bk)
    partial = torch.zeros((split_k, gemm_n, gemm_m), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(gemm_n, bn), split_k)
    _wgrad_kernel[grid](
        x_cl, go_cl, partial,
        n, d, h, w, cin, cout, od, oh, ow,
        kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw,
        gemm_m, gemm_n, gemm_k,
        x_cl.stride(0), x_cl.stride(2), x_cl.stride(3), x_cl.stride(4), x_cl.stride(1),
        go_cl.stride(0), go_cl.stride(2), go_cl.stride(3), go_cl.stride(4), go_cl.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        split_k,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=1,
        num_warps=warps, num_stages=stages,
    )
    packed = partial[0] if split_k == 1 else partial.sum(0)
    return packed.to(go.dtype)


def _maybe_hip_fwd(x, weight, bias, stride, padding, dilation):
    if _backend_name() != "hip":
        return None
    kd, kh, kw = weight.shape[2:]
    if (kd, kh, kw) != (3, 3, 3) or stride != (1, 1, 1) or dilation != (1, 1, 1):
        return None
    try:
        from rdna4_conv3d.hip_backend import hip_conv3d_fwd
    except Exception:
        return None
    return hip_conv3d_fwd(x, weight, bias, padding)


# ===========================================================================
# Autograd + nn.Conv3d drop-in
# ===========================================================================
class Conv3dWmmaFn(torch.autograd.Function):
    # PyTorch 2.14: save tensors in setup_context, not in forward(ctx, ...).
    # AMP + old-style ctx.save_for_backward left saved_tensors empty.

    @staticmethod
    def forward(x, weight, bias, stride, padding, dilation):
        cin, cout = int(x.shape[1]), int(weight.shape[0])
        ks = (int(weight.shape[2]), int(weight.shape[3]), int(weight.shape[4]))
        if not wmma_ok_fwd(cin, cout, ks, stride, dilation, 1):
            return F.conv3d(x, weight, bias, stride, padding, dilation)
        _require_triton()
        compute_dtype = x.dtype
        if compute_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.float16)
            weight = weight.to(torch.float16)
            bias = None if bias is None else bias.to(torch.float16)
        n, cin, d, h, w = x.shape
        od = _out_size(d, ks[0], stride[0], padding[0], dilation[0])
        oh = _out_size(h, ks[1], stride[1], padding[1], dilation[1])
        ow = _out_size(w, ks[2], stride[2], padding[2], dilation[2])
        y = _maybe_hip_fwd(x, weight, bias, stride, padding, dilation)
        if y is None:
            w_packed = _pack_weight(weight)
            y = _launch_fwd(x, w_packed, bias, stride, padding, dilation,
                            ks, (od, oh, ow), x.dtype)
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            y = y.contiguous()
        if compute_dtype not in (torch.float16, torch.bfloat16):
            y = y.to(compute_dtype)
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, stride, padding, dilation = inputs
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.kernel_size = (int(weight.shape[2]), int(weight.shape[3]), int(weight.shape[4]))
        ctx.x_layout = torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) \
            else torch.contiguous_format

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation
        ks = ctx.kernel_size
        cin, cout = int(x.shape[1]), int(weight.shape[0])
        go = grad_out if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out.to(torch.float16)
        x_c = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.float16)
        w_c = weight if weight.dtype in (torch.float16, torch.bfloat16) else weight.to(torch.float16)

        # One NDHWC conversion each for go and x, shared by the two legs. Under a
        # NCDHW model go used to be converted twice (once in each launcher).
        wmma_dx_leg = ctx.needs_input_grad[0] and wmma_ok_dx(cin, cout, ks, stride, dilation, 1)
        wmma_dw_leg = ctx.needs_input_grad[1] and wmma_ok_dw(cin, cout, ks, stride, dilation, 1)
        go_cl = _as_ndhwc(go) if (wmma_dx_leg or wmma_dw_leg) else None
        x_cl = _as_ndhwc(x_c) if wmma_dw_leg else None

        gx = None
        if ctx.needs_input_grad[0]:
            if wmma_dx_leg:
                gx = _launch_dgrad(go, _pack_weight(w_c), x.shape, stride, padding, dilation, ks,
                                   go_cl=go_cl)
            else:
                gx = torch.nn.grad.conv3d_input(
                    x.shape, w_c, go.contiguous(), stride, padding, dilation, 1)
            if gx.dtype != x.dtype:
                gx = gx.to(x.dtype)
            if ctx.x_layout == torch.contiguous_format:
                gx = gx.contiguous()

        dw = None
        if ctx.needs_input_grad[1]:
            if wmma_dw_leg:
                dw = _unpack_weight(_launch_wgrad(x_c, go, ks, stride, padding, dilation,
                                                  x_cl=x_cl, go_cl=go_cl),
                                    cout, cin, *ks)
            else:
                dw = torch.nn.grad.conv3d_weight(
                    x_c, weight.shape, go.contiguous(), stride, padding, dilation, 1)
            if dw.dtype != weight.dtype:
                dw = dw.to(weight.dtype)

        db = go.to(x.dtype).sum(dim=(0, 2, 3, 4)) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return gx, dw, db, None, None, None


def _autocast_dtype():
    """The dtype autocast would run a conv in, or None if autocast is off.

    Conv3dWmmaFn/ConvTranspose3dWmmaFn are custom autograd Functions that call
    Triton directly, so autocast -- which intercepts at the ATen op level -- never
    sees them. Without this, a conv fed fp32 does its own fp32->fp16 cast, computes,
    and casts the OUTPUT BACK to fp32, so every downstream BatchNorm/ReLU/cat/pool
    runs in fp32 at double the bytes. Measured dtype trace through one _block:

        nn.Conv3d  : fp32 in -> fp16 out -> BN fp16 -> ReLU fp16   (autocast-aware)
        Conv3dWmma : fp32 in -> fp32 out -> BN fp32 -> ReLU fp32   (before this fix)

    Casting here instead, OUTSIDE the Function, is what autocast itself does: the
    cast lands in the graph, backward returns the weight grad in fp32 for the fp32
    master weights, and the output stays fp16 like every other AMP op. It makes this
    module behave the same as nn.Conv3d under autocast rather than differently.
    """
    try:
        if not torch.is_autocast_enabled("cuda"):
            return None
        dt = torch.get_autocast_dtype("cuda")
    except TypeError:  # older signature, no device_type argument
        if not torch.is_autocast_enabled():
            return None
        dt = torch.get_autocast_gpu_dtype()
    return dt if dt in (torch.float16, torch.bfloat16) else None


def _autocast_args(x, weight, bias):
    dt = _autocast_dtype()
    if dt is None or not x.is_cuda:
        return x, weight, bias
    x = x.to(dt) if x.dtype != dt else x
    weight = weight.to(dt) if weight.dtype != dt else weight
    if bias is not None and bias.dtype != dt:
        bias = bias.to(dt)
    return x, weight, bias


def conv3d_wmma(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if groups != 1:
        return F.conv3d(x, weight, bias, stride, padding, dilation, groups)
    x, weight, bias = _autocast_args(x, weight, bias)
    return Conv3dWmmaFn.apply(x, weight, bias, _t3(stride), _t3(padding), _t3(dilation))


def wmma_fwd(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Isolated forward. Returns NCDHW unless x is channels_last_3d."""
    _require_triton()
    st, pd, dl = _t3(stride), _t3(padding), _t3(dilation)
    kd, kh, kw = int(weight.shape[2]), int(weight.shape[3]), int(weight.shape[4])
    n, cin, d, h, w = x.shape
    od = _out_size(d, kd, st[0], pd[0], dl[0])
    oh = _out_size(h, kh, st[1], pd[1], dl[1])
    ow = _out_size(w, kw, st[2], pd[2], dl[2])
    packed = _pack_weight(weight)
    y = _launch_fwd(x, packed, bias, st, pd, dl, (kd, kh, kw), (od, oh, ow), x.dtype)
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        return y
    return y.contiguous()


def wmma_dw(x, weight, go, stride=1, padding=0, dilation=1):
    """Isolated dW. Returns OIDHW like nn.Conv3d.weight.grad."""
    _require_triton()
    st, pd, dl = _t3(stride), _t3(padding), _t3(dilation)
    ks = (int(weight.shape[2]), int(weight.shape[3]), int(weight.shape[4]))
    packed = _launch_wgrad(x, go, ks, st, pd, dl)
    return _unpack_weight(packed, weight.shape[0], weight.shape[1], *ks)


def wmma_dx(x, weight, go, stride=1, padding=0, dilation=1):
    """Isolated dX. Skinny Cin/Cout dispatches to MIOpen under hybrid policy."""
    st, pd, dl = _t3(stride), _t3(padding), _t3(dilation)
    ks = (int(weight.shape[2]), int(weight.shape[3]), int(weight.shape[4]))
    cin, cout = int(x.shape[1]), int(weight.shape[0])
    if not wmma_ok_dx(cin, cout, ks, st, dl, 1):
        w = weight if weight.dtype == go.dtype else weight.to(go.dtype)
        gx = torch.nn.grad.conv3d_input(x.shape, w, go.contiguous(), st, pd, dl, 1)
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            return gx.contiguous(memory_format=torch.channels_last_3d)
        return gx.contiguous()
    _require_triton()
    packed = _pack_weight(weight)
    gx = _launch_dgrad(go, packed, x.shape, st, pd, dl, ks)
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        return gx
    return gx.contiguous()


def swap_conv3d(module: nn.Module) -> int:
    """Replace every ``nn.Conv3d`` with ``Conv3dWmma``, copying weights. Returns count."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv3d:
            new = Conv3dWmma(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups,
                bias=child.bias is not None,
            )
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data)
            setattr(module, name, new)
            n += 1
        else:
            n += swap_conv3d(child)
    return n


class Conv3dWmma(nn.Conv3d):
    """Drop-in nn.Conv3d. Per-op WMMA/MIOpen dispatch on gfx1201; else nn.Conv3d."""

    def forward(self, x):
        if self.groups != 1 or not x.is_cuda:
            return super().forward(x)
        cin, cout = int(x.shape[1]), int(self.weight.shape[0])
        plan = conv3d_plan(cin, cout, self.kernel_size, self.stride, self.dilation, self.groups)
        if plan["fwd"] == "miopen" and plan["dw"] == "miopen" and plan["dx"] == "miopen":
            return super().forward(x)
        return conv3d_wmma(x, self.weight, self.bias, self.stride, self.padding, self.dilation, 1)


# ---------------------------------------------------------------------------
# ConvTranspose3d k=2 s=2 pad=0 — output-centric gather, one GEMM each way.
# Weight pack: (Cin, Cout, 2,2,2) -> (Cin, 8*Cout), tap = kd*4+kh*2+kw.
# FWD: Y_8 = X @ W          (coalesced X, strided 2x2x2 store)
# dX:  dX  = dY_8 @ W^T     (WG owns dX tile; gather 8 dY taps; one store)
# dW:  dW  = dY_8^T @ X     (split-K, no atomics)
# ---------------------------------------------------------------------------
def _tr_pack_w(weight: torch.Tensor) -> torch.Tensor:
    cin, cout = weight.shape[:2]
    return weight.permute(0, 2, 3, 4, 1).contiguous().view(cin, 8 * cout)


def _tr_unpack_w(packed: torch.Tensor, cin: int, cout: int) -> torch.Tensor:
    return packed.view(cin, 2, 2, 2, cout).permute(0, 4, 1, 2, 3).contiguous()


def transpose_k2s2_fwd(x, weight, bias=None):
    """weight is PyTorch ConvTranspose3d layout (Cin, Cout, 2, 2, 2)."""
    _require_triton()
    n, cin, din, hin, win = x.shape
    cout = int(weight.shape[1])
    x_cl = _as_ndhwc(x)
    y = torch.empty((n, cout, din * 2, hin * 2, win * 2), device=x.device, dtype=x.dtype,
                    memory_format=torch.channels_last_3d)
    w_kn = _tr_pack_w(weight)
    gemm_m = n * din * hin * win
    gemm_n = 8 * cout
    gemm_k = cin
    bm, bn, bk, warps, stages = _pick_tiles(gemm_m, gemm_n, gemm_k, "fwd")
    grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(gemm_n, bn),)
    has_bias = bias is not None
    bias_arg = bias if has_bias else w_kn
    _tr_k2s2_fwd_kernel[grid](
        x_cl, w_kn, y, bias_arg,
        din, hin, win, cout,
        gemm_m, gemm_n, gemm_k,
        cin, 1, w_kn.stride(0), w_kn.stride(1),
        y.stride(0), y.stride(2), y.stride(3), y.stride(4), y.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return y


def transpose_k2s2_dx(x_shape, weight, go):
    _require_triton()
    n, cin, din, hin, win = x_shape
    cout = int(weight.shape[1])
    go_cl = _as_ndhwc(go)
    gx = torch.empty((n, cin, din, hin, win), device=go.device, dtype=go.dtype,
                     memory_format=torch.channels_last_3d)
    w_nk = _tr_pack_w(weight).t().contiguous()
    gemm_m = n * din * hin * win
    gemm_n = cin
    gemm_k = 8 * cout
    bm, bn, bk, warps, stages = _pick_tiles(gemm_m, gemm_n, gemm_k, "dgrad")
    grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(gemm_n, bn),)
    _tr_k2s2_dx_kernel[grid](
        go_cl, w_nk, gx,
        din, hin, win, cout,
        gemm_m, gemm_n, gemm_k,
        go_cl.stride(0), go_cl.stride(2), go_cl.stride(3), go_cl.stride(4), go_cl.stride(1),
        w_nk.stride(0), w_nk.stride(1),
        gx.stride(0), gx.stride(2), gx.stride(3), gx.stride(4), gx.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return gx


def transpose_k2s2_dw(x, weight_shape, go):
    _require_triton()
    cin, cout = int(weight_shape[0]), int(weight_shape[1])
    n, _, din, hin, win = x.shape
    x_cl = _as_ndhwc(x)
    go_cl = _as_ndhwc(go)
    gemm_m = 8 * cout
    gemm_n = cin
    gemm_k = n * din * hin * win
    bm, bn, bk, warps, stages = _pick_tiles(gemm_m, gemm_n, gemm_k, "wgrad")
    split_k = _split_k(gemm_m, gemm_n, gemm_k, bm, bn, bk)
    partial = torch.zeros((split_k, gemm_n, gemm_m), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(gemm_m, bm) * triton.cdiv(gemm_n, bn), split_k)
    _tr_k2s2_dw_kernel[grid](
        x_cl, go_cl, partial,
        din, hin, win, cin, cout,
        gemm_m, gemm_n, gemm_k,
        x_cl.stride(0), x_cl.stride(2), x_cl.stride(3), x_cl.stride(4), x_cl.stride(1),
        go_cl.stride(0), go_cl.stride(2), go_cl.stride(3), go_cl.stride(4), go_cl.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        split_k,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=1,
        num_warps=warps, num_stages=stages,
    )
    packed = partial[0] if split_k == 1 else partial.sum(0)
    return _tr_unpack_w(packed.to(go.dtype), cin, cout)


class ConvTranspose3dWmmaFn(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias):
        compute_dtype = x.dtype
        if compute_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.float16)
            weight = weight.to(torch.float16)
            bias = None if bias is None else bias.to(torch.float16)
        y = transpose_k2s2_fwd(x, weight, bias)
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            y = y.contiguous()
        if compute_dtype not in (torch.float16, torch.bfloat16):
            y = y.to(compute_dtype)
        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias = inputs
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.x_layout = torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) \
            else torch.contiguous_format

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        go = grad_out if grad_out.dtype in (torch.float16, torch.bfloat16) else grad_out.to(torch.float16)
        x_c = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.float16)
        w_c = weight if weight.dtype in (torch.float16, torch.bfloat16) else weight.to(torch.float16)
        gx = transpose_k2s2_dx(x.shape, w_c, go)
        dw = transpose_k2s2_dw(x_c, weight.shape, go)
        db = go.to(x.dtype).sum(dim=(0, 2, 3, 4)) if ctx.has_bias else None
        if gx.dtype != x.dtype:
            gx = gx.to(x.dtype)
        if dw.dtype != weight.dtype:
            dw = dw.to(weight.dtype)
        if ctx.x_layout == torch.contiguous_format:
            gx = gx.contiguous()
        return gx, dw, db


class ConvTranspose3dWmma(nn.ConvTranspose3d):
    """k=2 s=2 pad=0 drop-in. Other configs fall back to MIOpen."""

    def forward(self, x):
        k = self.kernel_size
        if (k != (2, 2, 2) or self.stride != (2, 2, 2) or self.padding != (0, 0, 0)
                or self.output_padding != (0, 0, 0) or self.groups != 1 or not x.is_cuda):
            return super().forward(x)
        x, weight, bias = _autocast_args(x, self.weight, self.bias)
        return ConvTranspose3dWmmaFn.apply(x, weight, bias)
