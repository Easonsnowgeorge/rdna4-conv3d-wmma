"""Optional rocWMMA HIP fprop. Compiled on first use against gfx1201."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load
    src = Path(__file__).with_name("conv3d_wmma_hip.hip")
    extra = ["-O3", "-ffast-math"]
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    # gcnArchName can look like "gfx1201:xnack-"
    gfx = arch.split(":")[0] if arch else "gfx1201"
    extra.append(f"--offload-arch={gfx}")
    return load(
        name="rdna4_conv3d_wmma_hip",
        sources=[str(src)],
        extra_cuda_cflags=extra,
        extra_include_paths=["/opt/rocm/include"],
        verbose=False,
    )


def hip_conv3d_fwd(x, weight, bias, padding):
    """x, weight in NCDHW/OIDHW. Returns NDHWC fp16 activation."""
    if x.dtype != torch.float16:
        return None
    x_cl = x.contiguous(memory_format=torch.channels_last_3d)
    packed = weight.permute(2, 3, 4, 1, 0).contiguous().view(-1, weight.shape[0])
    pd, ph, pw = padding
    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)
    return _mod().fwd_k3(x_cl, packed, bias, int(pd), int(ph), int(pw))
