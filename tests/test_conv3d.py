"""Numerical checks. Skip when there is no CUDA. On non-RDNA4 the module is a fallback."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdna4_conv3d import Conv3dWmma, conv3d_wmma, swap_conv3d


def _cuda():
    return torch.cuda.is_available()


def _rdna4():
    if not _cuda():
        return False
    name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return "gfx120" in name


pytestmark = pytest.mark.skipif(not _cuda(), reason="CUDA/ROCm GPU required")


def test_swap_copies_weights():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv3d(4, 8, 3, padding=1)

    m = M()
    w = m.c.weight.data.clone()
    n = swap_conv3d(m)
    assert n == 1
    assert isinstance(m.c, Conv3dWmma)
    assert torch.equal(m.c.weight.data, w)


def test_fallback_matches_conv3d_on_unsupported_geom():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8, 8, 8, device="cuda", dtype=torch.float32)
    conv = Conv3dWmma(4, 5, 3, stride=2, padding=1).cuda()
    y = conv(x)
    y_ref = F.conv3d(x, conv.weight, conv.bias, 2, 1)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _rdna4(), reason="WMMA path is gfx120x-only")
@pytest.mark.parametrize(
    "n,cin,cout,d,h,w,k,pad",
    [
        (2, 5, 7, 9, 11, 13, 3, 1),
        (2, 24, 1, 8, 16, 16, 1, 0),
        (1, 24, 24, 8, 16, 16, 3, 1),
        (1, 1, 24, 8, 16, 16, 3, 1),
    ],
)
def test_wmma_close_to_fp16_miopen(n, cin, cout, d, h, w, k, pad):
    torch.manual_seed(0)
    x = torch.randn(n, cin, d, h, w, device="cuda", dtype=torch.float16, requires_grad=True)
    wt = torch.randn(cout, cin, k, k, k, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn(cout, device="cuda", dtype=torch.float16, requires_grad=True)
    x2, w2, b2 = (
        x.detach().requires_grad_(True),
        wt.detach().requires_grad_(True),
        b.detach().requires_grad_(True),
    )
    y = conv3d_wmma(x, wt, b, 1, pad)
    y2 = F.conv3d(x2, w2, b2, 1, pad)
    torch.testing.assert_close(y, y2, atol=2e-2, rtol=8e-2)
    go = torch.randn_like(y2)
    y.backward(go)
    y2.backward(go)
    torch.testing.assert_close(x.grad, x2.grad, atol=8e-2, rtol=8e-2)
    # dW is a large-K reduction; keep the bound loose, matching the isolated benches
    torch.testing.assert_close(wt.grad, w2.grad, atol=0.5, rtol=0.5)
