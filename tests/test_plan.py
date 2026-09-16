"""CPU-safe tests: geometry dispatch and tile picker. No GPU required."""
import os

os.environ.setdefault("RDNA4_CONV3D_BACKEND", "hybrid")

from rdna4_conv3d.conv3d_wmma import _geom_ok, _pick_bk, _pick_tiles, conv3d_plan


def test_geom_ok_k1_k3_only():
    assert _geom_ok(3, 1, 1, 1)
    assert _geom_ok(1, 1, 1, 1)
    assert _geom_ok((3, 3, 3), (1, 1, 1), (1, 1, 1), 1)
    assert not _geom_ok(5, 1, 1, 1)
    assert not _geom_ok(3, 2, 1, 1)
    assert not _geom_ok(3, 1, 2, 1)
    assert not _geom_ok(3, 1, 1, 2)


def test_pick_bk_fit_not_straddle():
    assert _pick_bk(24) == 16
    assert _pick_bk(48) == 16
    assert _pick_bk(32) == 32
    assert _pick_bk(96) == 32
    assert _pick_bk(192) == 32


def test_wgrad_tiles_split_on_cout():
    # 16 < M=Cout <= 32 → bk=32 warps=8
    bm, bn, bk, warps, stages = _pick_tiles(24, 24 * 27, 64 * 128 * 128, "wgrad")
    assert bk == 32 and warps == 8 and stages == 2
    # Cout=48 stays on the wide config
    _, _, bk2, warps2, _ = _pick_tiles(48, 48 * 27, 32 * 64 * 64, "wgrad")
    assert bk2 == 64 and warps2 == 4
    # Cout=1 must not take 8 warps
    _, _, _, warps1, _ = _pick_tiles(1, 24 * 1, 64 * 128 * 128, "wgrad")
    assert warps1 == 4


def test_plan_values_are_wmma_or_miopen():
    plan = conv3d_plan(24, 24, 3)
    assert set(plan) == {"fwd", "dw", "dx"}
    assert plan["fwd"] in ("wmma", "miopen")
    # grouped / stride-2 always miopen
    slow = conv3d_plan(24, 24, 3, stride=2)
    assert slow == {"fwd": "miopen", "dw": "miopen", "dx": "miopen"}
