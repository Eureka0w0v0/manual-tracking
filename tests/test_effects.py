"""FaceEffect 的输入输出契约.

renderer._fill 拿 fx 的返回值直接 `cv2.copyTo` 进画布 ROI, 所以:

1. **同尺寸同 dtype** —— 差一个像素就是 OpenCV 层的断言崩溃, 不是画歪。
   四叉树 / 半调 / 点云都按格子分块, 所以专挑不能被块大小整除的奇数尺寸喂。
2. **不许改输入** —— `_poly_window` 交给 fx 的是 `frame_bgr` 的**切片视图**,
   不是副本。哪个 effect 原地写一笔, 就会污染原始摄像头帧, 然后后面所有面
   都吃到脏数据, 表现为"某个面的效果渗到别的面上"。
"""

import numpy as np
import pytest

from manual_tracking.effects import BLUE_LUT, BOX_FACES, XRAY_CMAP, fx_mirror

ALL_FX = [pytest.param(f.fx, id=f.tag) for f in BOX_FACES] + [
    pytest.param(fx_mirror(0.0), id="mirror-平"),
    pytest.param(fx_mirror(1.0), id="mirror-冷"),
]

# 1x1 = 退化; 3x7 / 17x23 = 素数边长, 躲不开任何分块逻辑的余数分支;
# 64x64 = 正常大小的对照
SIZES = [(1, 1), (3, 7), (17, 23), (64, 64)]


@pytest.mark.parametrize("fx", ALL_FX)
@pytest.mark.parametrize(("h", "w"), SIZES)
def test_shape_and_dtype_preserved(fx, h, w, rng):
    src = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    out = fx(src)
    assert out.shape == src.shape, f"{h}x{w}: {out.shape} != {src.shape}"
    assert out.dtype == np.uint8, out.dtype


@pytest.mark.parametrize("fx", ALL_FX)
def test_input_not_mutated(fx, rng):
    """输入是画布的视图, 原地改会污染整帧."""
    src = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    before = src.copy()
    fx(src)
    assert np.array_equal(src, before), "effect 原地改了输入帧"


@pytest.mark.parametrize("fx", ALL_FX)
def test_handles_flat_input(fx, rng):
    """全同色输入: 方差为 0, 是四叉树递归和局部对比度伪深度的退化分支."""
    src = np.full((24, 40, 3), 128, np.uint8)
    out = fx(src)
    assert out.shape == src.shape
    assert np.isfinite(out.astype(np.float32)).all()


@pytest.mark.parametrize("lut", [BLUE_LUT, XRAY_CMAP])
def test_colormap_layout(lut):
    """cv2.applyColorMap 只吃 256x1x3 的 uint8 表, 形状错了当场抛."""
    assert lut.shape == (256, 1, 3), lut.shape
    assert lut.dtype == np.uint8, lut.dtype
