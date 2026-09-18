"""FaceEffect 的输入输出契约.

paint.fill 拿 fx 的返回值直接 `cv2.copyTo` 进画布 ROI, 所以:

1. **同尺寸同 dtype** —— 差一个像素就是 OpenCV 层的断言崩溃, 不是画歪。
   四叉树 / 半调 / 点云都按格子分块, 所以专挑不能被块大小整除的奇数尺寸喂。
2. **不许改输入** —— `paint.poly_window` 交给 fx 的是 `frame_bgr` 的**切片视图**,
   不是副本。哪个 effect 原地写一笔, 就会污染原始摄像头帧, 然后后面所有面
   都吃到脏数据, 表现为"某个面的效果渗到别的面上"。
3. **不许从对侧搬内容** —— 面的像素只能来自面内对应位置附近。riso 的通道
   错位曾用 np.roll 实现, 那是环绕的(见最后一条)。
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


def test_riso_channel_split_does_not_wrap():
    """riso 的通道错位不许把内容从面的对侧卷过来.

    实现是三个通道各在错开的位置上做阈值。那个错位原先用 np.roll 做, 而
    np.roll 是**环绕**的: 最左 split 列会原样出现在最右边 —— 实测左白右黑
    的输入, 最右一列 B 通道从 20 跳到 238。那条 5px 宽的假套色带偏偏长得
    像 riso 本身的套色不准, 肉眼认不出是 bug, 但它的内容来自面的另一侧,
    盒子一转就跟着乱变。现在改成 BORDER_REPLICATE 扩边取窗。
    """
    riso = next(f.fx for f in BOX_FACES if f.tag == "前")
    src = np.zeros((40, 60, 3), np.uint8)
    src[:, :30] = 255  # 左半纯白 / 右半纯黑, 交界正好在中线
    out = riso(src)
    # 纯黑那半的最右一列必须和它内侧同色(都在暗部, 三通道一致)
    assert np.array_equal(out[:, -1], out[:, -10]), "右边缘卷进了左半的内容"
    assert np.array_equal(out[:, 0], out[:, 9]), "左边缘卷进了右半的内容"
    # 但明暗交界处的套色带**必须**还在 —— 修边界不能顺手把效果本身削平
    seam = {tuple(int(v) for v in px) for px in out[20, 24:36]}
    assert len(seam) >= 2, f"交界处的通道错位消失了: {seam}"
