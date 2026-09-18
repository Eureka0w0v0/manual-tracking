"""新渲染路径的冒烟契约: 不开摄像头, 合成手直接喂 render().

这些路径(炸开/霓虹/涟漪/拖影)都带几何投影或 ROI 数学, "改错了"表现为
OpenCV 断言崩溃或 ROI 越界 —— 冒烟就能抓到; 好不好看归哥哥实拍。
"""

import numpy as np
import pytest

from manual_tracking.renderer import VectorOverlayRenderer
from manual_tracking.tracker import FrameHands
from synth import hand

H, W = 720, 1280


@pytest.fixture
def frame(rng) -> np.ndarray:
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def test_cube_explodes_and_returns(frame):
    """张开手 → explode→1 且画面确实变了; 握拳 → 回到 0 → 画面回到常规形态."""
    r = VectorOverlayRenderer(style="cube")
    out_closed = None
    for k in range(30):
        out_closed = r.render(frame, FrameHands(k, [hand(640, 400, pinch=False, spread=1.0)]))
    assert r.cube.explode < 0.05

    out_open = None
    for k in range(30, 60):
        out_open = r.render(frame, FrameHands(k, [hand(640, 400, pinch=False, spread=2.0)]))
    assert r.cube.explode > 0.9
    assert out_open.shape == frame.shape and out_open.dtype == np.uint8
    diff = float(np.abs(out_open.astype(np.int16) - out_closed.astype(np.int16)).mean())
    assert diff > 1.0, "炸开前后画面几乎没变, exploded 路径没生效"

    for k in range(60, 90):
        r.render(frame, FrameHands(k, [hand(640, 400, pinch=False, spread=0.5)]))
    assert r.cube.explode < 0.05


def test_exploded_render_is_finite_at_extremes(frame):
    """炸开时把立方体推到画面角落: 投影/ROI 数学不许越界或产生非法值."""
    r = VectorOverlayRenderer(style="cube")
    for k in range(30):
        r.render(frame, FrameHands(k, [hand(120, 90, pinch=False, spread=2.0)]))
    r.cube.pos = np.array([30.0, 20.0], np.float32)  # 角落, 面片大部分出画
    out = r.render(frame, FrameHands(99, [hand(120, 90, pinch=False, spread=2.0)]))
    assert out.shape == frame.shape


def test_ripple_draws_without_error(frame):
    """在盒上捏合(抓取建立)出涟漪, 画完整个生命周期不出错."""
    r = VectorOverlayRenderer(style="cube")
    r.render(frame, FrameHands(0, [hand(640, 456, pinch=False)]))
    out = r.render(frame, FrameHands(1, [hand(640, 456, pinch=True)]))
    assert len(r.cube.ripples) == 1, "在盒上捏合应产生涟漪"
    for k in range(2, 11):
        out = r.render(frame, FrameHands(k, [hand(640, 456, pinch=True)]))
    assert out.shape == frame.shape


def test_wire_neon_renders_and_glows(frame):
    """霓虹骨架: 画面变亮(加法辉光), 手贴画面边缘时 ROI clamp 不越界."""
    r = VectorOverlayRenderer(style="wire")
    base = r.render(frame, FrameHands(0, []))
    lit = r.render(frame, FrameHands(1, [hand(640, 400, pinch=False)]))
    assert float(lit.astype(np.int16).mean()) > float(base.astype(np.int16).mean())
    for k, (cx, cy) in enumerate([(5, 5), (W - 5, 5), (5, H - 5), (W - 5, H - 5)]):
        out = r.render(frame, FrameHands(2 + k, [hand(cx, cy, pinch=False)]))
        assert out.shape == frame.shape


def test_screen_ease_grows_box(frame):
    """screen 出现动画: 前几帧盒子从小长大(ease), 而不是第一帧就全尺寸."""
    r = VectorOverlayRenderer(style="screen")
    pair = [hand(400, 360, pinch=False, tid=0), hand(900, 360, pinch=False, tid=1)]
    r.render(frame, FrameHands(0, pair))
    early = r.box._ease
    for k in range(1, 10):
        r.render(frame, FrameHands(k, pair))
    assert 0.0 < early < 0.5, f"第一帧 ease 应在爬坡途中, 实为 {early}"
    assert r.box._ease > 0.99, "10 帧后应长到全尺寸"
