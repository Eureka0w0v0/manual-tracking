"""合成手: 不开摄像头、不跑 MediaPipe, 直接造 HandPose 喂给几何层.

为什么 21 个点全都要设:
  掌心走 PALM_RING 六点均值, glassbox.grip() 还要 INDEX_DIP/PINKY_DIP/PINKY_TIP。
  少设一个, 那个 (0,0) 就把结果往画面左上角拽 —— 测出来的"跟手程度"是假的,
  而且不报错。所以这里按真实手的解剖比例把 21 个点一次铺满。

坐标系: 屏幕像素, +x 右 / +y 下。手指朝上(−y), 掌宽 = PALM px。
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from manual_tracking.landmarks import (  # noqa: E402
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from manual_tracking.tracker import HandPose  # noqa: E402

SHAPE = (720, 1280)  # (h, w)
PALM = 160.0  # 掌宽 |MCP5−MCP17| 的像素尺度

# landmark → 相对手中心的偏移(掌宽的倍数)。掌根那六个点是 cube_check 一路用下来
# 的原始布局, 不要动 —— 动了 palm_center 就变, 它那些"手走 300px 立方体走 300px"
# 的断言数值会跟着漂。指节是按真实手比例补的, 只影响 grip()/orient() 这类量。
# 掌根六点(palm_center / 掌宽由它们决定, spread 不碰):
_PALM_LAYOUT: dict[int, tuple[float, float]] = {
    WRIST: (0.0, 0.9),
    THUMB_CMC: (-0.55, 0.7),
    INDEX_MCP: (-0.5, 0.0),
    MIDDLE_MCP: (-0.15, -0.05),
    RING_MCP: (0.18, 0.0),
    PINKY_MCP: (0.5, 0.1),
}
# 指部 13 点(spread 缩放它们相对手中心的偏移 = 张开/握拳):
_FINGER_LAYOUT: dict[int, tuple[float, float]] = {
    # 拇指: 从 CMC 斜着往外上方伸
    THUMB_MCP: (-0.78, 0.42),
    THUMB_IP: (-0.92, 0.16),
    # 四指: MCP → PIP → DIP → TIP 逐节向上(−y), 中指最长、小指最短
    INDEX_PIP: (-0.52, -0.38),
    INDEX_DIP: (-0.53, -0.58),
    MIDDLE_PIP: (-0.17, -0.46),
    MIDDLE_DIP: (-0.18, -0.68),
    MIDDLE_TIP: (-0.18, -0.86),
    RING_PIP: (0.19, -0.40),
    RING_DIP: (0.20, -0.60),
    RING_TIP: (0.21, -0.76),
    PINKY_PIP: (0.53, -0.24),
    PINKY_DIP: (0.55, -0.40),
    PINKY_TIP: (0.56, -0.54),
}


def hand(
    cx: float,
    cy: float,
    *,
    pinch: bool,
    tid: int = 0,
    handedness: str = "Right",
    palm: float = PALM,
    spread: float = 1.0,
) -> HandPose:
    """一只合成手: 整体刚性平移, 捏合时拇指尖贴到食指尖, 松开时拉开 0.9 掌宽.

    spread 缩放**指部**相对手中心的偏移(掌根/掌宽不动): 1.0 = 半开(历史姿态,
    既有断言全部建立在它上面), 2.0 = 五指张开(指弧/掌宽比 ≈ 1.13, 过
    EXPLODE_HI), 0.5 = 握拳(比值 ≈ 0.28)。
    """
    p = np.zeros((21, 3), np.float32)
    for lm, (fx, fy) in _PALM_LAYOUT.items():
        p[lm] = (cx + palm * fx, cy + palm * fy, 0.0)
    for lm, (fx, fy) in _FINGER_LAYOUT.items():
        p[lm] = (cx + palm * fx * spread, cy + palm * fy * spread, 0.0)
    p[INDEX_TIP] = (cx, cy - palm * 0.6 * spread, 0.0)
    off = 0.0 if pinch else palm * 0.9
    p[THUMB_TIP] = (cx - off, cy - palm * 0.6 * spread, 0.0)
    return HandPose(handedness=handedness, score=1.0, points=p, track_id=tid)


def pair2(cx: float, half: float = 200.0, cy: float = 400.0) -> list[HandPose]:
    """一对都捏住的手, 中点在 cx, 相距 2*half."""
    return [hand(cx - half, cy, pinch=True, tid=0), hand(cx + half, cy, pinch=True, tid=1)]
