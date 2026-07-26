"""手部姿态 → 少数几个低噪标量/锚点. 不碰画布, 不知道有"盒子"这回事.

renderer(mirror/banner)和 glassbox(screen)都从这里取手部量, 所以它是共享叶子。
每个函数只做一件事, 且都有原片实测背书——为什么用这个点而不是那个点, 见
docs/GLASS_BOX_GEOMETRY.md。
"""

from __future__ import annotations

import numpy as np

from .landmarks import INDEX_DIP, INDEX_MCP, INDEX_TIP, PALM_RING, PINKY_DIP, PINKY_MCP, PINKY_TIP, THUMB_TIP, WRIST
from .tracker import HandPose

ORIENT_GAIN = 2.2  # 手掌朝向→明暗的灵敏度(越大翻手反应越猛)
_PALM_IDX = np.array(PALM_RING, dtype=np.int32)


def palm_center(hand: HandPose) -> np.ndarray:
    return hand.points[_PALM_IDX, :2].mean(axis=0).astype(np.float32)


def pinch(hand: HandPose) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """index tip, thumb tip, pinch center, pinch gap."""
    i = hand.points[INDEX_TIP, :2].astype(np.float32)
    t = hand.points[THUMB_TIP, :2].astype(np.float32)
    return i, t, (i + t) * 0.5, float(np.linalg.norm(i - t))


def orient(hand: HandPose) -> float:
    """手掌朝向 -1..1: 掌心/掌背对镜头时饱和, 侧立≈0.

    用手腕-食指根-小指根三点的 2D 有向面积——翻手腕时剧烈过零,
    比指尖 z 深度差稳定得多。
    """
    p = hand.points
    u = p[INDEX_MCP, :2] - p[WRIST, :2]
    v = p[PINKY_MCP, :2] - p[WRIST, :2]
    area = float(u[0] * v[1] - u[1] * v[0])
    scale = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)), 20.0)
    sign = 1.0 if hand.handedness == "Right" else -1.0
    return float(np.clip(sign * ORIENT_GAIN * area / (scale * scale), -1.0, 1.0))


def grip(hand: HandPose) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """一只手贡献给长方体的量: 掌心 / 指弧中点 / 指弧展开量 / 掌宽 / 掌面朝向.

    锚点不取单点, 而是在 掌心↔指弧中点 之间插值(BOX_ANCHOR_LIFT):
      掌心(PALM_RING 均值)最稳但偏腕——纯用它盒子会挂在手下方;
      指弧中点是原片实测位置(帧 312 距盒子前面中心 34px, 掌心差 102px);
      插值兼顾两者: 靠近指弧恢复原片高度, 掺一点掌心衰减指尖噪声。
    掌宽 |MCP5−MCP17| 是单目深度线索——同一只手离镜头越近投影越大, 两手的
    掌宽比给出长轴的深度分量, 盒子因此能指向镜头外的方向而不只是绕长轴滚。
    """
    xy = hand.points[:, :2].astype(np.float32)
    i = xy[INDEX_TIP] * 0.75 + xy[INDEX_DIP] * 0.25
    p = xy[PINKY_TIP] * 0.75 + xy[PINKY_DIP] * 0.25
    palm = float(np.linalg.norm(xy[INDEX_MCP] - xy[PINKY_MCP]))
    return (
        palm_center(hand),
        (i + p) * 0.5,
        float(np.linalg.norm(i - p)),
        palm,
        orient(hand),
    )
