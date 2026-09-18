"""wire — 霓虹电流骨架: 辉光 + 芯线 + 沿骨骼流动的光点.

辉光 = 粗线画进黑图层 → 高斯模糊 → **加法**混合回画布(自发光, 不是覆盖)。
模糊只在手部 bbox 里做: 1080p 全幅模糊 ~8ms, bbox 只要 ~0.3ms。

wire 曾经只是"纯骨架调试"; 现在它是一种正式风格, 但仍然是最便宜的那个 ——
不碰盒子几何, 也不采样背景。
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import CONNECTIONS, TIP_IDS
from .paint import WHITE_HOT
from .tracker import HandPose

NEON_GLOW = (255, 160, 40)  # 辉光层(电青蓝 BGR)
NEON_CORE = (255, 240, 210)  # 芯线(近白偏青)
NEON_FLOW = 0.06  # 流动光点的相位步进(/帧); 一根骨头 ~0.5s 走完
GLOW_PAD = 48  # 辉光 bbox 外扩(px) —— 模糊会溢出骨架本身的包围盒
GLOW_BLUR = 6  # 高斯 sigma(px)


def draw(canvas: np.ndarray, hand: HandPose, phase: int) -> None:
    pts = hand.as_int()
    h, w = canvas.shape[:2]
    x0 = max(int(pts[:, 0].min()) - GLOW_PAD, 0)
    y0 = max(int(pts[:, 1].min()) - GLOW_PAD, 0)
    x1 = min(int(pts[:, 0].max()) + GLOW_PAD, w)
    y1 = min(int(pts[:, 1].max()) + GLOW_PAD, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    roi = canvas[y0:y1, x0:x1]
    glow = np.zeros_like(roi)
    lp = pts - (x0, y0)
    for a, b in CONNECTIONS:
        cv2.line(glow, tuple(lp[a]), tuple(lp[b]), NEON_GLOW, 5, cv2.LINE_AA)
    cv2.GaussianBlur(glow, (0, 0), GLOW_BLUR, dst=glow)
    cv2.add(roi, glow, dst=roi)
    for a, b in CONNECTIONS:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), NEON_CORE, 2, cv2.LINE_AA)
    # 流动光点: 每根骨头一个, 相位错开 —— "电流"在骨架里跑
    for k, (a, b) in enumerate(CONNECTIONS):
        t = (phase * NEON_FLOW + k * 0.37) % 1.0
        p = pts[a] + (pts[b] - pts[a]).astype(np.float32) * t
        c = (int(p[0]), int(p[1]))
        cv2.circle(canvas, c, 4, NEON_GLOW, -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 2, WHITE_HOT, -1, cv2.LINE_AA)
    for tid in TIP_IDS:
        cv2.circle(canvas, (int(pts[tid, 0]), int(pts[tid, 1])), 3, WHITE_HOT, -1, cv2.LINE_AA)
