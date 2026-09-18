"""banner — TouchDesigner 横幅(v2).

四角 = 双手食指尖(上边) + 拇指尖(下边)。四层条带: 黄阈值头带 / X-ray 中窗
(solarize, 左右内缩 7% + 黄侧线) / 白软阈值分隔线 / 悬出画外的红脚带。内容
全部是摄像头画面的屏幕空间 1:1 双色调变换, 没有整板描边。

这一层只吃**已经分好左右**的两只手(角色滞回归 renderer 管, 三种风格共用)。
"""

from __future__ import annotations

import numpy as np

from .effects import BANNER_YELLOW, RED_CMAP, WHITE_CMAP, XRAY_CMAP, YELLOW_CMAP, _fx_lut
from .glassbox import MIN_SPAN_PX
from .handgeom import pinch
from .paint import edge_line, fill
from .tracker import HandPose


def draw(canvas: np.ndarray, frame_bgr: np.ndarray, left: HandPose, right: HandPose) -> None:
    iL, tL, cL, _ = pinch(left)
    iR, tR, cR, _ = pinch(right)
    if float(np.linalg.norm(cR - cL)) < MIN_SPAN_PX:
        return

    def band(v0: float, v1: float, u0: float = 0.0, u1: float = 1.0) -> np.ndarray:
        """v: 食指边(0)→拇指边(1)可越界; u: 左手(0)→右手(1)."""
        rows = []
        for v in (v0, v1):
            lv = iL + (tL - iL) * v
            rv = iR + (tR - iR) * v
            rows.append((lv * (1 - u0) + rv * u0, lv * (1 - u1) + rv * u1))
        (p00, p01), (p10, p11) = rows
        return np.array([p00, p01, p11, p10], np.float32)

    # 实测分层: 黄头带 / X-ray 中窗(左右内缩7%) / 白分隔线 / 悬出的红脚带
    mid = band(0.20, 0.94, 0.07, 0.93)
    fill(canvas, frame_bgr, mid, _fx_lut(XRAY_CMAP))
    fill(canvas, frame_bgr, band(0.0, 0.20), _fx_lut(YELLOW_CMAP))
    fill(canvas, frame_bgr, band(0.94, 1.02), _fx_lut(WHITE_CMAP))
    fill(canvas, frame_bgr, band(1.02, 1.28), _fx_lut(RED_CMAP))
    # 中窗左右侧缘的黄色细线(原效果唯一的"描边")
    edge_line(canvas, mid[0], mid[3], BANNER_YELLOW, 2)
    edge_line(canvas, mid[1], mid[2], BANNER_YELLOW, 2)
