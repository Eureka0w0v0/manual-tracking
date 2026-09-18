"""mirror — 折纸镜面(抖音 manualtracking / AM, v1 前半).

四角钉在双手拇指尖+食指尖; 平摊 = 单面, 顶边×底边相交 = 麻花双翼(右翼在前),
双手捏死 = 双瓣细白线。面 = 反相镜面 clamp(283 − 0.56×bg), 折起的面采样点
外移并乘冷灰。描边只描自由边, 带 ±2px 色差晕。

参数全部来自对原视频的逐像素逆向测量, 改之前先读 docs/GLASS_BOX_GEOMETRY.md。
这一层只吃**已经分好左右**的两只手(角色滞回归 renderer 管, 与 screen 共用)。
"""

from __future__ import annotations

import cv2
import numpy as np

from .effects import fx_mirror
from .glassbox import MIN_SPAN_PX
from .handgeom import orient, pinch
from .paint import EDGE, edge_line, fill, seg_cross
from .tracker import HandPose

# ---- tunables ----
PINCH_SHUT_PX = 16.0  # 双手捏距都小于此值 → 侧视细线(实测捏合张开 10-34px@1920)
BASE_B = 0.85  # 默认亮度(纸平摊时接近亮白)
B_SWING = 0.65  # 翻手带来的亮度摆幅
COOL_START = 0.7  # 亮度低于此值开始变冷
COOL_RATE = 1.8  # 变冷速度
MIRROR_SHIFT = 0.35  # 折起的面采样点外移量(跨距比例)
FRINGE_WARM = (40, 150, 255)  # 描边色差晕: 亮侧橙
FRINGE_COOL = (235, 225, 90)  # 描边色差晕: 暗侧青


def draw(canvas: np.ndarray, frame_bgr: np.ndarray, left: HandPose, right: HandPose) -> None:
    iL, tL, cL, gapL = pinch(left)
    iR, tR, cR, gapR = pinch(right)

    span_v = cR - cL
    span = float(np.linalg.norm(span_v))
    if span < MIN_SPAN_PX:
        return

    # 双手都捏死 → 纸转到侧面: 双瓣白线(两条白线夹一道细缝, 实测无光晕)
    if max(gapL, gapR) < PINCH_SHUT_PX:
        perp = np.array([-span_v[1], span_v[0]], np.float32) / span
        for s in (-2.0, 2.0):
            edge_line(canvas, cL + perp * s, cR + perp * s)
        return

    u = span_v / span

    # 每半张纸的明暗由那只手的手掌朝向决定: 默认亮白, 翻手变暗
    bL = float(np.clip(BASE_B + B_SWING * orient(left), 0.2, 1.0))
    bR = float(np.clip(BASE_B + B_SWING * orient(right), 0.2, 1.0))

    # 角点钉在指尖上(实测偏差 ≤0.13×捏距): TL=左食指 BL=左拇指 BR=右拇指 TR=右食指
    # 顶边×底边相交 → 麻花态: 两个交叉三角翼, 右翼后画(压在前面)
    x = seg_cross(iL, iR, tR, tL)
    faces: list[tuple[np.ndarray, float, float]]  # (poly, b, 采样偏移方向)
    if x is not None:
        faces = [
            (np.array([iL, x, tL], np.float32), bL, -1.0),
            (np.array([x, iR, tR], np.float32), bR, +1.0),
        ]
    else:
        faces = [(np.array([iL, iR, tR, tL], np.float32), (bL + bR) * 0.5, 0.0)]

    for poly, b, side in faces:
        # 折起的面: 镜面采样点沿跨距方向外移, 采到别处(亮墙反相成暗面)
        shift_v = u * (side * (1.0 - b) * MIRROR_SHIFT * span)
        cool = float(np.clip((COOL_START - b) * COOL_RATE, 0.0, 1.0))
        fill(canvas, frame_bgr, poly, fx_mirror(cool), (float(shift_v[0]), float(shift_v[1])))
        pr = np.round(poly).astype(np.int32)
        # 描边只描自由边(整面轮廓), 附 ±2px 色差晕
        cv2.polylines(canvas, [pr + (2, 1)], True, FRINGE_WARM, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [pr - (2, 1)], True, FRINGE_COOL, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [pr], True, EDGE, 3, cv2.LINE_AA)
