"""绘制原语: 把一块多边形填成"被某种像素处理过的背景". 不知道有"手"这回事.

五种风格唯一的上色出口都在这里。早先有 _cmap_fill / _fx_fill / _mirror_fill
三个函数, 骨架逐字相同、只有"怎么把 src 变成颜色"那一行不同; 那一行外提成
effects.FaceEffect 之后就只剩这一个 fill()。

跟 effects 的分工: effects 决定**像素怎么变**, 这里决定**画在哪块区域、以
什么透明度和亮度叠上去**。两边都不知道手和盒子的存在。
"""

from __future__ import annotations

import cv2
import numpy as np

from .effects import FaceEffect

# 共享颜色(不止一个风格用到; 只有一处用的就地写在那个模块里)
EDGE = (255, 255, 255)  # 自由边描边(实测原片纯白 4px@1080p)
WHITE_HOT = (230, 250, 255)  # 骨架关节/指尖的高光点


def seg_cross(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray | None:
    """Intersection of segments p1p2 / p3p4 (interior only), else None."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-6:
        return None
    r = p3 - p1
    t = float(r[0] * d2[1] - r[1] * d2[0]) / denom
    u = float(r[0] * d1[1] - r[1] * d1[0]) / denom
    if 0.02 < t < 0.98 and 0.02 < u < 0.98:
        return (p1 + d1 * t).astype(np.float32)
    return None


def edge_line(
    canvas: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int] = EDGE,
    width: int = 3,
) -> None:
    cv2.line(
        canvas,
        (int(round(float(a[0]))), int(round(float(a[1])))),
        (int(round(float(b[0]))), int(round(float(b[1])))),
        color,
        width,
        cv2.LINE_AA,
    )


def poly_window(
    canvas: np.ndarray,
    frame_bgr: np.ndarray,
    poly: np.ndarray,
    shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]] | None:
    """bbox 局部窗口: (画布 roi, 同尺寸源帧窗口(可偏移, 贴边钳制), 多边形掩码, 原点)."""
    h, w = canvas.shape[:2]
    p = np.round(poly).astype(np.int32)
    x0, y0 = max(0, p[:, 0].min() - 1), max(0, p[:, 1].min() - 1)
    x1, y1 = min(w, p[:, 0].max() + 2), min(h, p[:, 1].max() + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0
    sx = int(np.clip(x0 + shift[0], 0, w - bw))
    sy = int(np.clip(y0 + shift[1], 0, h - bh))
    mask = np.zeros((bh, bw), np.uint8)
    cv2.fillPoly(mask, [p - (x0, y0)], 255)
    return canvas[y0:y1, x0:x1], frame_bgr[sy : sy + bh, sx : sx + bw], mask, (x0, y0)


def fill(
    canvas: np.ndarray,
    frame_bgr: np.ndarray,
    poly: np.ndarray,
    fx: FaceEffect,
    shift: tuple[float, float] = (0.0, 0.0),
    alpha: float = 1.0,
    shade: float = 1.0,
) -> None:
    """把 poly 围出的区域填成 fx(背景窗口(uv+shift)).

    alpha<1 时与画布已有内容混合 —— 玻璃盒靠它拿到"透"的质感; shade<1 时
    整面压暗 —— 环境光照(Lambert)从这里进, 亮度是乘法, 透明度是混合, 两个
    通道互不污染。
    """
    got = poly_window(canvas, frame_bgr, poly, shift)
    if got is None:
        return
    roi, src, mask, _ = got
    out = np.ascontiguousarray(fx(src))
    if shade < 0.999:
        cv2.convertScaleAbs(out, dst=out, alpha=shade)
    if alpha < 1.0:
        out = cv2.addWeighted(out, alpha, roi, 1.0 - alpha, 0.0)
    cv2.copyTo(out, mask, roi)
