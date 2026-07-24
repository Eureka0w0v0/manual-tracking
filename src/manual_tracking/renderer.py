"""Hand overlay renderer — 参数全部来自对原视频的逐像素逆向测量.

mirror — 折纸镜面(抖音 manualtracking / AM, v1 前半):
  四角钉在双手拇指尖+食指尖; 平摊=单面, 顶边×底边交叉=麻花双翼(右翼在前),
  双手捏死=双瓣细白线。面 = 反相镜面 clamp(283 - 0.56*bg), 折起的面采样点
  外移并乘冷灰。描边只描自由边, 带 ±2px 色差晕。

screen — 彩色玻璃盒(v1 后半):
  盒长=两手食指尖距, 端棱=食指尖→小指尖(C 形捧握), 顶面向上后挤出。
  面色 = 每面一张实测亮度 LUT(顶面蓝=反相型, 前面绿=正相, 翻背=红),
  顶面采样点下移(折射感)+横条 glitch, 全棱白描边, 对角折痕可见。

banner — TouchDesigner 横幅(v2):
  四角 = 双手食指尖(上边)+拇指尖(下边)。四层条带: 黄阈值头带 /
  X-ray 中窗(solarize, 左右内缩+黄侧线) / 白软阈值分隔线 / 悬出的红脚带,
  内容全部为摄像头画面的屏幕空间 1:1 双色调变换, 无整板描边。

wire — 纯骨架调试。
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .landmarks import (
    CONNECTIONS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_TIP,
    PALM_RING,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)
from .tracker import FrameHands, HandPose

# BGR
GOLD = (40, 170, 255)
ORANGE = (20, 110, 240)
WHITE_HOT = (230, 250, 255)
EDGE = (255, 255, 255)  # free-edge outlines (实测纯白 4px@1080p)
FRINGE_WARM = (40, 150, 255)  # 描边色差晕: 亮侧橙
FRINGE_COOL = (235, 225, 90)  # 描边色差晕: 暗侧青
BANNER_YELLOW = (26, 177, 223)  # #DFB11A
BANNER_Y_DARK = (5, 16, 32)  # #201005
BANNER_RED = (8, 23, 213)  # #D51708
BANNER_R_DARK = (0, 8, 58)  # #3A0800
BANNER_WHITE = (218, 230, 236)  # #ECE6DA
BANNER_W_DARK = (16, 26, 42)  # #2A1A10

TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_PALM_IDX = np.array(PALM_RING, dtype=np.int32)

# ---- 风格注册表(唯一权威; live/__main__ 从这里导入, 不要手抄) ----
STYLES = ("mirror", "screen", "banner", "wire")
STYLE_ALIASES = {
    "fabric": "mirror",
    "frame": "mirror",
    "planes": "mirror",
    "fluid": "mirror",
    "track": "screen",
    "outline": "wire",
}


def canon_style(name: str) -> str:
    s = STYLE_ALIASES.get(name, name)
    return s if s in STYLES else "mirror"


# ---- tunables (哥哥要调效果基本都在这里) ----
MIN_SPAN_PX = 40.0  # 双手跨距小于此值不画特效
PINCH_SHUT_PX = 16.0  # 双手捏距都小于此值 → 侧视细线(实测捏合张开 10-34px@1920)
ORIENT_GAIN = 2.2  # 手掌朝向→明暗的灵敏度(越大翻手反应越猛)
BASE_B = 0.85  # 默认亮度(纸平摊时接近亮白)
B_SWING = 0.65  # 翻手带来的亮度摆幅
# 反相镜面(逐帧实测: face = clamp(283 - 0.56*bg), 暗面再乘冷灰)
MIRROR_A = 283.0
MIRROR_B = 0.56
COOL_G, COOL_R = 0.86, 0.84
COOL_START = 0.7  # 亮度低于此值开始变冷
COOL_RATE = 1.8  # 变冷速度
MIRROR_SHIFT = 0.35  # 折起的面采样点外移量(跨距比例)
BOX_DEPTH = 0.26  # 盒子顶面屏幕进深(跨距比例, 实测 0.21-0.30)
BOX_SAMPLE_K = 0.20  # 顶面采样点下移量(跨距比例, 实测 250-300px@1080)
BOX_UP_BIAS = 0.9  # 顶面挤出方向里混入的固定俯视分量
# 蓝顶面横条 glitch(实测 h15-40 w50-400 @1080p, 按 720p 采集缩放到 2/3)
GLITCH_STRIPS = 3
GLITCH_H = (10, 27)
GLITCH_W = (40, 260)
GLITCH_SHIFT = 40
GLITCH_HOLD = 2  # 每 N 帧换一次图案, 逐帧换会闪成噪声
BANNER_THRESH = 115  # banner 双色调亮度阈值


# ---- 手部几何 ----


def _palm_center(hand: HandPose) -> np.ndarray:
    return hand.points[_PALM_IDX, :2].mean(axis=0).astype(np.float32)


def _left_right(hands: list[HandPose]) -> tuple[HandPose, HandPose]:
    """按掌心屏幕 x 排出画面左手/右手."""
    a, b = sorted(hands[:2], key=lambda h: float(_palm_center(h)[0]))
    return a, b


def _pinch(hand: HandPose) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """index tip, thumb tip, pinch center, pinch gap."""
    i = hand.points[INDEX_TIP, :2].astype(np.float32)
    t = hand.points[THUMB_TIP, :2].astype(np.float32)
    return i, t, (i + t) * 0.5, float(np.linalg.norm(i - t))


def _orient(hand: HandPose) -> float:
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


def _tri_sign(tri: np.ndarray) -> float:
    """三角形有向面积符号(判断面是否翻到背面)."""
    a, b, c = tri[0], tri[1], tri[2]
    return float(np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))


def _seg_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray | None:
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


def _edge_line(
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


# ---- 颜色映射(gray→BGR 的 256 级 colormap, 配 cv2.applyColorMap) ----


def _build_lut(points: list[tuple[float, tuple[float, float, float]]]) -> np.ndarray:
    """把 (背景亮度, BGR) 控制点插值成 256x1x3 colormap (实测 gradient map)."""
    lums = np.array([p[0] for p in points], np.float32)
    chans = np.array([p[1] for p in points], np.float32)
    xs = np.arange(256, dtype=np.float32)
    lut = np.stack([np.interp(xs, lums, chans[:, c]) for c in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8).reshape(256, 1, 3)


def _duotone_cmap(
    dark: tuple[int, int, int],
    bright: tuple[int, int, int],
    thresh: int,
    soft: int = 0,
) -> np.ndarray:
    """双色调 colormap: 亮度阈值拍成两色(soft>0 时软过渡)."""
    xs = np.arange(256, dtype=np.float32)
    if soft > 0:
        w = np.clip((xs - (thresh - soft)) / (2.0 * soft), 0.0, 1.0)
    else:
        w = (xs > thresh).astype(np.float32)
    cm = np.array(dark, np.float32) * (1.0 - w[:, None]) + np.array(bright, np.float32) * w[:, None]
    return np.clip(cm, 0, 255).astype(np.uint8).reshape(256, 1, 3)


def _xray_cmap() -> np.ndarray:
    """中窗模式 A "苍白 X-ray": 以 0.5 为轴的 solarize(输出下限 0.5), 去饱和."""
    g = np.arange(256, dtype=np.float32) / 255.0
    sol = np.clip((0.5 + np.abs(g - 0.5)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sol.reshape(256, 1), cv2.COLOR_GRAY2BGR)


# 逐像素实测的盒子三面 LUT: 顶面蓝=反相型, 前面绿/背面红=正相型
BLUE_LUT = _build_lut(
    [
        (30, (254, 142, 82)),
        (60, (252, 120, 94)),
        (110, (227, 116, 64)),
        (150, (200, 87, 55)),
        (200, (213, 94, 58)),
        (250, (234, 135, 116)),
    ]
)
GREEN_LUT = _build_lut(
    [
        (16, (44, 76, 42)),
        (48, (61, 108, 61)),
        (80, (80, 131, 77)),
        (144, (117, 190, 153)),
        (176, (154, 233, 215)),
        (255, (224, 255, 255)),
    ]
)
RED_LUT = _build_lut(
    [
        (48, (44, 39, 66)),
        (80, (48, 40, 89)),
        (112, (52, 44, 102)),
        (144, (57, 46, 123)),
        (176, (58, 49, 161)),
        (255, (70, 60, 200)),
    ]
)
YELLOW_CMAP = _duotone_cmap(BANNER_Y_DARK, BANNER_YELLOW, BANNER_THRESH)
WHITE_CMAP = _duotone_cmap(BANNER_W_DARK, BANNER_WHITE, BANNER_THRESH, soft=45)
RED_CMAP = _duotone_cmap(BANNER_R_DARK, BANNER_RED, BANNER_THRESH)
XRAY_CMAP = _xray_cmap()


def _mirror_lut(cool: float) -> np.ndarray:
    """反相镜面的逐通道 LUT: clamp(A - B*bg), 冷灰只压 G/R."""
    vals = np.clip(MIRROR_A - MIRROR_B * np.arange(256, dtype=np.float32), 0, 255)
    lut = np.stack(
        [
            vals,
            vals * (1.0 - (1.0 - COOL_G) * cool),
            vals * (1.0 - (1.0 - COOL_R) * cool),
        ],
        axis=1,
    )
    return np.clip(lut, 0, 255).astype(np.uint8).reshape(1, 256, 3)


# ---- 填充原语 ----


def _poly_window(
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


def _cmap_fill(
    canvas: np.ndarray,
    frame_bgr: np.ndarray,
    poly: np.ndarray,
    cmap: np.ndarray,
    shift: tuple[float, float] = (0.0, 0.0),
) -> None:
    """不透明填充: face = cmap[背景亮度(uv+shift)]."""
    got = _poly_window(canvas, frame_bgr, poly, shift)
    if got is None:
        return
    roi, src, mask, _ = got
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.copyTo(cv2.applyColorMap(gray, cmap), mask, roi)


def _mirror_fill(
    canvas: np.ndarray,
    frame_bgr: np.ndarray,
    poly: np.ndarray,
    shift: tuple[float, float],
    cool: float,
) -> None:
    """反相镜面填充(不透明, 逐通道): 平贴时 shift≈0 → 背景以负片鬼影透出."""
    got = _poly_window(canvas, frame_bgr, poly, shift)
    if got is None:
        return
    roi, src, mask, _ = got
    cv2.copyTo(cv2.LUT(src, _mirror_lut(cool)), mask, roi)


class VectorOverlayRenderer:
    """按 style(见模块 docstring)把手部特效画到帧上; 无跨帧状态."""

    def __init__(
        self,
        style: str = "mirror",
        *,
        show_source: bool = True,
        source_dim: float = 0.65,
    ) -> None:
        self.style = canon_style(style)
        self.show_source = show_source
        self.source_dim = float(source_dim)

    def render(self, frame_bgr: np.ndarray, frame_hands: FrameHands) -> np.ndarray:
        if self.show_source:
            if self.source_dim >= 0.98:
                out = frame_bgr.copy()
            else:
                out = cv2.convertScaleAbs(frame_bgr, alpha=self.source_dim, beta=0)
        else:
            out = np.empty_like(frame_bgr)
            out[:] = (8, 6, 12)

        hands = frame_hands.hands
        if len(hands) >= 2:
            if self.style == "mirror":
                self._draw_sheet(out, frame_bgr, hands)
            elif self.style == "screen":
                self._draw_box(out, frame_bgr, hands, frame_hands.index)
            elif self.style == "banner":
                self._draw_banner(out, frame_bgr, hands)

        for i, h in enumerate(hands):
            self._skeleton(out, h, i)
        return out

    def _skeleton(self, canvas: np.ndarray, hand: HandPose, index: int = 0) -> None:
        """Bone lines + joints only. No filled palm graphics."""
        color = GOLD if index == 0 else ORANGE
        pts = hand.as_int()
        for a, b in CONNECTIONS:
            p0 = (int(pts[a, 0]), int(pts[a, 1]))
            p1 = (int(pts[b, 0]), int(pts[b, 1]))
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
        for i in range(21):
            x, y = int(pts[i, 0]), int(pts[i, 1])
            r = 4 if i in TIP_IDS else 3
            cv2.circle(canvas, (x, y), r, WHITE_HOT, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), r, color, 1, cv2.LINE_AA)

    # ---------- mirror: 折纸镜面 ----------

    def _draw_sheet(self, canvas: np.ndarray, frame_bgr: np.ndarray, hands: list[HandPose]) -> None:
        left, right = _left_right(hands)
        iL, tL, cL, gapL = _pinch(left)
        iR, tR, cR, gapR = _pinch(right)

        span_v = cR - cL
        span = float(np.linalg.norm(span_v))
        if span < MIN_SPAN_PX:
            return

        # 双手都捏死 → 纸转到侧面: 双瓣白线(两条白线夹一道细缝, 实测无光晕)
        if max(gapL, gapR) < PINCH_SHUT_PX:
            perp = np.array([-span_v[1], span_v[0]], np.float32) / span
            for s in (-2.0, 2.0):
                _edge_line(canvas, cL + perp * s, cR + perp * s)
            return

        u = span_v / span

        # 每半张纸的明暗由那只手的手掌朝向决定: 默认亮白, 翻手变暗
        bL = float(np.clip(BASE_B + B_SWING * _orient(left), 0.2, 1.0))
        bR = float(np.clip(BASE_B + B_SWING * _orient(right), 0.2, 1.0))

        # 角点钉在指尖上(实测偏差 ≤0.13×捏距): TL=左食指 BL=左拇指 BR=右拇指 TR=右食指
        # 顶边×底边相交 → 麻花态: 两个交叉三角翼, 右翼后画(压在前面)
        x = _seg_cross(iL, iR, tR, tL)
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
            _mirror_fill(canvas, frame_bgr, poly, (float(shift_v[0]), float(shift_v[1])), cool)
            pr = np.round(poly).astype(np.int32)
            # 描边只描自由边(整面轮廓), 附 ±2px 色差晕
            cv2.polylines(canvas, [pr + (2, 1)], True, FRINGE_WARM, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [pr - (2, 1)], True, FRINGE_COOL, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [pr], True, EDGE, 3, cv2.LINE_AA)

    # ---------- screen: 彩色玻璃盒 ----------

    def _draw_box(
        self, canvas: np.ndarray, frame_bgr: np.ndarray, hands: list[HandPose], seed: int
    ) -> None:
        left, right = _left_right(hands)

        # 端棱 = 食指尖→小指尖(C 形捧握), 盒长 = 两手食指尖距(实测)
        iL = left.points[INDEX_TIP, :2].astype(np.float32)
        pL = left.points[PINKY_TIP, :2].astype(np.float32)
        iR = right.points[INDEX_TIP, :2].astype(np.float32)
        pR = right.points[PINKY_TIP, :2].astype(np.float32)
        span = float(np.linalg.norm(iR - iL))
        if span < MIN_SPAN_PX:
            return

        def _up(i: np.ndarray, p: np.ndarray) -> np.ndarray:
            """顶面挤出方向: 端棱垂线朝上的一支, 再混入固定俯视分量(相机低头感)."""
            e = p - i
            n = float(np.linalg.norm(e))
            if n < 6.0:
                return np.array([0.0, -1.0], np.float32)
            e = e / n
            up = np.array([e[1], -e[0]], np.float32)
            if up[1] > 0:
                up = -up
            up = up + np.array([0.0, -BOX_UP_BIAS], np.float32)
            return up / float(np.linalg.norm(up))

        depth = BOX_DEPTH * span
        lb = iL + _up(iL, pL) * depth
        rb = iR + _up(iR, pR) * depth

        front = np.array([iL, iR, pR, pL], np.float32)
        top = np.array([lb, rb, iR, iL], np.float32)

        # 直纹面: 面沿对角线劈成两个三角形, 翻到背面的三角形渲染红色背面;
        # 未折叠(同号)时整面一次填充
        ref = _tri_sign(front[:3])
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for quad, lut, shift in (
            (front, GREEN_LUT, (0.0, 0.0)),
            (top, BLUE_LUT, (0.0, BOX_SAMPLE_K * span)),
        ):
            t1 = quad[:3]
            t2 = np.array([quad[2], quad[3], quad[0]], np.float32)
            s1, s2 = _tri_sign(t1), _tri_sign(t2)
            if s1 == s2:
                _cmap_fill(canvas, frame_bgr, quad, lut if s1 == ref else RED_LUT, shift)
            else:
                _cmap_fill(canvas, frame_bgr, t1, lut if s1 == ref else RED_LUT, shift)
                _cmap_fill(canvas, frame_bgr, t2, lut if s2 == ref else RED_LUT, shift)
                folds.append((quad[0], quad[2]))  # 四角不共面 → 对角折痕描边

        self._glitch(canvas, frame_bgr, top, seed)

        for poly in (top, front):
            cv2.polylines(canvas, [np.round(poly).astype(np.int32)], True, EDGE, 3, cv2.LINE_AA)
        for a, b in folds:
            _edge_line(canvas, a, b)

    def _glitch(self, canvas: np.ndarray, frame_bgr: np.ndarray, top: np.ndarray, seed: int) -> None:
        """蓝顶面横条故障: 水平位移的原色背景条(不染蓝)."""
        got = _poly_window(canvas, frame_bgr, top)
        if got is None:
            return
        roi, _, mask, (x0, y0) = got
        bh, bw = mask.shape
        if bw < 60 or bh < 20:
            return
        h, w = canvas.shape[:2]
        rng = random.Random(seed // GLITCH_HOLD)
        for _ in range(GLITCH_STRIPS):
            sh = min(rng.randint(*GLITCH_H), bh - 1)
            sw = min(rng.randint(*GLITCH_W), bw - 1)
            ly = rng.randint(0, bh - sh - 1)
            lx = rng.randint(0, bw - sw - 1)
            gx = int(np.clip(x0 + lx + rng.randint(-GLITCH_SHIFT, GLITCH_SHIFT), 0, w - sw))
            strip = frame_bgr[y0 + ly : y0 + ly + sh, gx : gx + sw]
            cv2.copyTo(strip, mask[ly : ly + sh, lx : lx + sw], roi[ly : ly + sh, lx : lx + sw])

    # ---------- banner: v2 TouchDesigner 四层横幅 ----------

    def _draw_banner(self, canvas: np.ndarray, frame_bgr: np.ndarray, hands: list[HandPose]) -> None:
        left, right = _left_right(hands)
        iL, tL, cL, _ = _pinch(left)
        iR, tR, cR, _ = _pinch(right)
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
        _cmap_fill(canvas, frame_bgr, mid, XRAY_CMAP)
        _cmap_fill(canvas, frame_bgr, band(0.0, 0.20), YELLOW_CMAP)
        _cmap_fill(canvas, frame_bgr, band(0.94, 1.02), WHITE_CMAP)
        _cmap_fill(canvas, frame_bgr, band(1.02, 1.28), RED_CMAP)
        # 中窗左右侧缘的黄色细线(原效果唯一的"描边")
        _edge_line(canvas, mid[0], mid[3], BANNER_YELLOW, 2)
        _edge_line(canvas, mid[1], mid[2], BANNER_YELLOW, 2)
