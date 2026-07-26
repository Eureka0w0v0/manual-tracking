"""Hand overlay renderer — 参数全部来自对原视频的逐像素逆向测量.

mirror — 折纸镜面(抖音 manualtracking / AM, v1 前半):
  四角钉在双手拇指尖+食指尖; 平摊=单面, 顶边×底边交叉=麻花双翼(右翼在前),
  双手捏死=双瓣细白线。面 = 反相镜面 clamp(283 - 0.56*bg), 折起的面采样点
  外移并乘冷灰。描边只描自由边, 带 ±2px 色差晕。

screen — 彩色玻璃盒(v1 后半, 参数化刚体长方体):
  手不再直接钉顶点, 而是给 6 个低噪参数; 长方体在 3D 里造好再投影,
  刚性和透视是构造出来的, 不靠事后正则化。原片帧 312 逐点配准实测:
    食指/中指/小指三个指尖共同压在同一条"前棱"上(指弧贴盒子前缘),
    只有拇指在后侧; 后上角没有任何手指对应, 是几何推出来的第四点。
  参数: 前面锚点=每手(食指尖+小指尖)/2 → 长轴与盒长; 指弧展开量 → 盒高;
    orient(掌面前缩) → 绕长轴 roll。进深取盒高的固定比例。
  相机俯角与用户 roll 同轴(都绕长轴), 合成单一角 ψ, 无翻面状态机。
  弱透视 f/(f−toward) 给出真实两点透视(后棱自动短于前棱)。
  面可见性 = 3D 外法线朝向相机(物理正确)。
  六个面六种像素处理(见 effects.BOX_FACES): 顶=蓝反相+横条 glitch(原片实测) /
  前=riso 套色版画 / 背=硬阈值双色 / 底=点云全息 / 左端=四叉树马赛克 /
  右端=半调网点。中间四个复刻自 douyin 那段 TouchDesigner 屏录。
  只描可见面的棱(背面的棱被实体挡住)。
  五指收拢 → 整个盒子收起(什么都不画); 双手合拢 → 白色种子点。

banner — TouchDesigner 横幅(v2):
  四角 = 双手食指尖(上边)+拇指尖(下边)。四层条带: 黄阈值头带 /
  X-ray 中窗(solarize, 左右内缩+黄侧线) / 白软阈值分隔线 / 悬出的红脚带,
  内容全部为摄像头画面的屏幕空间 1:1 双色调变换, 无整板描边。

wire — 纯骨架调试。
"""

from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np

from .glassbox import GlassBox
from .handgeom import ORIENT_GAIN, grip, orient, palm_center, pinch
from .effects import (
    BOX_FACES,
    GLITCH_H,
    GLITCH_HOLD,
    GLITCH_SHIFT,
    GLITCH_STRIPS,
    GLITCH_W,
    Face,
    BANNER_YELLOW,
    RED_CMAP,
    WHITE_CMAP,
    XRAY_CMAP,
    YELLOW_CMAP,
    FaceEffect,
    fx_mirror,
    _fx_lut,
)
from .landmarks import (
    CONNECTIONS,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_TIP,
    PALM_RING,
    PINKY_DIP,
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
ROLE_HYST_PX = 25.0  # 左右角色互换需越过的掌心 x 差(防双手并拢时颜色频闪)
ORIENT_GAIN = 2.2  # 手掌朝向→明暗的灵敏度(越大翻手反应越猛)
BASE_B = 0.85  # 默认亮度(纸平摊时接近亮白)
B_SWING = 0.65  # 翻手带来的亮度摆幅
COOL_START = 0.7  # 亮度低于此值开始变冷
COOL_RATE = 1.8  # 变冷速度
MIRROR_SHIFT = 0.35  # 折起的面采样点外移量(跨距比例)
BOX_SAMPLE_K = 0.20  # 顶面采样点下移量(盒长比例, 实测 250-300px@1080)


# ---- 手部几何 ----


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


def _fill(
    canvas: np.ndarray,
    frame_bgr: np.ndarray,
    poly: np.ndarray,
    fx: FaceEffect,
    shift: tuple[float, float] = (0.0, 0.0),
) -> None:
    """把 poly 围出的区域填成 fx(背景窗口(uv+shift)) —— 三种风格唯一的上色出口.

    早先有 _cmap_fill / _fx_fill / _mirror_fill 三个函数, 骨架逐字相同、只有
    "怎么把 src 变成颜色"那一行不同。现在那一行外提成 FaceEffect, 三合一。
    """
    got = _poly_window(canvas, frame_bgr, poly, shift)
    if got is None:
        return
    roi, src, mask, _ = got
    cv2.copyTo(np.ascontiguousarray(fx(src)), mask, roi)


class VectorOverlayRenderer:
    """按 style(见模块 docstring)把手部特效画到帧上.

    跨帧状态只剩两小份: 盒高/ψ 的标量 EMA、左右角色滞回。
    (角点级滤波、n̂ 符号滞回、每手卷向 EMA 随刚体参数化一并删除。)
    """

    def __init__(
        self,
        style: str = "mirror",
        *,
        show_source: bool = True,
        source_dim: float = 0.65,
    ) -> None:
        self._style = canon_style(style)
        self.show_source = show_source
        self.source_dim = float(source_dim)
        self.box = GlassBox()  # screen 的几何求解器(自带跨帧状态与 5 个实时旋钮)
        self._role_ids: tuple[int, int] | None = None  # (左手 sid, 右手 sid)

    @property
    def style(self) -> str:
        return self._style

    @style.setter
    def style(self, name: str) -> None:
        """切风格顺带清状态: 换走再换回来时不该拿几秒前的 ψ 继续插值."""
        new = canon_style(name)
        if new != self._style:
            self._style = new
            self._reset_box()  # 角色滞回与风格无关, 不该跟着清

    @property
    def box_debug(self) -> str:
        """HUD 字符串(screen 专用). 转发给 GlassBox, live 不必知道有这一层."""
        return self.box.debug

    def _reset_box(self) -> None:
        """只清 screen 的盒子状态. 双手合拢/切风格时用。

        **不碰 _role_ids** —— 那是三种风格共用的左右手角色滞回, 由 _ordered
        维护。早先版本一并清掉, 于是双手合拢的每一帧都在重置角色滞回, 而
        ROLE_HYST_PX 的存在理由恰恰就是"防双手并拢时角色逐帧翻转"——护栏
        在最该生效的场景里被自己关掉了。
        """
        self.box.reset()

    def _reset_state(self) -> None:
        """清掉全部跨帧状态(手离场). 下一帧当作冷启动."""
        self._reset_box()
        self.box._shut = False  # 手都离场了, 收拢滞回也该归零, 回来时重新判
        self._role_ids = None

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
        else:
            # 手不足两只 → 清跨帧状态。_box_geometry 里那份只覆盖"双手合拢"
            # (它要 len(hands)>=2 才被调到), 手移出画面走的是这条路: 实测离场
            # 3 秒回来时 _box_ema/_psi_rate 原封不动, 盒子要边转边追 18 帧(0.6s)。
            self._reset_state()

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

    def _ordered(self, hands: list[HandPose]) -> tuple[HandPose, HandPose]:
        """画面左手/右手, 带 25px 滞回——双手并拢时角色不逐帧翻转.

        角色翻转会让面片绕序反号(蓝绿↔红 LUT 频闪)、麻花判定抖动;
        实测无滞回时并拢悬停每两帧换一次位。
        """
        a, b = hands[0], hands[1]
        xa, xb = float(palm_center(a)[0]), float(palm_center(b)[0])
        ids = (a.track_id, b.track_id)
        if (
            self._role_ids is not None
            and min(ids) >= 0
            and set(ids) == set(self._role_ids)
        ):
            lid = self._role_ids[0]
            left, right = (a, b) if a.track_id == lid else (b, a)
            lx, rx = (xa, xb) if a.track_id == lid else (xb, xa)
            if lx > rx + ROLE_HYST_PX:  # 明确越过才交换角色
                left, right = right, left
        else:
            left, right = (a, b) if xa <= xb else (b, a)
        self._role_ids = (left.track_id, right.track_id)
        return left, right

    # ---------- mirror: 折纸镜面 ----------

    def _draw_sheet(self, canvas: np.ndarray, frame_bgr: np.ndarray, hands: list[HandPose]) -> None:
        left, right = self._ordered(hands)
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
                _edge_line(canvas, cL + perp * s, cR + perp * s)
            return

        u = span_v / span

        # 每半张纸的明暗由那只手的手掌朝向决定: 默认亮白, 翻手变暗
        bL = float(np.clip(BASE_B + B_SWING * orient(left), 0.2, 1.0))
        bR = float(np.clip(BASE_B + B_SWING * orient(right), 0.2, 1.0))

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
            _fill(canvas, frame_bgr, poly, fx_mirror(cool), (float(shift_v[0]), float(shift_v[1])))
            pr = np.round(poly).astype(np.int32)
            # 描边只描自由边(整面轮廓), 附 ±2px 色差晕
            cv2.polylines(canvas, [pr + (2, 1)], True, FRINGE_WARM, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [pr - (2, 1)], True, FRINGE_COOL, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [pr], True, EDGE, 3, cv2.LINE_AA)

    # ---------- screen: 彩色玻璃盒 ----------

    def _draw_box(
        self, canvas: np.ndarray, frame_bgr: np.ndarray, hands: list[HandPose], seed: int
    ) -> None:
        left, right = self._ordered(hands)
        if self.box.is_shut(left, right):
            # 五指收拢 → 整个盒子收起, 什么都不画(连种子点也不画: 那是"双手合拢"
            # 的语汇, 表示盒子塌成一点; 收拢是"收工", 该干净地消失)
            self.box.reset()
            return
        geo = self.box.solve(left, right)
        if geo is None:
            # 双手合拢 → 白色种子点(盒子出现/收起的中间态)
            c = (self.box.anchor(left) + self.box.anchor(right)) * 0.5
            ci = (int(round(float(c[0]))), int(round(float(c[1]))))
            cv2.circle(canvas, ci, 5, EDGE, -1, cv2.LINE_AA)
            cv2.circle(canvas, ci, 2, WHITE_HOT, -1, cv2.LINE_AA)
            return
        scr, cam, focal, length = geo

        # 面可见性 = 外法线朝向相机(凸体, 物理正确; 不再用 2D 绕序启发式)。
        # 外法线用"面心 − 体心"定向, 免去手工排 CCW 的符号坑。
        eye = np.array([0.0, 0.0, focal], np.float32)
        center = cam.mean(axis=0)
        vis: list[tuple[float, int, Face]] = []
        for fi, face in enumerate(BOX_FACES):
            q = cam[list(face.verts)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            face_c = q.mean(axis=0)
            if float(n @ (face_c - center)) < 0.0:
                n = -n
            if float(n @ (eye - face_c)) > 0.0:
                vis.append((float(face_c[2]), fi, face))
        # 凸体 + 背面剔除 ⇒ 可见面在投影上恰好铺满剪影一次, 互不重叠(实测原片
        # 187 帧两两交集面积恒为 0)。所以这里排序纯粹是让绘制顺序确定, 与遮挡
        # 无关——去掉它只会让共享棱的抗锯齿舍入差 1 个灰阶。
        vis.sort(key=lambda v: v[0])
        # HUD 用: 当前哪些面朝着镜头。调 roll 时靠它区分"几何没转到"和
        # "画了但读不出来"(红背 LUT 在暗底上是全场最暗的一块, 很容易漏看)
        self.box.debug += "  " + ("+".join(v[2].tag for v in vis) or "-")

        if not vis:  # 盒高恰好为 0: 所有面零面积, 兜底描一条侧视细线
            _edge_line(canvas, scr[0], scr[4])
            return

        for _, _fi, face in vis:
            quad = scr[list(face.verts)]
            shift = (0.0, BOX_SAMPLE_K * length) if face.is_top else (0.0, 0.0)
            _fill(canvas, frame_bgr, quad, face.fx, shift)
            if face.is_top:
                self._glitch(canvas, frame_bgr, quad, seed)

        # 只描可见面的棱, 每条棱画一次(背面的棱被实体挡住, 不该露)
        drawn: set[tuple[int, int]] = set()
        for _, _, face in vis:
            idx = face.verts
            for a, b in zip(idx, idx[1:] + idx[:1]):
                e = (a, b) if a < b else (b, a)
                if e not in drawn:
                    drawn.add(e)
                    _edge_line(canvas, scr[a], scr[b])

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
        left, right = self._ordered(hands)
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
        _fill(canvas, frame_bgr, mid, _fx_lut(XRAY_CMAP))
        _fill(canvas, frame_bgr, band(0.0, 0.20), _fx_lut(YELLOW_CMAP))
        _fill(canvas, frame_bgr, band(0.94, 1.02), _fx_lut(WHITE_CMAP))
        _fill(canvas, frame_bgr, band(1.02, 1.28), _fx_lut(RED_CMAP))
        # 中窗左右侧缘的黄色细线(原效果唯一的"描边")
        _edge_line(canvas, mid[0], mid[3], BANNER_YELLOW, 2)
        _edge_line(canvas, mid[1], mid[2], BANNER_YELLOW, 2)
