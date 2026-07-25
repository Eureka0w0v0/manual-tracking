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
    _orient(掌面前缩) → 绕长轴 roll。进深取盒高的固定比例。
  相机俯角与用户 roll 同轴(都绕长轴), 合成单一角 ψ, 无翻面状态机。
  弱透视 f/(f−toward) 给出真实两点透视(后棱自动短于前棱)。
  面可见性 = 3D 外法线朝向相机(物理正确)。
  每面不同特效: 顶面蓝反相 LUT+横条 glitch / 前面·端面绿 LUT /
  底面 X-ray / 背面红 LUT。只描可见面的棱(背面的棱被实体挡住)。
  五指收拢 → 盒高→0 塌成扁带; 双手合拢 → 白色种子点。

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
ROLE_HYST_PX = 25.0  # 左右角色互换需越过的掌心 x 差(防双手并拢时颜色频闪)
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
BOX_SAMPLE_K = 0.20  # 顶面采样点下移量(盒长比例, 实测 250-300px@1080)
# 长方体参数化: 下面四个常数由原片帧 312 的四个角点 + 后/前棱比联立数值反解,
# 四角残差 RMS 0.3px(盒高 231px), 透视比 0.808 vs 实测 0.81。改动请重跑标定。
BOX_H_GAIN = 1.14  # 盒真高 / 指弧展开量(食指尖→小指尖); 反解 313/274
BOX_DEPTH_RATIO = 1.47  # 进深 / 盒真高; 反解 461/313
BOX_ROLL_BIAS = 0.830  # 静止时的截面转角 rad(=相机俯角, 47.6°); 保证看得见蓝顶面
# 掌面朝向 o → 绕长轴 roll。**不是线性 gain**, 而是两端收敛到 ±180°:
#     ψ = BIAS + (π·sign(o) − BIAS) · |o|^EXPO
# 为什么不用线性 gain: 背红面的投影面积对 ψ 是单峰的(峰在 180°, 那里红面
# 满屏), 线性映射只能在直线上挑落点, 顾此失彼——实测 gain 2.0 正端 ψ=162°
# 红面积 89.6%, 提到 3.0 后正端跑到 219° 越过峰值, 红面积反而掉到 56.4%,
# 同时可见面切换从 3.37 次/秒涨到 5.29 次/秒(颜色频闪)。
# expo 把两端直接钉在峰值 ±180° 上, 且 o=0 时恰好回到 BIAS(静止外观逐像素
# 不变)。dψ/do = (π − sign(o)·BIAS)·EXPO·|o|^(EXPO−1) > 0 恒成立 → 严格单调。
# EXPO 越大中心越钝、静止越稳(1.0 线性 / 1.5 平衡 / 2.0 最稳)。
BOX_ROLL_EXPO = 1.5
BOX_AXIS_Z_GAIN = 1.2  # 掌宽比 → 长轴深度分量; 一只手往前伸盒子就指向镜头(0 = 长轴锁在像平面)
BOX_ANCHOR_LIFT = 0.85  # 锚点在 掌心(0)↔指弧中点(1) 之间的位置; 帧 312 反解最优 0.95
BOX_DEPTH_BIAS = 0.5  # 锚点在进深方向的位置, 同时也是绕长轴旋转的不动点:
#   0   = 前面压在手上(原片位置), 但翻转时是"前棱当轴甩", 盒子整体堆在手后面
#   0.5 = 体心落在手上 → 盒子跟着手整体转(手感对), 深度上以手为中心
#   1   = 后面压在手上
BOX_FOCAL = 1.10  # 弱透视焦距 = 该值 × max(盒长, 高+深)
BOX_SMOOTH = 0.6  # 盒高/长轴深度的轻 EMA(慢变量, 固定系数够用)
# ψ 的速度自适应 EMA(与 tracker.py 的 landmark 级 One Euro 同源思路, 但这里
# 按"帧"而非墙钟计时, 保证离线渲染可复现)。固定 EMA 实测在 0.4s 快速翻手中
# 恒定落后 31°、手停后还要 5 帧追上——那就是"太滑不跟手"的来源。
BOX_ROLL_RESP = 0.40  # 静止时每帧吸收的新值比例(= 旧固定 EMA 的 1−0.6, 抖动不劣化)
BOX_ROLL_RESP_GAIN = 1.35  # 每 (rad/帧) 角速度把上面这个值抬高多少
BOX_ROLL_RESP_MAX = 0.92  # 上限, 留一点滤波防单帧误检直接甩过去
BOX_RATE_SMOOTH = 0.5  # 角速度估计自身的 EMA(不平滑的话增益会跟着抖)
# ψ 的角速度硬限幅(度/检测帧)。_orient 在饱和区会被噪声掀翻符号——实测原片
# 里出现过"一只手 o 从 −1.00 一帧跳到 +0.74 而另一只手没动", 两手平均后 ψ
# 单帧弹 199.5°(p99 117.6°, >90° 的帧占 1.6%)。手物理上不可能 33ms 转 180°,
# 所以超过这个速率的一律是误检。20°/帧 @30fps = 600°/s, 比最快的翻腕还快
# 一倍有余, 不会削掉真实动作。限幅作用在滤波之后, 直接约束"看到的"角速度。
BOX_ROLL_MAX_RATE = 20.0
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


def _grip(hand: HandPose) -> tuple[np.ndarray, np.ndarray, float, float, float]:
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
        _palm_center(hand),
        (i + p) * 0.5,
        float(np.linalg.norm(i - p)),
        palm,
        _orient(hand),
    )


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
# 背面红。原片逐像素实测值是一条很窄的暗红带(灰60→亮度52.7, 灰160→82.0,
# 全域跨度仅 60), 实测**比 source_dim=0.65 压过的背景还暗**(灰160 处背景 104
# vs 红面 82) —— 翻过去了也读不出来, 用户反馈的"看不到后面"有一半是这个。
# 下面这组保持红相(红度 @灰110 从 54 提到 169)但把动态范围拉到 174, 每一档
# 都亮过背景。原片测量值保留在上面的注释里, 要还原保真度就换回去。
RED_LUT = _build_lut(
    [
        (16, (24, 20, 58)),
        (48, (32, 26, 112)),
        (80, (38, 30, 168)),
        (112, (46, 36, 212)),
        (144, (58, 46, 238)),
        (176, (84, 68, 250)),
        (216, (134, 116, 254)),
        (255, (190, 180, 255)),
    ]
)
YELLOW_CMAP = _duotone_cmap(BANNER_Y_DARK, BANNER_YELLOW, BANNER_THRESH)
WHITE_CMAP = _duotone_cmap(BANNER_W_DARK, BANNER_WHITE, BANNER_THRESH, soft=45)
RED_CMAP = _duotone_cmap(BANNER_R_DARK, BANNER_RED, BANNER_THRESH)
XRAY_CMAP = _xray_cmap()

# 长方体拓扑: 顶点索引 = x*4 + u*2 + w
#   x: 0=左端 1=右端 / u: 0=下 1=上 / w: 0=前(贴指弧) 1=后(远离镜头)
# 绕序无所谓——外法线在运行时用"面心 − 体心"定向, 不靠手工排 CCW。
_BOX_FACES: tuple[tuple[tuple[int, int, int, int], np.ndarray], ...] = (
    ((0, 4, 6, 2), GREEN_LUT),  # 前面
    ((1, 5, 7, 3), RED_LUT),  # 背面(翻过去才露)
    ((0, 4, 5, 1), XRAY_CMAP),  # 底面(翻上来才露)
    ((2, 6, 7, 3), BLUE_LUT),  # 顶面(带横条 glitch)
    ((0, 2, 3, 1), GREEN_LUT),  # 左端面
    ((4, 6, 7, 5), GREEN_LUT),  # 右端面
)
_BOX_TOP_FACE = 3  # 顶面在 _BOX_FACES 里的下标(采样偏移 + glitch 只给它)
_BOX_FACE_TAGS = ("前", "背", "底", "顶", "左", "右")  # HUD 显示当前可见面用


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
        self._box_ema: tuple[float, float, float] | None = None  # (盒高, ψ, 长轴深度分量)
        self._role_ids: tuple[int, int] | None = None  # (左手 sid, 右手 sid)
        self.roll_expo = BOX_ROLL_EXPO  # 实时可调(live 的 [ ] 键): 翻转曲线陡度
        self.anchor_lift = BOX_ANCHOR_LIFT  # 实时可调(live 的 ; ' 键): 盒子挂多高
        self.depth_bias = BOX_DEPTH_BIAS  # 实时可调(live 的 , . 键): 旋转不动点/进深中心
        self.roll_resp = BOX_ROLL_RESP  # 实时可调(live 的 9 0 键): 旋转跟手程度
        self.roll_max_rate = BOX_ROLL_MAX_RATE  # 实时可调(live 的 7 8 键): 角速度上限
        self._psi_rate = 0.0  # ψ 的角速度估计(rad/帧), 驱动自适应滤波
        self._b0_prev: np.ndarray | None = None  # 上帧的截面"上"轴(符号帧间传播)
        self.box_debug = ""  # live HUD 用: 当前 ψ / 双手掌朝向 / 长轴深度

    @property
    def style(self) -> str:
        return self._style

    @style.setter
    def style(self, name: str) -> None:
        """切风格顺带清状态: 换走再换回来时不该拿几秒前的 ψ 继续插值."""
        new = canon_style(name)
        if new != self._style:
            self._style = new
            self._reset_state()

    def _reset_state(self) -> None:
        """清掉全部跨帧状态(手离场/合拢/切风格). 下一帧当作冷启动."""
        self._box_ema = None
        self._psi_rate = 0.0
        self._b0_prev = None
        self._role_ids = None
        self.box_debug = ""

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
        xa, xb = float(_palm_center(a)[0]), float(_palm_center(b)[0])
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

    def _box_geometry(
        self, left: HandPose, right: HandPose
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """双手参数 → 刚体长方体, 返回 (屏幕 8 顶点, 相机系 8 顶点, 焦距, 盒长).

        顶点索引 = x*4 + u*2 + w, 与 _BOX_FACES 一致。

        长轴是**真 3D 向量**: 屏幕位移来自两掌心连线, 深度分量来自两手掌宽比
        (投影尺寸 ∝ 1/距离, 一只手往前伸盒子就指向镜头)。所以盒子能朝任意
        方向, 端面也能露出来——长轴锁在像平面时端面结构上永不可见。
        截面两轴 (b̂ 屏幕上 / ĉ 朝观察者) 由长轴正交化得到, 再绕长轴转 ψ:
            b̂ = b̂₀cosψ + ĉ₀sinψ      ĉ = −b̂₀sinψ + ĉ₀cosψ
        ψ = 相机俯角 + 用户 roll——两者同轴, 合成一个角, 不需要翻面状态机。
        顶点 = t·长轴 + u·b̂ + w·ĉ, 再按 f/(f−z) 弱透视投影。

        w 的原点(= 锚点 = 手)就是旋转的不动点, 位置由 depth_bias 决定:
        默认 0.5 把体心放在手上, 盒子跟着手整体转; 取 0 会把前面钉在手上,
        翻转就变成"拿前棱当轴甩"、盒子整体堆在手后面(实测手感不对)。

        刚性是构造出来的: 8 个顶点由 4 个标量(长/高/深/ψ)+一个 3D 轴生成,
        恒为长方体。旧版 8 个角各自独立跟指尖, 实测"本应等长"的两条深度棱
        中位差 1.56×(p90 2.94×), 楔形在那个架构里无解。
        """
        cL, fL, sL, pL, oL = _grip(left)
        cR, fR, sR, pR, oR = _grip(right)
        t = self.anchor_lift  # 0=掌心(偏低) 1=指弧中点(原片高度)
        gL, gR = cL + (fL - cL) * t, cR + (fR - cR) * t
        span_v = gR - gL
        span = float(np.linalg.norm(span_v))
        if span < MIN_SPAN_PX:
            # 双手合拢: 清掉滤波历史。否则再张开时 ψ 从旧值继续插值, 且
            # _psi_rate 还留着上次的大角速度 → 恢复的头几帧 k 贴在
            # BOX_ROLL_RESP_MAX 上, 滤波形同虚设。
            self._reset_state()
            return None

        # 长轴深度分量: 掌宽比 → 单目深度线索(哪只手更近就更大)。
        # 存 EMA 的是**比值**不是像素: 比值是无量纲的, 而像素随 span 变——快速
        # 收手时旧的大 dz 会被带进来, 实测 span 900→60 时 dz/span 从 1.08 冲到
        # 4.03(长轴偏出像平面 76°), 突破了 BOX_AXIS_Z_GAIN 这个上界。
        dzr_raw = BOX_AXIS_Z_GAIN * (pR - pL) / max(pR + pL, 1e-3)
        height_raw = BOX_H_GAIN * (sL + sR) * 0.5
        o = (oL + oR) * 0.5
        psi_raw = BOX_ROLL_BIAS + (np.pi * np.sign(o) - BOX_ROLL_BIAS) * abs(o) ** self.roll_expo
        if self._box_ema is None:
            height, psi, dzr = height_raw, psi_raw, dzr_raw
        else:
            a = BOX_SMOOTH
            height = self._box_ema[0] * a + height_raw * (1.0 - a)
            dzr = self._box_ema[2] * a + dzr_raw * (1.0 - a)
            # ψ: 转得越快滤波越松 → 静止不抖, 快速翻手仍然跟手
            prev = self._box_ema[1]
            # 走最短弧: ψ 是角度, ±π 是同一姿态。线性插值会绕远路——实测原片
            # 186 个帧里有 3 帧 |ψ_raw − prev| > 180°(最大 216.9°, 最短弧只要
            # 143.1°), 叠加限幅后要多花 ~4 帧才转到位。
            err = float(np.arctan2(np.sin(psi_raw - prev), np.cos(psi_raw - prev)))
            self._psi_rate = self._psi_rate * BOX_RATE_SMOOTH + abs(err) * (1.0 - BOX_RATE_SMOOTH)
            k = min(self.roll_resp + BOX_ROLL_RESP_GAIN * self._psi_rate, BOX_ROLL_RESP_MAX)
            cap = np.radians(self.roll_max_rate)  # 硬限幅: 挡掉 _orient 掀翻符号造成的弹飞
            psi = prev + float(np.clip(err * k, -cap, cap))
        self._box_ema = (height, psi, dzr)
        dz = dzr * span  # 比值 → 当帧像素
        self.box_debug = f"psi{np.degrees(psi):+5.0f} oL{oL:+.2f} oR{oR:+.2f} dz{dz:+4.0f}"

        depth = BOX_DEPTH_RATIO * height
        axis = np.array([span_v[0], span_v[1], dz], np.float32)  # (屏幕x, 屏幕y, 朝观察者)
        length = float(np.linalg.norm(axis))
        axis_hat = axis / length
        # 截面正交基: ĉ₀ 朝观察者(+z), b̂₀ 屏幕"上"。
        # ĉ₀ 必须**恒定朝向观察者**——早先版本用 −axiŝ×(0,−1,0), 其 z 分量正比于
        # span_v.x, 于是双手 x 差过零时整组基翻号、盒子瞬间里外翻(可见面变成互补
        # 集)。而 ROLE_HYST_PX=25 的角色滞回恰好让 dx∈(−25,0) 成为可达状态, 不是
        # 理论边角。改为把 +ẑ 对长轴做 Gram-Schmidt 正交化: z 分量恒 ≥0, 无符号翻转。
        c0 = np.array([0.0, 0.0, 1.0], np.float32) - axis_hat * float(axis_hat[2])
        n0 = float(np.linalg.norm(c0))
        if n0 < 1e-3:  # 长轴几乎正对镜头, +ẑ 退化 → 换屏幕"上"当参考
            c0 = np.array([0.0, -1.0, 0.0], np.float32)
            c0 = c0 - axis_hat * float(axis_hat @ c0)
            n0 = max(float(np.linalg.norm(c0)), 1e-6)
        c0 = c0 / n0
        # b̂₀ = ±(ĉ₀ × âxis), 符号得挑一个。按"屏幕上"挑(b0[1]<0)会在 b0[1]=0 处
        # 180° 翻转 —— 解析上该条件等价于 span_v.x=0(长轴竖直), 而 ROLE_HYST_PX
        # 守的是**掌心** x 差, 与 lift 后的锚点 x 差实测相隔 559px(p95), 根本护不住:
        # 锚点 dx 过零实测单帧 608px 跳变、3.8% 画面像素变化。
        # 改为帧间传播(与上帧同向者胜): 长轴扫过竖直时连续穿过, 不存在翻转点。
        b0 = np.cross(c0, axis_hat)
        if self._b0_prev is not None:
            if float(b0 @ self._b0_prev) < 0.0:
                b0 = -b0
        elif b0[1] > 0:  # 冷启动才用"屏幕上"定初值
            b0 = -b0
        self._b0_prev = b0.copy()
        cos_p, sin_p = float(np.cos(psi)), float(np.sin(psi))
        b_hat = b0 * cos_p + c0 * sin_p
        c_hat = -b0 * sin_p + c0 * cos_p

        # f 至少 1.1×(高+深) → f−z 恒为正, 双手贴近时不会被透视除爆
        focal = BOX_FOCAL * max(length, height + depth)
        anchor = (gL + gR) * 0.5

        cam = np.empty((8, 3), np.float32)
        scr = np.empty((8, 2), np.float32)
        # w 的原点 = 锚点 = 绕长轴旋转的不动点。depth_bias=0.5 时体心落在手上,
        # 盒子跟着手整体转; =0 时前面压在手上, 翻转会变成"前棱当轴甩"。
        w_front = depth * self.depth_bias
        for xi, t in enumerate((-0.5, 0.5)):
            for ui, u in enumerate((-0.5 * height, 0.5 * height)):
                for wi, w in enumerate((w_front, w_front - depth)):
                    p3 = axis * t + b_hat * u + c_hat * w
                    k = xi * 4 + ui * 2 + wi
                    # 相机系右手基: x 右, y 上(屏幕 y 取负), z 朝观察者
                    cam[k] = (p3[0], -p3[1], p3[2])
                    scr[k] = anchor + p3[:2] * (focal / (focal - p3[2]))
        return scr, cam, focal, length

    def _draw_box(
        self, canvas: np.ndarray, frame_bgr: np.ndarray, hands: list[HandPose], seed: int
    ) -> None:
        left, right = self._ordered(hands)
        geo = self._box_geometry(left, right)
        if geo is None:
            # 双手合拢 → 白色种子点(盒子出现/收起的中间态)
            t = self.anchor_lift
            c = sum(g[0] + (g[1] - g[0]) * t for g in (_grip(left), _grip(right))) * 0.5
            ci = (int(round(float(c[0]))), int(round(float(c[1]))))
            cv2.circle(canvas, ci, 5, EDGE, -1, cv2.LINE_AA)
            cv2.circle(canvas, ci, 2, WHITE_HOT, -1, cv2.LINE_AA)
            return
        scr, cam, focal, length = geo

        # 面可见性 = 外法线朝向相机(凸体, 物理正确; 不再用 2D 绕序启发式)。
        # 外法线用"面心 − 体心"定向, 免去手工排 CCW 的符号坑。
        eye = np.array([0.0, 0.0, focal], np.float32)
        center = cam.mean(axis=0)
        vis: list[tuple[float, int, tuple[int, int, int, int], np.ndarray]] = []
        for fi, (idx, lut) in enumerate(_BOX_FACES):
            q = cam[list(idx)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            face_c = q.mean(axis=0)
            if float(n @ (face_c - center)) < 0.0:
                n = -n
            if float(n @ (eye - face_c)) > 0.0:
                vis.append((float(face_c[2]), fi, idx, lut))
        # 凸体 + 背面剔除 ⇒ 可见面在投影上恰好铺满剪影一次, 互不重叠(实测原片
        # 187 帧两两交集面积恒为 0)。所以这里排序纯粹是让绘制顺序确定, 与遮挡
        # 无关——去掉它只会让共享棱的抗锯齿舍入差 1 个灰阶。
        vis.sort(key=lambda v: v[0])
        # HUD 用: 当前哪些面朝着镜头。调 roll 时靠它区分"几何没转到"和
        # "画了但读不出来"(红背 LUT 在暗底上是全场最暗的一块, 很容易漏看)
        self.box_debug += "  " + ("+".join(_BOX_FACE_TAGS[v[1]] for v in vis) or "-")

        if not vis:  # 盒高恰好为 0: 所有面零面积, 兜底描一条侧视细线
            _edge_line(canvas, scr[0], scr[4])
            return

        for _, fi, idx, lut in vis:
            quad = scr[list(idx)]
            shift = (0.0, BOX_SAMPLE_K * length) if fi == _BOX_TOP_FACE else (0.0, 0.0)
            _cmap_fill(canvas, frame_bgr, quad, lut, shift)
            if fi == _BOX_TOP_FACE:
                self._glitch(canvas, frame_bgr, quad, seed)

        # 只描可见面的棱, 每条棱画一次(背面的棱被实体挡住, 不该露)
        drawn: set[tuple[int, int]] = set()
        for _, _, idx, _lut in vis:
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
