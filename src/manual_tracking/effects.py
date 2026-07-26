"""每个面的像素处理(effect)与它们用到的颜色映射.

这一层**完全不知道"手"的存在** —— 输入输出都是 np.ndarray, 无跨帧状态,
所以能单独拿出来看/改/测。renderer 只负责决定"哪个面用哪个 effect、
画在哪块多边形里"。

术语:
  colormap / LUT   256x1x3 的 uint8 查表, 配 cv2.applyColorMap 用: 灰度 → BGR
  FaceEffect       src_bgr → out_bgr(同尺寸)的像素处理函数
  _fx_*            "参数 → FaceEffect" 的工厂; 参数在构造期烘焙进查表/缓存,
                   所以每帧只剩几次 OpenCV 调用(见各函数里的 before→after 实测)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import cv2
import numpy as np

# ---- banner 风格的双色调配色(BGR; 注释是原片取色的 hex) ----
BANNER_YELLOW = (26, 177, 223)  # #DFB11A
BANNER_Y_DARK = (5, 16, 32)  # #201005
BANNER_RED = (8, 23, 213)  # #D51708
BANNER_R_DARK = (0, 8, 58)  # #3A0800
BANNER_WHITE = (218, 230, 236)  # #ECE6DA
BANNER_W_DARK = (16, 26, 42)  # #2A1A10

# ---- 每面特效的参数(六个面各一套, 见 BOX_FACES) ----
POSTER_LEVELS = 6  # 前面色阶断层的级数(2=极简剪影, 6=版画感, ≥24 基本等于连续)
RED_SPLIT_PX = 3  # 背面横向色差: R 右移/B 左移的像素数(原片实测 R 相对 B +3px)
SCAN_PERIOD = 4  # 底面全息扫描线周期(行)
SCAN_DEPTH = 0.35  # 扫描线压暗深度 0-1
EMBOSS_BASE = 210.0  # 端面浮雕的中性底(差分为 0 处的亮度)
EMBOSS_GAIN = 1.1  # 浮雕对角差分增益(越大线条越硬)
EMBOSS_TINT = (1.00, 0.94, 0.88)  # 浮雕的冷白染色 BGR 乘子
HALFTONE_CELL = 6  # 端面网点的格子边长(px); 越小点越密
HALFTONE_PAPER = (228, 240, 244)  # 网点底色(暖白纸) BGR
HALFTONE_INK = (43, 23, 23)  # 网点墨色 BGR
# 蓝顶面横条 glitch(实测 h15-40 w50-400 @1080p, 按 720p 采集缩放到 2/3)
GLITCH_STRIPS = 3
GLITCH_H = (10, 27)
GLITCH_W = (40, 260)
GLITCH_SHIFT = 40
GLITCH_HOLD = 2  # 每 N 帧换一次图案, 逐帧换会闪成噪声
BANNER_THRESH = 115  # banner 双色调亮度阈值

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


# ---- 每面的像素处理(effect): src_bgr → out_bgr, 同尺寸 ----
# 所有 _fx_* 都是"参数 → 处理函数"的工厂, 处理函数满足这个签名:
FaceEffect = Callable[[np.ndarray], np.ndarray]
# 原片实测结论(逐像素, 帧 264/288):
#   · 面内是真正的 gradient map, 不是平涂——绿前面 BGR 三通道都跑满 0~255,
#     亮度 std 34; 蓝顶面 B 148~255 / G 23~155 / R 27~122
#   · 红背面有横向色差: R 相对 B 位移 +3px(r=0.683), 白描边上肉眼可见冷暖边
#   · **没有扫描线**: 去趋势后行方向频谱无主导频率, 2 行调制深度只有 0.07 灰阶
#     (肉眼看到的"横条"是 h.264 压缩块, 不是特效)
# 端面/底面在原片里几乎没露过, 属于自由创作区; 取用户自己那套 hand-frame-glitch
# 的风格语汇(浮雕线稿 / 半调网点 / 全息扫描线)。


def _fx_lut(lut: np.ndarray, split: int = 0) -> FaceEffect:
    """gradient map; split>0 时附加横向色差(R 右移、B 左移)."""

    def fn(src: np.ndarray) -> np.ndarray:
        out = cv2.applyColorMap(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), lut)
        if split:
            out[..., 2] = np.roll(out[..., 2], split, axis=1)
            out[..., 0] = np.roll(out[..., 0], -split, axis=1)
        return out

    return fn


def _fx_poster(lut: np.ndarray, levels: int) -> FaceEffect:
    """gradient map + 色阶断层: 先把灰度量化成 levels 级再查表 → 版画式硬边。

    量化在**查表前**做, 所以断层落在 LUT 的采样点上, 每一级都是 LUT 上的一个
    确定颜色, 不会出现插值出来的中间色。
    """
    step = 256.0 / max(levels, 2)
    q = (np.clip((np.arange(256) / step).astype(np.int32), 0, levels - 1) * step + step * 0.5).astype(
        np.uint8
    )

    def fn(src: np.ndarray) -> np.ndarray:
        g = cv2.LUT(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), q)
        return cv2.applyColorMap(g, lut)

    return fn


def _fx_scan(lut: np.ndarray, period: int, depth: float) -> FaceEffect:
    """gradient map + 全息扫描线: 每 period 行压暗 depth.

    只对 1/period 的行做原地缩放(切片是视图), 不是整幅乘一个列向量——后者
    要分配一个全画幅 float 中间量, 实测慢 4 倍。
    """
    keep = 1.0 - depth

    def fn(src: np.ndarray) -> np.ndarray:
        out = cv2.applyColorMap(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), lut)
        rows = out[::period]
        cv2.convertScaleAbs(rows, dst=rows, alpha=keep)
        return out

    return fn


_EMBOSS_K = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], np.float32)


def _fx_emboss(base: float, gain: float, tint: tuple[float, float, float]) -> FaceEffect:
    """浮雕线稿: 对角差分 + 常数底 → 白色浅浮雕(端面读作"截面").

    差分值域是 [-255,255], 所以整条 base+gain*d 曲线预算成 511 项查表; 染色
    再用一张 256 项 colormap。两次查表代替两个全画幅 float 乘法(实测 13.2→1.3ms)。
    """
    ramp = np.clip(base + np.arange(-255, 256, dtype=np.float32) * gain, 0, 255).astype(np.uint8)
    tint_lut = np.clip(
        np.arange(256, dtype=np.float32)[:, None] * np.array(tint, np.float32), 0, 255
    ).astype(np.uint8)[:, None, :]

    def fn(src: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        d = cv2.filter2D(g, cv2.CV_16S, _EMBOSS_K)
        return cv2.applyColorMap(ramp[d + 255], tint_lut)

    return fn


def _halftone_thresh(cell: int) -> np.ndarray:
    """一个 cell 的阈值瓦片: 到中心距离² 归一化. 越靠边阈值越低 → 点从中心长大."""
    c = (cell - 1) * 0.5
    y, x = np.mgrid[0:cell, 0:cell].astype(np.float32)
    r = np.hypot(y - c, x - c) / max(c, 1e-6)
    return np.clip(1.0 - r * r, 0.0, 1.0)


def _fx_halftone(cell: int, paper: tuple[int, int, int], ink: tuple[int, int, int]) -> FaceEffect:
    """半调网点: 亮度低于"到中心距离"阈值的像素上墨 → 点随暗部长大.

    平铺阈值图**按需增长后长期复用**, 每帧只做一次切片(视图, 免费)+ 一次
    uint8 比较 + 一次调色板索引。原先每帧重新 np.tile 要 16.6ms, 现在 1.5ms。
    """
    tile = (_halftone_thresh(cell) * 255.0).astype(np.uint8)
    duo = np.empty((256, 1, 3), np.uint8)
    duo[:128] = np.array(paper, np.uint8)  # 掩码 0 = 亮 = 纸
    duo[128:] = np.array(ink, np.uint8)  # 掩码 255 = 暗 = 墨
    cache: dict[str, np.ndarray] = {}

    def fn(src: np.ndarray) -> np.ndarray:
        h, w = src.shape[:2]
        big = cache.get("t")
        if big is None or big.shape[0] < h or big.shape[1] < w:
            reps = (max(h, big.shape[0] if big is not None else 0) // cell + 1,
                    max(w, big.shape[1] if big is not None else 0) // cell + 1)
            big = np.tile(tile, reps)
            cache["t"] = big
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # compare → 0/255 掩码, 再走 applyColorMap 的双色表; 全程 OpenCV, 不做
        # numpy 花式索引(那一步实测占 10ms 里的 8ms)
        return cv2.applyColorMap(cv2.compare(g, big[:h, :w], cv2.CMP_LT), duo)

    return fn


# 长方体拓扑: 顶点索引 = x*4 + u*2 + w
#   x: 0=左端 1=右端 / u: 0=下 1=上 / w: 0=前(贴指弧) 1=后(远离镜头)
# 绕序无所谓——外法线在运行时用"面心 − 体心"定向, 不靠手工排 CCW。
# 六个面六种处理, 彼此一眼可分:

class Face(NamedTuple):
    """长方体的一个面. 顶点索引 = x*4 + u*2 + w.

    x: 0=左端 1=右端 / u: 0=下 1=上 / w: 0=前 1=后(远离镜头)
    绕序无所谓——外法线在运行时用"面心 − 体心"定向, 不靠手工排 CCW。

    早先这是三份平行数组(顶点表 / _BOX_TOP_FACE / _BOX_FACE_TAGS)靠同一个
    下标索引, 没有任何机制保证对齐: 中间插一个面, glitch 会跑到别的面上、
    HUD 会报错误的面名, 而且不抛异常。合成一个元组后同步点归零。
    """

    tag: str  # HUD 上显示的一个字
    verts: tuple[int, int, int, int]
    fx: FaceEffect
    is_top: bool = False  # 顶面独享: 采样偏移 + 横条 glitch


# 六个面六种处理, 彼此一眼可分。前三个的配色/映射有原片逐像素实测背书,
# 后三个(底/两端)原片几乎没露过, 取自用户 hand-frame-glitch 项目的风格语汇。
BOX_FACES: tuple[Face, ...] = (
    Face("前", (0, 4, 6, 2), _fx_poster(GREEN_LUT, POSTER_LEVELS)),
    Face("背", (1, 5, 7, 3), _fx_lut(RED_LUT, split=RED_SPLIT_PX)),
    Face("底", (0, 4, 5, 1), _fx_scan(XRAY_CMAP, SCAN_PERIOD, SCAN_DEPTH)),
    Face("顶", (2, 6, 7, 3), _fx_lut(BLUE_LUT), is_top=True),
    Face("左", (0, 2, 3, 1), _fx_emboss(EMBOSS_BASE, EMBOSS_GAIN, EMBOSS_TINT)),
    Face("右", (4, 6, 7, 5), _fx_halftone(HALFTONE_CELL, HALFTONE_PAPER, HALFTONE_INK)),
)


# 反相镜面(逐帧实测: face = clamp(283 - 0.56*bg), 暗面再乘冷灰)
MIRROR_A = 283.0
MIRROR_B = 0.56
COOL_G, COOL_R = 0.86, 0.84


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


def fx_mirror(cool: float) -> FaceEffect:
    """反相镜面: 逐通道 LUT, 平贴时 shift≈0 → 背景以负片鬼影透出."""
    lut = _mirror_lut(cool)

    def fn(src: np.ndarray) -> np.ndarray:
        return cv2.LUT(src, lut)

    return fn
