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
# 硬阈值双色(丝网印): 取自 douyin TouchDesigner 屏录实测
DUOTONE_DARK = (146, 101, 42)  # 深蓝 BGR(实测占 57%)
DUOTONE_LIGHT = (240, 232, 222)  # 白 BGR(实测占 43%)
DUOTONE_THRESH = 128  # 亮度阈值; 实测中间调占比 0% → 硬切
# 四叉树自适应马赛克
QUAD_MIN_CELL = 8  # 最小块边长(px), 再小就切不动了
QUAD_MAX_DEPTH = 6  # 最大递归层数(兜底, 防极端纹理下节点爆炸)
QUAD_VAR = 90.0  # 亮度方差超过它就继续切; 越小切得越碎
QUAD_LINE = (30, 30, 30)  # 块描边色 BGR(原片是近黑细线)
QUAD_WORK_MAX = 360  # 计算分辨率上限(px); 块结构在缩略图上算完放大, 观感不变
# 点云/全息(单目近似; 原片用深度相机, 这里用局部对比度代替深度)
PC_CELL = 3  # 点阵周期(px); 调出亮点密度 14.7%, 对齐原片实测的 13.3%
PC_GAIN = 3.5  # 局部对比度增益(伪深度强度): 越大越"只剩轮廓", 越小越像亮度图
PC_FLOOR = 4  # 抬黑场(灰阶): 压掉平坦区的点, 让主体浮出黑底
PC_GLOW = 3.0  # 辉光半径(px); 0 = 关掉, 点会变成硬像素块
PC_TINT = (255, 214, 120)  # 青蓝 BGR; 实测色相 H≈98 且 p10-p90 仅 93-101
# 点云的计算分辨率上限(px)。点阵本来就是 cell 周期的低频结构, 在缩略图上算完
# 放大, 点会等比变大但密度观感不变 —— 实测 1660x847 的面从 5.57ms 降到 1ms 级。
PC_WORK_MAX = 420
# riso 版画(双色 + 通道错位边条); 参数取自 douyin 屏录实测
RISO_DARK = (20, 142, 18)  # 暗部纯绿 BGR(实测绿区平均值)
RISO_LIGHT = (238, 246, 248)  # 亮部近白 BGR
RISO_THRESH = 118  # 亮度阈值
RISO_SPLIT = 5  # 通道横向错位(px); 实测三通道边缘平均 x 相差十几像素
RISO_DITHER = 90.0  # 有序抖动幅度(灰阶); 制造颗粒边界, 对齐原片绿区 std 23.7
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

# ---- 共享原语(两个以上 effect 用到; 单独一个用的就地写) ----


def _shrink_for_work(src: np.ndarray, work_max: int) -> np.ndarray:
    """把 src 缩到最长边 ≤ work_max; 已经够小就原样返回(不拷贝).

    四叉树和点云的输出都是**低频结构**(块状 / cell 周期的点阵), 在缩略图上
    算完再放大, 观感一致但省一个数量级(1920x1080 满帧 13.5ms → 1.5ms)。

    缩小必须用 INTER_NEAREST: INTER_AREA 会读遍全部像素, 实测 1660x847→420
    要 7.81ms, 而整个特效预算才 2-3ms —— 它会是这条路径上唯一的大头。面积平均
    带来的抗锯齿在块状/点阵输出上根本看不出来, 换 INTER_NEAREST 后同一步只要
    0.05ms(156x)。
    """
    h, w = src.shape[:2]
    if max(h, w) <= work_max:
        return src
    k = work_max / max(h, w)
    return cv2.resize(
        src, (max(int(w * k), 8), max(int(h * k), 8)), interpolation=cv2.INTER_NEAREST
    )


class _Tiled:
    """按需增长的平铺缓存: 一张周期性小图铺满 ≥(h, w) 的画布, 不够大才重铺.

    三个 effect 都要它(riso 的抖动偏置 / halftone 的阈值瓦片 / 点云的点孔掩码):
    面的 bbox 每帧都在变, 但只有**变大**时才需要重铺 —— 变小直接切片(返回视图,
    免费)。原先每帧重新 np.tile, halftone 实测要 16.6ms。

    必须按 max(新, 旧) 增长。只看新尺寸的话, 盒子转动时 bbox 在"高瘦"和"矮胖"
    之间来回, 每一帧都会推翻上一次的缓存, 缓存等于没有。
    """

    __slots__ = ("_tile", "_big")

    def __init__(self, tile: np.ndarray) -> None:
        self._tile = tile
        self._big: np.ndarray | None = None

    def take(self, h: int, w: int) -> np.ndarray:
        big = self._big
        if big is None or big.shape[0] < h or big.shape[1] < w:
            th, tw = self._tile.shape[:2]
            gh = max(h, 0 if big is None else big.shape[0])
            gw = max(w, 0 if big is None else big.shape[1])
            big = np.tile(self._tile, (gh // th + 1, gw // tw + 1))
            self._big = big
        return big[:h, :w]


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


# 顶面蓝的逐像素实测 LUT(反相型: 背景越亮, 面越暗)
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
# 端面/底面在原片里几乎没露过, 属于自由创作区; 取 douyin 屏录 + 用户自己那套
# hand-frame-glitch 的风格语汇(点云全息 / 四叉树马赛克 / 半调网点)。


def _fx_lut(lut: np.ndarray) -> FaceEffect:
    """gradient map: 灰度 → LUT 查表, 一次 OpenCV 调用."""

    def fn(src: np.ndarray) -> np.ndarray:
        return cv2.applyColorMap(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), lut)

    return fn


def _fx_duotone(dark: tuple[int, int, int], light: tuple[int, int, int], thresh: int) -> FaceEffect:
    """硬阈值双色(丝网印/宝丽来): 亮度过线取亮色, 否则暗色, 没有中间调.

    参数取自 douyin 那段 TouchDesigner 屏录的实测: 蓝框区域只有两簇色
    BGR(146,101,42) 深蓝 57% / (240,232,222) 白 43%, 中间调**占比 0%**,
    水平扫描线的跳变宽度 2px —— 是硬阈值, 不是 gradient map。

    实现就是一张两段式 colormap, 与 _fx_lut 同样只有一次 applyColorMap。
    """
    duo = np.empty((256, 1, 3), np.uint8)
    duo[:thresh] = np.array(dark, np.uint8)
    duo[thresh:] = np.array(light, np.uint8)

    def fn(src: np.ndarray) -> np.ndarray:
        return cv2.applyColorMap(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), duo)

    return fn


def _fx_riso(
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
    thresh: int,
    split: int,
    dither: float = 0.0,
) -> FaceEffect:
    """双色版画 + 通道错位边条(riso / 孔版印刷的套色不准感).

    douyin 那段里第四个特效。汐儿一开始以为是**粒子沙堆**(彩色颗粒下落堆积),
    实测推翻了: 底部绿区占比在帧 158/168/178/188 上恒定 26%, **完全没有堆积**;
    看着像沙丘的波浪边其实是人物深色上衣与白墙的交界。所以它不需要粒子系统,
    也没有跨帧状态。

    真正的机制是**每个通道在错开的位置上做阈值**——实测三通道的强边缘平均 x
    分别是 B 91.4 / R 98.2 / G 113.8, 彼此错开十几像素。三通道一致的地方是纯
    深色或纯亮色, 不一致的窄带就出现青/黄/品红, 正是孔版印刷套色不准的样子。

    颜色取自实测: 暗部 BGR(20,142,18) 纯绿, 亮部近白。
    """
    offs = (-split, 0, split)  # B 左移 / G 不动 / R 右移
    # 有序抖动矩阵(Bayer 4x4): 阈值随位置微抖 → 边界不是光滑曲线而是**颗粒状**,
    # 这正是原片那股"沙"质感的来源(实测原片绿区内部 std 23.7, 不是平涂)。
    #
    # 实现走**全 OpenCV 路径**: 抖动图缓存成 uint8 偏置(不是每帧 np.tile 出
    # float 再乘), 用 cv2.add 一次加完; 阈值+上色合成一张 256 项 LUT, 每通道
    # 一次 cv2.LUT。原先用 int16 加法 + 三次 np.roll/np.where 要 2.4ms,
    # 中途试过"抖动烘焙成 16 张相位表"反而更慢(16 次 copyTo, 11.9ms), 已否掉。
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)
    # 偏置写成无符号: 先给灰度减去 dither/2 的直流, 抖动就能全用加法
    bias = _Tiled(np.clip(bayer / 16.0 * dither, 0, 255).astype(np.uint8))
    dc = int(round(dither * 0.5))
    luts = [
        np.where(np.arange(256, dtype=np.int16) >= thresh + dc, light[i], dark[i]).astype(np.uint8)
        for i in range(3)
    ]

    def fn(src: np.ndarray) -> np.ndarray:
        h, w = src.shape[:2]
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        g = cv2.add(g, bias.take(h, w))  # 饱和加法, 不会回绕
        # 错位靠**扩边**取窗, 不用 np.roll —— roll 是环绕的, 会把面最左边
        # split 列卷到最右边去(实测左白右黑的输入, 最右一列 B 从 20 跳到 238)。
        # 那条 5px 宽的假套色带偏偏长得像 riso 本来的套色不准, 肉眼认不出是
        # bug, 但它的内容来自面的另一侧, 盒子一转就跟着乱变。复制边缘像素则
        # 只是把边上那一列拉长, 与"整张版错开一点"的物理语义一致。
        pad = cv2.copyMakeBorder(g, 0, 0, split, split, cv2.BORDER_REPLICATE)
        return cv2.merge(
            [cv2.LUT(pad[:, split - o : split - o + w], luts[i]) for i, o in enumerate(offs)]
        )

    return fn


def _fx_quadtree(
    min_cell: int, max_depth: int, var_thresh: float, line: tuple[int, int, int]
) -> FaceEffect:
    """四叉树自适应马赛克: 细节多的地方递归切小块, 平坦区留大块, 每块描边.

    来自 douyin 那段 TouchDesigner 屏录 —— 脸部切到最细、白墙保持大块, 是
    四个特效里辨识度最高的一个。

    怎么做到实时: 用**积分图**(cv2.integral 一次给出 sum 和 sum²), 于是任意
    矩形的均值/方差都是 O(1) 的四点查表, 与块大小无关。递归本身只对"够花"的
    块继续下探, 平坦区一层就停 —— 实际访问的节点数远小于满四叉树。

    切分判据是**亮度方差**: 方差大 = 块内有边缘/纹理 = 值得再切。
    """
    ink = np.array(line, np.uint8)

    def fn(src: np.ndarray) -> np.ndarray:
        full_h, full_w = src.shape[:2]
        src = _shrink_for_work(src, QUAD_WORK_MAX)  # 块结构是低频的, 缩略图上算够用
        h, w = src.shape[:2]
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # 一次算好: 灰度的 sum/sum²(判方差) + 彩色的 sum(取叶子均色)。
        # 有了它们, 递归里再没有任何与块面积成正比的运算。
        sm, sq = cv2.integral2(g)
        sc = cv2.integral(src)  # (h+1, w+1, 3)

        def box(t: np.ndarray, x: int, y: int, bw: int, bh: int):
            return t[y + bh, x + bw] - t[y, x + bw] - t[y + bh, x] + t[y, x]

        out = np.empty_like(src)
        stack = [(0, 0, w, h, 0)]
        while stack:
            x, y, bw, bh, d = stack.pop()
            if bw < 2 or bh < 2:
                continue
            n = bw * bh
            mean = box(sm, x, y, bw, bh) / n
            var = box(sq, x, y, bw, bh) / n - mean * mean
            if d < max_depth and bw > min_cell * 2 and bh > min_cell * 2 and var > var_thresh:
                hw, hh = bw // 2, bh // 2
                stack.append((x, y, hw, hh, d + 1))
                stack.append((x + hw, y, bw - hw, hh, d + 1))
                stack.append((x, y + hh, hw, bh - hh, d + 1))
                stack.append((x + hw, y + hh, bw - hw, bh - hh, d + 1))
                continue
            cell = out[y : y + bh, x : x + bw]
            cell[:] = box(sc, x, y, bw, bh) / n  # 叶子均色, O(1)
            cell[0, :] = ink  # 上边
            cell[:, 0] = ink  # 左边
        if (h, w) != (full_h, full_w):
            out = cv2.resize(out, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        return out

    return fn


def _fx_pointcloud(
    cell: int, gain: float, floor: int, glow: int, tint: tuple[int, int, int]
) -> FaceEffect:
    """点云 / 全息扫描: 主体拆成发光的青蓝点阵, 暗处近黑.

    对照 douyin 那段 TouchDesigner 屏录实测: 亮部色相 H≈98(青蓝, p10-p90 只有
    93-101 → 单一色相)、饱和 S≈101、亮点密度 13.3%(平均间距 2.7px)、暗区占 27%。

    **这是单目近似**。原片那种"手在前更实、身体在后更淡"的分层来自深度相机,
    这里没有深度信息, 用**局部对比度**代替: 边缘/纹理强的地方点更亮更密, 平坦
    区衰减。观感神似(都是"物体由发光点采样而成"), 但不是同一个物理量 —— 想要
    真分层得接深度相机或跑分割网络。

    实现: 点阵掩码(cell 周期的规则栅格) × 局部对比度增益 → 单色相着色。
    全程 OpenCV, 无逐点循环。
    """
    tint_lut = np.clip(
        np.arange(256, dtype=np.float32)[:, None] * (np.array(tint, np.float32) / 255.0), 0, 255
    ).astype(np.uint8)[:, None, :]
    # 点内开洞用的掩码: 只留每个 cell 中心一小块。它是 cell 周期的固定图案,
    # 铺一次就能一直用 —— 原先每帧 np.tile, 实测占这个 effect 的 17%。
    hole = np.zeros((cell, cell), np.uint8)
    _c0 = cell // 2
    hole[max(_c0 - 1, 0) : _c0 + 1, max(_c0 - 1, 0) : _c0 + 1] = 255
    holes = _Tiled(hole)

    def fn(src: np.ndarray) -> np.ndarray:
        full_h, full_w = src.shape[:2]
        src = _shrink_for_work(src, PC_WORK_MAX)  # 点阵是 cell 周期的低频结构
        h, w = src.shape[:2]
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # 1) 采样成点阵: 缩到 1/cell 再最近邻放大 —— 每个 cell 只剩一个采样值,
        #    这一步才是"点云"的来源(先前版本用整幅栅格相与, 点太稀且丢结构)。
        sh, sw = max(h // cell, 2), max(w // cell, 2)
        small = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA)
        # 2) 每个点的亮度 = 该处结构强度。局部对比度当伪深度: 边缘/纹理处的点亮,
        #    平坦区(墙面/背景)的点暗下去 —— 视觉上物体"浮"出黑底。
        det = cv2.absdiff(small, cv2.GaussianBlur(small, (0, 0), 2.0))
        v = cv2.addWeighted(small, 0.35, cv2.convertScaleAbs(det, alpha=gain), 1.0, -float(floor))
        pts = cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST)
        # 3) 点内开洞: 只留每个 cell 的中心一小块, 其余归零 → 看得见"点"
        pts = cv2.bitwise_and(pts, holes.take(h, w))
        # 4) 辉光: 点扩散成小光斑, 叠回自身 → 发光感而不是硬像素
        if glow:
            pts = cv2.addWeighted(pts, 1.0, cv2.GaussianBlur(pts, (0, 0), glow), 1.6, 0.0)
        out = cv2.applyColorMap(pts, tint_lut)
        if (h, w) != (full_h, full_w):
            out = cv2.resize(out, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        return out

    return fn


def _halftone_thresh(cell: int) -> np.ndarray:
    """一个 cell 的阈值瓦片: 到中心距离² 归一化. 越靠边阈值越低 → 点从中心长大."""
    c = (cell - 1) * 0.5
    y, x = np.mgrid[0:cell, 0:cell].astype(np.float32)
    r = np.hypot(y - c, x - c) / max(c, 1e-6)
    return np.clip(1.0 - r * r, 0.0, 1.0)


def _fx_halftone(cell: int, paper: tuple[int, int, int], ink: tuple[int, int, int]) -> FaceEffect:
    """半调网点: 亮度低于"到中心距离"阈值的像素上墨 → 点随暗部长大.

    阈值图交给 _Tiled 按需增长后长期复用, 每帧只剩一次切片(视图, 免费)+ 一次
    uint8 比较 + 一次调色板索引。
    """
    thresh = _Tiled((_halftone_thresh(cell) * 255.0).astype(np.uint8))
    duo = np.empty((256, 1, 3), np.uint8)
    duo[:128] = np.array(paper, np.uint8)  # 掩码 0 = 亮 = 纸
    duo[128:] = np.array(ink, np.uint8)  # 掩码 255 = 暗 = 墨

    def fn(src: np.ndarray) -> np.ndarray:
        h, w = src.shape[:2]
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # compare → 0/255 掩码, 再走 applyColorMap 的双色表; 全程 OpenCV, 不做
        # numpy 花式索引(那一步实测占 10ms 里的 8ms)
        return cv2.applyColorMap(cv2.compare(g, thresh.take(h, w), cv2.CMP_LT), duo)

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
    Face("前", (0, 4, 6, 2), _fx_riso(RISO_DARK, RISO_LIGHT, RISO_THRESH, RISO_SPLIT, RISO_DITHER)),
    Face("背", (1, 5, 7, 3), _fx_duotone(DUOTONE_DARK, DUOTONE_LIGHT, DUOTONE_THRESH)),
    Face("底", (0, 4, 5, 1), _fx_pointcloud(PC_CELL, PC_GAIN, PC_FLOOR, PC_GLOW, PC_TINT)),
    Face("顶", (2, 6, 7, 3), _fx_lut(BLUE_LUT), is_top=True),
    Face("左", (0, 2, 3, 1), _fx_quadtree(QUAD_MIN_CELL, QUAD_MAX_DEPTH, QUAD_VAR, QUAD_LINE)),
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
