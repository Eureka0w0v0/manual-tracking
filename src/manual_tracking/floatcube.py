"""悬浮立方体 —— 有自己位姿的物体, 手是控制器.

与 glassbox 的根本区别: 玻璃盒**没有自己的状态**, 每帧由双手的位置/姿态直接
解出来, 手一走它就没了; 这里的立方体**自己存着位置、朝向、大小**, 手只是去
推它转它, 松手后它留在原地继续飘。

交互(两个手势, 都只用 2D 位置增量 —— 那是 landmark 里最可靠的量):
  单手捏住拖动    → 转动立方体(横拖绕竖轴, 竖拖绕横轴), 用来翻面
  双手同时捏住    → 平移(跟两手中点) + 缩放(跟两手距离)
  松开           → 停在当前位姿, 带一点惯性和极慢自转

为什么转动用"拖动增量"而不是"手腕朝向":
  handgeom.orient 是掌面有向面积, 数学上是 cos 型 —— 整圈里走 0→+1→0→−1,
  既饱和又非单调(见 docs/GLASS_BOX_GEOMETRY.md §6)。玻璃盒受制于它, 一直
  往一个方向翻会自己转回来。拖动增量没有这个毛病: 手往一个方向拖多远,
  立方体就转多少, 可以无限累积。
"""

from __future__ import annotations

import numpy as np

from .landmarks import INDEX_MCP, INDEX_TIP, PINKY_MCP, THUMB_TIP
from .tracker import HandPose

# ---- tunables ----
CUBE_SIZE0 = 0.22  # 初始边长(占画面短边的比例)
CUBE_SIZE_MIN = 0.06
CUBE_SIZE_MAX = 0.60
CUBE_FOCAL = 2.2  # 弱透视焦距 = 该值 × 边长; 越小透视越夸张
PINCH_ON = 0.42  # 捏合判据: |拇指尖−食指尖| / 掌宽 低于它 = 捏住
PINCH_OFF = 0.62  # 高于它 = 松开(滞回, 防在临界处抖动误触)
ORBIT_GAIN = 0.011  # 拖动 1px 转多少弧度(≈ 拖 285px 转 180°)
DRIFT = 0.006  # 松手后的稳态自转(rad/帧 ≈ 10°/秒), 让它看着是"活的"
SPIN_DAMP = 0.90  # 松手瞬间的惯性衰减(每帧乘这个)
SPIN_MAX = 0.25  # 单帧角速度上限(rad), 防误检把立方体甩飞
MOVE_SMOOTH = 0.35  # 平移/缩放的跟手程度(0=不动, 1=完全跟手)


def _pinch_ratio(hand: HandPose) -> tuple[float, np.ndarray]:
    """捏合程度(|拇指尖−食指尖| / 掌宽)与捏合点. 比值与手离镜头远近无关."""
    xy = hand.points[:, :2].astype(np.float32)
    t, i = xy[THUMB_TIP], xy[INDEX_TIP]
    palm = float(np.linalg.norm(xy[INDEX_MCP] - xy[PINKY_MCP]))
    return float(np.linalg.norm(t - i)) / max(palm, 1e-3), (t + i) * 0.5


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    """绕单位轴转 ang 弧度的旋转矩阵(Rodrigues)."""
    c, s = float(np.cos(ang)), float(np.sin(ang))
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        np.float32,
    )


class FloatCube:
    """悬浮立方体: 自带位置/朝向/大小, 手去操作它."""

    def __init__(self) -> None:
        self.orbit_gain = ORBIT_GAIN  # live 实时可调
        self.drift = DRIFT
        self.pos: np.ndarray | None = None  # 屏幕中心 (x, y); None = 还没放置
        self.size = 0.0  # 边长(px)
        self.rot = np.eye(3, dtype=np.float32)  # 立方体局部系 → 相机系
        self._spin = np.zeros(3, np.float32)  # 当前角速度(轴×角, rad/帧)
        # 下面三份跨帧状态都按 track_id 记, 不按列表下标: 两只手在 hands[] 里的
        # 先后顺序会随检测结果对调, 用下标存会把左手的拖动基准接到右手上, 一帧
        # 蹦出个几百 px 的假增量。
        self._pinch: dict[int, bool] = {}  # 每只手的捏合滞回状态
        self._grab: tuple[int, np.ndarray] | None = None  # 单手拖动: (手, 上帧捏合点)
        self._two: tuple[frozenset[int], np.ndarray, float] | None = None  # 双手: (手对, 中点, 距离)
        self.debug = ""

    def reset(self) -> None:
        self.pos = None
        self._grab = None
        self._two = None
        self._spin[:] = 0.0
        self._pinch.clear()

    @staticmethod
    def _key(index: int, hand: HandPose) -> int:
        """手的跨帧身份. 滤波关掉时 track_id 是 -1, 退回列表下标."""
        return hand.track_id if hand.track_id >= 0 else -1 - index

    def _pinching(self, keys: list[int], ratios: list[float]) -> list[bool]:
        out = []
        for k, r in zip(keys, ratios):
            self._pinch[k] = r < (PINCH_OFF if self._pinch.get(k, False) else PINCH_ON)
            out.append(self._pinch[k])
        for gone in set(self._pinch) - set(keys):  # 手离场就忘掉它, 别无限攒
            del self._pinch[gone]
        return out

    def update(self, hands: list[HandPose], shape: tuple[int, int]) -> None:
        """吃这一帧的手, 更新立方体的位姿. shape = (h, w)."""
        h, w = shape
        if self.pos is None:  # 首次出现: 摆在画面中央
            self.pos = np.array([w * 0.5, h * 0.5], np.float32)
            self.size = CUBE_SIZE0 * min(h, w)

        use = hands[:2]
        keys = [self._key(i, hd) for i, hd in enumerate(use)]
        ratios, pts = zip(*(_pinch_ratio(hd) for hd in use)) if use else ((), ())
        pins = self._pinching(keys, list(ratios))
        n = sum(pins)

        if n >= 2:  # ---- 双手: 平移 + 缩放 ----
            self._grab = None
            pair = frozenset(keys)
            mid = (pts[0] + pts[1]) * 0.5
            dist = float(np.linalg.norm(pts[1] - pts[0]))
            if self._two is not None and self._two[0] == pair:  # 换了一双手就重新起算
                _, pmid, pdist = self._two
                self.pos += (mid - pmid) * MOVE_SMOOTH
                if pdist > 1e-3:
                    self.size = float(
                        np.clip(
                            self.size * (1.0 + (dist / pdist - 1.0) * MOVE_SMOOTH),
                            CUBE_SIZE_MIN * min(h, w),
                            CUBE_SIZE_MAX * min(h, w),
                        )
                    )
                # 别让它被推出画面 —— 推出去就只能按 X 归位了
                m = self.size * 0.5
                self.pos = np.clip(self.pos, (m, m), (w - m, h - m)).astype(np.float32)
            self._two = (pair, mid.copy(), dist)
            self._spin *= SPIN_DAMP
        elif n == 1:  # ---- 单手: 拖动 = 转动 ----
            self._two = None
            i = pins.index(True)
            p, k = pts[i], keys[i]
            if self._grab is not None and self._grab[0] == k:  # 换手就重新起算
                d = p - self._grab[1]
                # 横拖 → 绕相机竖轴(y); 竖拖 → 绕相机横轴(x)。屏幕 y 朝下,
                # 所以往下拖时立方体上缘朝自己转过来, 手感与真实抓握一致。
                self._spin = np.array(
                    [d[1] * self.orbit_gain, d[0] * self.orbit_gain, 0.0], np.float32
                )
                mag = float(np.linalg.norm(self._spin))
                if mag > SPIN_MAX:  # 误检限幅: 手不可能一帧转这么多
                    self._spin *= SPIN_MAX / mag
            self._grab = (k, p.copy())
        else:  # ---- 松手: 惯性 + 极慢自转 ----
            self._grab = None
            self._two = None
            self._spin *= SPIN_DAMP
            self._spin[1] += self.drift * (1.0 - SPIN_DAMP)  # 稳态收敛到 drift

        ang = float(np.linalg.norm(self._spin))
        if ang > 1e-6:
            self.rot = _rot(self._spin / ang, ang) @ self.rot
            # 累积浮点误差会让 rot 慢慢不正交(立方体会被剪切), 用 SVD 拉回来
            u, _s, vt = np.linalg.svd(self.rot)
            self.rot = (u @ vt).astype(np.float32)
        self.debug = f"size{self.size:4.0f} spin{ang:5.3f} pinch{n}"

    def project(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
        """→ (屏幕 8 顶点, 相机系 8 顶点, 焦距). 顶点序与 effects.BOX_FACES 一致."""
        s = self.size * 0.5
        focal = CUBE_FOCAL * self.size
        cam = np.empty((8, 3), np.float32)
        scr = np.empty((8, 2), np.float32)
        for xi, x in enumerate((-s, s)):
            for ui, u in enumerate((-s, s)):
                for wi, wv in enumerate((s, -s)):
                    p3 = self.rot @ np.array([x, u, wv], np.float32)
                    k = xi * 4 + ui * 2 + wi
                    # 相机系右手基: x 右, y 上(屏幕 y 朝下故取负), z 朝观察者
                    cam[k] = (p3[0], -p3[1], p3[2])
                    scr[k] = self.pos + p3[:2] * (focal / (focal - p3[2]))
        return scr, cam, focal
