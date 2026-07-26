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

from .handgeom import palm_center
from .landmarks import INDEX_MCP, INDEX_TIP, PINKY_MCP, THUMB_TIP
from .tracker import HandPose

# ---- tunables ----
CUBE_SIZE0 = 0.22  # 初始边长(占画面短边的比例)
CUBE_SIZE_MIN = 0.06
CUBE_SIZE_MAX = 0.60
CUBE_FOCAL = 2.2  # 弱透视焦距 = 该值 × 边长; 越小透视越夸张
PINCH_ON = 0.42  # 捏合判据: |拇指尖−食指尖| / 掌宽 低于它 = 捏住
PINCH_OFF = 0.62  # 高于它 = 松开(滞回, 防在临界处抖动误触)
ORBIT_GAIN = 0.016  # 拖动 1px 转多少弧度(≈ 拖 196px 转 180°, 一次手臂行程能翻一圈多)
DRIFT = 0.006  # 松手后的稳态自转(rad/帧 ≈ 10°/秒), 让它看着是"活的"
SPIN_DAMP = 0.85  # 松手瞬间的惯性衰减(每帧乘这个; 大=松手后还飘很久)
# 单帧拖动上限, 定在**像素域**而不是角度域: 它的职责是挡检测跳变(实测真手
# p99 = 109 px/帧, 而误检瞬移能到 878 px/帧), 跟灵敏度无关。原先写成角速度
# 上限, 除以 gain 只剩 22.7 px/帧 —— 实测 30.6% 的帧被误伤, 手一快就"拖了
# 转不动"; 而且调大 gain 会让限幅更早触发, 越调灵敏越不跟手。
DRAG_MAX_PX = 220.0
# 平移/缩放是 1:1 跟手的 —— 手走多远立方体走多远, 中间不打任何折扣。
# 原先这里有个 MOVE_SMOOTH=0.35 系数乘在**位移增量**上, 名字叫"平滑"其实是
# 增益打折: 手拖 100px 立方体只走 35px, 拖起来永远追不上手。实测双手掌心中点
# 的静止残余抖动只有 1.54 px/帧(边长 232px 的 0.7%), 双手距离 0.26%, 1:1
# 直接跟完全撑得住, 不需要拿增益去换稳。
# 双手模式的宽限帧数. 双手捏在一起时两只手互相遮挡, MediaPipe 会偶尔只认出
# 一只 —— 掉一帧就切回"单手=转动", 哥哥明明在推它, 它却突然转起来。宽限期内
# 宁可停住不动, 也不要乱转。
TWO_HAND_HOLD = 8

MODE_IDLE, MODE_TURN, MODE_MOVE = "idle", "turn", "move"


def _pinch_ratio(hand: HandPose) -> float:
    """捏合程度 = |拇指尖−食指尖| / 掌宽. 用比值, 与手离镜头远近无关."""
    xy = hand.points[:, :2].astype(np.float32)
    palm = float(np.linalg.norm(xy[INDEX_MCP] - xy[PINKY_MCP]))
    return float(np.linalg.norm(xy[THUMB_TIP] - xy[INDEX_TIP])) / max(palm, 1e-3)


def _drag_point(hand: HandPose) -> np.ndarray:
    """拖动的参考点: 掌心, 不是捏合点.

    捏合点(拇指尖+食指尖中点)看着更"抓得住", 但那两个指尖捏在一起时互相
    遮挡, 是全手最不准的两个 landmark —— 实测静止抖动 2.66 px/帧, 掌心
    (PALM_RING 多点平均)只有 1.94。捏住之后本来就是整只手在移动, 掌心
    一样跟得上, 还稳 27%。
    """
    return palm_center(hand)[:2]


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
        self._hold = 0  # 双手模式的剩余宽限帧
        # 下面两个是给 renderer 画反馈用的只读状态 —— 本模块自己不碰画布
        self.mode = MODE_IDLE
        self.marks: list[tuple[np.ndarray, bool]] = []  # (掌心, 这只手捏住了没)
        self.debug = ""

    def reset(self) -> None:
        self.pos = None
        self._grab = None
        self._two = None
        self._hold = 0
        self._spin[:] = 0.0
        self._pinch.clear()

    @staticmethod
    def _key(index: int, hand: HandPose) -> int:
        """手的跨帧身份. 滤波关掉时 track_id 是 -1, 退回列表下标."""
        return hand.track_id if hand.track_id >= 0 else -1 - index

    def _pinching(self, keys: list[int], ratios: list[float]) -> list[bool]:
        out = []
        for k, r in zip(keys, ratios, strict=True):  # 同源于 use, 等长
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
        pins = self._pinching(keys, [_pinch_ratio(hd) for hd in use])
        pts = [_drag_point(hd) for hd in use]
        n = sum(pins)
        self.marks = list(zip(pts, pins, strict=True))

        if n >= 2:  # ---- 双手: 平移 + 缩放 ----
            self._grab = None
            pair = frozenset(keys)
            mid = (pts[0] + pts[1]) * 0.5
            dist = float(np.linalg.norm(pts[1] - pts[0]))
            if self._two is not None and self._two[0] == pair:  # 换了一双手就重新起算
                _, pmid, pdist = self._two
                self.pos += mid - pmid  # 1:1: 两手中点走多远, 立方体走多远
                if pdist > 1e-3:
                    self.size = float(
                        np.clip(
                            self.size * (dist / pdist),  # 1:1: 两手拉开多少倍, 它就大多少倍
                            CUBE_SIZE_MIN * min(h, w),
                            CUBE_SIZE_MAX * min(h, w),
                        )
                    )
                # 别让它被推出画面 —— 推出去就只能按 X 归位了
                m = self.size * 0.5
                self.pos = np.clip(self.pos, (m, m), (w - m, h - m)).astype(np.float32)
            self._two = (pair, mid.copy(), dist)
            self._spin *= SPIN_DAMP
            self._hold = TWO_HAND_HOLD
            self.mode = MODE_MOVE
        elif self._hold > 0:  # ---- 双手模式掉了一帧: 停住, 别掉去转动 ----
            self._hold -= 1
            self._grab = None
            self._two = None  # 手回来时重新起算基线, 免得攒出一次跳变
            self._spin *= SPIN_DAMP
            self.mode = MODE_MOVE
        elif n == 1:  # ---- 单手: 拖动 = 转动 ----
            self._two = None
            i = pins.index(True)
            p, k = pts[i], keys[i]
            if self._grab is not None and self._grab[0] == k:  # 换手就重新起算
                d = p - self._grab[1]
                mag = float(np.linalg.norm(d))
                if mag > DRAG_MAX_PX:  # 这么大只可能是检测跳变, 不是真手
                    d = d * (DRAG_MAX_PX / mag)
                # 横拖 → 绕相机竖轴(y); 竖拖 → 绕相机横轴(x)。屏幕 y 朝下,
                # 所以往下拖时立方体上缘朝自己转过来, 手感与真实抓握一致。
                # 拖多少转多少, 中间不再打折 —— 打折就是"不跟手"的来源。
                self._spin = np.array(
                    [d[1] * self.orbit_gain, d[0] * self.orbit_gain, 0.0], np.float32
                )
            self._grab = (k, p.copy())
            self.mode = MODE_TURN
        else:  # ---- 松手: 惯性 + 极慢自转 ----
            self._grab = None
            self._two = None
            self._spin *= SPIN_DAMP
            self._spin[1] += self.drift * (1.0 - SPIN_DAMP)  # 稳态收敛到 drift
            self.mode = MODE_IDLE

        ang = float(np.linalg.norm(self._spin))
        if ang > 1e-6:
            self.rot = _rot(self._spin / ang, ang) @ self.rot
            # 累积浮点误差会让 rot 慢慢不正交(立方体会被剪切), 用 SVD 拉回来
            u, _s, vt = np.linalg.svd(self.rot)
            self.rot = (u @ vt).astype(np.float32)
        self.debug = f"边长{self.size:4.0f} 捏住{n}只手"

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
