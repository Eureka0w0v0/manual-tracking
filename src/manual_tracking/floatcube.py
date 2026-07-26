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

from .handgeom import palm_center, pinch
from .landmarks import INDEX_MCP, INDEX_TIP, PINKY_MCP, PINKY_TIP, THUMB_TIP
from .tracker import HandPose

# ---- tunables ----
CUBE_SIZE0 = 0.22  # 初始边长(占画面短边的比例)
CUBE_SIZE_MIN = 0.06
CUBE_SIZE_MAX = 0.60
CUBE_FOCAL = 2.2  # 弱透视焦距 = 该值 × 边长; 越小透视越夸张
PINCH_ON = 0.42  # 捏合判据: |拇指尖−食指尖| / 掌宽 低于它 = 捏住
PINCH_OFF = 0.62  # 高于它 = 松开(滞回, 防在临界处抖动误触)
# 拖动 1px 转多少弧度。0.016 (拖 196px 翻 180°) 实拍反馈"太快、偶尔飞":
# 真手 p99=109px/帧 在那个档就是 100°/帧 = 3000°/s。0.009 → 拖 349px 翻 180°
# (屏宽 18% 一次从容行程翻一两个面), p99 降到 56°/帧。live 9/0 键随时调回。
ORBIT_GAIN = 0.009
# 拖动的一阶惯性: ω 每帧向"手速×gain"靠拢这个比例(1 = 直接控制, 零质量)。
# 0.40 → 时间常数 2.5 帧 ≈ 83ms: 起步能感到它"有分量", 停手 5 帧内被黏住;
# 误检尖峰只能把 ω 拉高 40%, 下一帧就被拉回 —— 质量本身就是滤波。
TURN_RESP = 0.40
DRIFT = 0.006  # 松手后的稳态自转(rad/帧 ≈ 10°/秒), 让它看着是"活的"
# 松手摩擦(每帧乘这个)。0.85 → 半衰 4.3 帧: 甩一下滑 1 秒左右渐停。
# 收尾角速度 v 的滑行总角 = v·damp/(1−damp), 被 CARRY_MAX_RAD 封顶在 100°。
SPIN_DAMP = 0.85
# 单帧拖动上限, 定在**像素域**而不是角度域: 它的职责是挡检测跳变(实测真手
# p99 = 109 px/帧, 而误检瞬移能到 878 px/帧), 跟灵敏度无关。原先写成角速度
# 上限, 除以 gain 只剩 22.7 px/帧 —— 实测 30.6% 的帧被误伤, 手一快就"拖了
# 转不动"; 而且调大 gain 会让限幅更早触发, 越调灵敏越不跟手。
DRAG_MAX_PX = 220.0
# 单帧转角目标的硬顶(角度域)。与 DRAG_MAX_PX 是**两层不同职责**: 像素层挡
# "位移大得离谱"(878px 瞬移), 但 220px 截完 × gain 仍是 113°, 而立方体对称
# 周期才 90° —— 目标越过一个周期, 面等于随机换。90° 在 gain=0.009 下等效
# 175px, 真手 p99=109px/帧 够不到 → 零误伤。gain 被 9/0 键调大时自动变兜底。
TURN_MAX_RAD = float(np.radians(90.0))
# 松手能带走的角速度上限。滑行总角 = 15°/(1−0.85) = 100° —— 猛甩正好翻过
# 一个面周期, 悠闲松手(收尾 ω 8°/帧 上下)滑半个面, 都可预期。
CARRY_MAX_RAD = float(np.radians(15.0))
# 被双手握住时的角速度衰减: 真实物体被两只手抓死, 旋转应当立刻锁住(3 帧 <13%)。
GRIP_DAMP = 0.5
# 平移/缩放是 1:1 跟手的 —— 手走多远立方体走多远, 中间不打任何折扣。
# 原先这里有个 MOVE_SMOOTH=0.35 系数乘在**位移增量**上, 名字叫"平滑"其实是
# 增益打折: 手拖 100px 立方体只走 35px, 拖起来永远追不上手。实测双手掌心中点
# 的静止残余抖动只有 1.54 px/帧(边长 232px 的 0.7%), 双手距离 0.26%, 1:1
# 直接跟完全撑得住, 不需要拿增益去换稳。
# 双手模式的宽限帧数. 双手捏在一起时两只手互相遮挡, MediaPipe 会偶尔只认出
# 一只 —— 掉一帧就切回"单手=转动", 哥哥明明在推它, 它却突然转起来。宽限期内
# 宁可停住不动, 也不要乱转。
TWO_HAND_HOLD = 8

# ---- 平移/缩放惯性(与转动同一套物理语言: 带走动量 → 摩擦滑行 → 碰壁反弹) ----
MOVE_DAMP = 0.90  # 平移摩擦(每帧乘); 半衰 6.6 帧, 甩出去滑 ~1 秒
MOVE_CARRY_MAX = 40.0  # 松手能带走的平移速度上限(px/帧); 滑行总程 = 40×0.9/0.1 = 360px
BOUNCE = 0.55  # 碰壁后速度保留比例; 两三次弹跳后自然停下
ZOOM_DAMP = 0.70  # 缩放速率摩擦; 尺寸是指数量, 惯性给短一点, 否则会滚雪球
ZOOM_CARRY_MAX = 0.04  # 松手能带走的缩放速率上限(倍/帧)
# ---- 炸开视图(exploded view): 张开的手 → 六个面沿外法线飞离体心 ----
# 驱动量 = 未捏合手的 指弧展开量/掌宽(与手离镜头远近无关)。捏住的手在控制
# 转动, 不参与; 双手都捏着(move)也不炸。阈值若不跟手, 抬手看 HUD debug 里
# 的"张开"实测值, 改下面两个数即可。
EXPLODE_LO = 0.70  # 比值低于它 = 完全收拢(合成手半开姿态是 0.56)
EXPLODE_HI = 1.05  # 高于它 = 全炸开
EXPLODE_RESP = 0.30  # 炸开程度的一阶惯性(和转动同款手感语言)
EXPLODE_DIST = 0.55  # 全炸开时面心飞离体心的距离(边长的倍数)
RIPPLE_LIFE = 7  # 抓握涟漪寿命(帧); floatcube 负责老化, renderer 只读
# 抓取判定半径(边长的倍数): 捏点落进 pos ± size×这个值 才算"抓住了"。
# 只在**建立**抓取那一刻判定 —— 建立后跟手保持(双手缩放会把手拉出半径,
# 中途脱手等于"捏着的东西自己滑掉"), 松开捏合才断。
GRAB_RADIUS = 0.75
# 抓着的手检测丢失的宽限帧数(≈250ms@30fps, 与 tracker 的 SLOT_TTL 同源)。
# MediaPipe 在快速移动/翻腕遮挡时掉 1-3 帧检测是常态; 没有宽限的话, 掉一帧
# 抓取就断, 手回来时早被拖出盒子半径 → 永远抓不回去, 体感"半路脱手"。
GRAB_TTL = 8
# 松开捏合需要连续这么多帧确认(建立仍是瞬时的 —— 响应优先, 断开保守)。
# 拖动翻腕时拇指/食指尖被自己手背遮挡, landmark 单帧乱跳会把比值冲过
# PINCH_OFF —— 那是误检不是松手。3 帧 = 100ms, 真实松手的延迟无感。
PINCH_OFF_FRAMES = 3

MODE_IDLE, MODE_TURN, MODE_MOVE = "idle", "turn", "move"


def _pinch_ratio(hand: HandPose) -> float:
    """捏合程度 = |拇指尖−食指尖| / 掌宽. 用比值, 与手离镜头远近无关."""
    xy = hand.points[:, :2].astype(np.float32)
    palm = float(np.linalg.norm(xy[INDEX_MCP] - xy[PINKY_MCP]))
    return float(np.linalg.norm(xy[THUMB_TIP] - xy[INDEX_TIP])) / max(palm, 1e-3)


def _spread_ratio(hand: HandPose) -> float:
    """张开程度 = 指弧展开量(食指尖↔小指尖) / 掌宽. 握拳变小, 五指张开变大."""
    xy = hand.points[:, :2]
    spread = float(np.linalg.norm(xy[INDEX_TIP] - xy[PINKY_TIP]))
    palm = float(np.linalg.norm(xy[INDEX_MCP] - xy[PINKY_MCP]))
    return spread / max(palm, 1e-3)


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
        self._vel = np.zeros(2, np.float32)  # 平移速度(px/帧); 松手带走 → 滑行/反弹
        self._zoom = 0.0  # 缩放速率(倍/帧 − 1); 松手带走, 快收敛
        # 下面三份跨帧状态都按 track_id 记, 不按列表下标: 两只手在 hands[] 里的
        # 先后顺序会随检测结果对调, 用下标存会把左手的拖动基准接到右手上, 一帧
        # 蹦出个几百 px 的假增量。
        self._pinch: dict[int, bool] = {}  # 每只手的捏合滞回状态
        self._unpin: dict[int, int] = {}  # 松开确认计数(见 PINCH_OFF_FRAMES)
        self._grabbing: dict[int, bool] = {}  # 每只手"抓住盒子没"(捏合∧建立时在盒上)
        self._grab_ttl: dict[int, int] = {}  # 检测丢手的抓取宽限(见 GRAB_TTL)
        self._grab: tuple[int, np.ndarray] | None = None  # 单手拖动: (手, 上帧捏合点)
        self._two: tuple[frozenset[int], np.ndarray, float] | None = None  # 双手: (手对, 中点, 距离)
        self._hold = 0  # 双手模式的剩余宽限帧
        # 下面两个是给 renderer 画反馈用的只读状态 —— 本模块自己不碰画布
        self.mode = MODE_IDLE
        self.marks: list[tuple[np.ndarray, bool]] = []  # (掌心, 这只手抓住盒子没)
        self.explode = 0.0  # 炸开程度 0..1(张开手→1); renderer 只读
        self.ripples: list[tuple[np.ndarray, int]] = []  # (捏点, 已存活帧数); 只读
        self.debug = ""

    def reset(self) -> None:
        self.pos = None
        self._grab = None
        self._two = None
        self._hold = 0
        self._spin[:] = 0.0
        self._vel[:] = 0.0
        self._zoom = 0.0
        self.explode = 0.0
        self.ripples.clear()
        self._pinch.clear()
        self._unpin.clear()
        self._grabbing.clear()
        self._grab_ttl.clear()

    @staticmethod
    def _key(index: int, hand: HandPose) -> int:
        """手的跨帧身份. 滤波关掉时 track_id 是 -1, 退回列表下标."""
        return hand.track_id if hand.track_id >= 0 else -1 - index

    def _pinching(self, keys: list[int], ratios: list[float]) -> list[bool]:
        out = []
        for k, r in zip(keys, ratios, strict=True):  # 同源于 use, 等长
            was = self._pinch.get(k, False)
            raw = r < (PINCH_OFF if was else PINCH_ON)
            if was and not raw:
                # 松开要连帧确认: 单帧尖峰(指尖被遮挡时 landmark 乱跳)不算松手
                self._unpin[k] = self._unpin.get(k, 0) + 1
                pin = self._unpin[k] < PINCH_OFF_FRAMES
            else:
                self._unpin[k] = 0
                pin = raw
            self._pinch[k] = pin
            out.append(pin)
        for gone in set(self._pinch) - set(keys):  # 手离场就忘掉它, 别无限攒
            del self._pinch[gone]
            self._unpin.pop(gone, None)
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

        # 抓取 = 捏合 ∧ 捏点落在盒子上(只在建立时判定, 见 GRAB_RADIUS)。
        # 为什么必须加位置条件: 捏合判据是拇指尖/食指尖的 2D **投影**距离,
        # 手朝镜头伸过去"触碰"时指尖投影天然挤在一起 → 误判捏合 —— 双手一伸
        # 就进 move, 盒子跟着乱跑。捏在空气里现在不控制任何东西。
        # 涟漪 = 抓取**建立**的回执(捏空气没有), 顺手在同一处记录。
        self.ripples = [(rp, ra + 1) for rp, ra in self.ripples if ra + 1 <= RIPPLE_LIFE]
        r_grab = self.size * GRAB_RADIUS
        grabs: list[bool] = []
        for hd, k2, pin2 in zip(use, keys, pins, strict=True):
            held = pin2 and self._grabbing.get(k2, False)
            if pin2 and not held:
                cpt = pinch(hd)[2]
                if float(np.linalg.norm(cpt - self.pos)) <= r_grab:
                    held = True
                    self.ripples.append((cpt, 0))
            grabs.append(held)
            self._grabbing[k2] = held
        for gone in set(self._grabbing) - set(keys):
            if self._grabbing[gone] and self._grab_ttl.get(gone, GRAB_TTL) > 0:
                # 检测短暂丢手: 抓取状态按 TTL 冻结保留, 手回来直接续上 ——
                # 即使那时捏点已在盒子半径之外(建立过就不要求重新建立)。
                self._grab_ttl[gone] = self._grab_ttl.get(gone, GRAB_TTL) - 1
            else:
                del self._grabbing[gone]
                self._grab_ttl.pop(gone, None)
        for k2 in keys:
            self._grab_ttl.pop(k2, None)  # 手回来了, 宽限复位
        n = sum(grabs)
        self.marks = list(zip(pts, grabs, strict=True))

        # 炸开驱动: 未**捏合**的手里最大的张开度(捏着的手在忙, 抓没抓着都不算)。
        spread = max(
            (_spread_ratio(hd) for hd, pin2 in zip(use, pins, strict=True) if not pin2),
            default=0.0,
        )
        target_ex = float(np.clip((spread - EXPLODE_LO) / (EXPLODE_HI - EXPLODE_LO), 0.0, 1.0))
        self.explode += (target_ex - self.explode) * EXPLODE_RESP

        if n >= 2:  # ---- 双手: 平移 + 缩放 ----
            self._grab = None
            pair = frozenset(keys)
            mid = (pts[0] + pts[1]) * 0.5
            dist = float(np.linalg.norm(pts[1] - pts[0]))
            if self._two is not None and self._two[0] == pair:  # 换了一双手就重新起算
                _, pmid, pdist = self._two
                step = mid - pmid
                self.pos += step  # 1:1: 两手中点走多远, 立方体走多远
                self._vel = self._vel * 0.5 + step * 0.5  # 速度估计; 松手带走成惯性
                if pdist > 1e-3:
                    self._zoom = self._zoom * 0.5 + (dist / pdist - 1.0) * 0.5
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
            self._spin *= GRIP_DAMP  # 双手握死: 残余旋转立刻锁住
            self._hold = TWO_HAND_HOLD
            self.mode = MODE_MOVE
        elif self._hold > 0:  # ---- 双手模式掉了一帧: 停住, 别掉去转动 ----
            self._hold -= 1
            self._grab = None
            self._two = None  # 手回来时重新起算基线, 免得攒出一次跳变
            self._spin *= GRIP_DAMP
            self.mode = MODE_MOVE
        elif n == 1:  # ---- 单手抓住: 拖动 = 转动 ----
            self._two = None
            self._vel *= GRIP_DAMP  # 被捏住: 平移/缩放动量被手吸收
            self._zoom *= GRIP_DAMP
            i = grabs.index(True)
            p, k = pts[i], keys[i]
            if self._grab is not None and self._grab[0] == k:  # 换手就重新起算
                d = p - self._grab[1]
                mag = float(np.linalg.norm(d))
                if mag > DRAG_MAX_PX:  # 这么大只可能是检测跳变, 不是真手
                    d = d * (DRAG_MAX_PX / mag)
                # 横拖 → 绕相机竖轴(y); 竖拖 → 绕相机横轴(x)。屏幕 y 朝下,
                # 所以往下拖时立方体上缘朝自己转过来, 手感与真实抓握一致。
                target = np.array(
                    [d[1] * self.orbit_gain, d[0] * self.orbit_gain, 0.0], np.float32
                )
                t = float(np.linalg.norm(target))
                if t > TURN_MAX_RAD:  # 目标越过对称周期 = 面随机换, 顶住
                    target *= TURN_MAX_RAD / t
                # 粘性耦合(力矩 ∝ 手速与 ω 的差): ω 追手速, 不等于手速。
                self._spin += (target - self._spin) * TURN_RESP
            self._grab = (k, p.copy())
            self.mode = MODE_TURN
        else:  # ---- 松手: 惯性 + 极慢自转 ----
            self._grab = None
            self._two = None
            self._spin *= SPIN_DAMP
            sn = float(np.linalg.norm(self._spin))
            if sn > CARRY_MAX_RAD:  # 松手滑一下可以, 乱飞不行
                self._spin *= CARRY_MAX_RAD / sn
            self._spin[1] += self.drift * (1.0 - SPIN_DAMP)  # 稳态收敛到 drift
            # 平移动量: 滑行 + 碰壁反弹(越界分量反号×BOUNCE, 位置钳回画面)
            v = float(np.linalg.norm(self._vel))
            if v > MOVE_CARRY_MAX:  # 误检瞬移的速度不带走
                self._vel *= MOVE_CARRY_MAX / v
                v = MOVE_CARRY_MAX
            if v > 0.05:
                self.pos += self._vel
                m = self.size * 0.5
                for ax, limit in ((0, w), (1, h)):
                    if self.pos[ax] < m or self.pos[ax] > limit - m:
                        self._vel[ax] = -self._vel[ax] * BOUNCE
                self.pos = np.clip(self.pos, (m, m), (w - m, h - m)).astype(np.float32)
                self._vel *= MOVE_DAMP
            # 缩放动量: 同款但短命 —— 尺寸是指数量, 拖长会滚雪球
            z = float(np.clip(self._zoom, -ZOOM_CARRY_MAX, ZOOM_CARRY_MAX))
            if abs(z) > 1e-4:
                self.size = float(
                    np.clip(
                        self.size * (1.0 + z),
                        CUBE_SIZE_MIN * min(h, w),
                        CUBE_SIZE_MAX * min(h, w),
                    )
                )
                self._zoom *= ZOOM_DAMP
            self.mode = MODE_IDLE

        ang = float(np.linalg.norm(self._spin))
        if ang > 1e-6:
            self.rot = _rot(self._spin / ang, ang) @ self.rot
            # 累积浮点误差会让 rot 慢慢不正交(立方体会被剪切), 用 SVD 拉回来
            u, _s, vt = np.linalg.svd(self.rot)
            self.rot = (u @ vt).astype(np.float32)
        # HUD 用 cv2.putText(Hershey 字体), **只认 ASCII** —— 中文会画成 "??"。
        # glassbox.debug 一直守着这个约定, 这里曾破戒("边长/捏住"上屏全是问号)。
        self.debug = f"size{self.size:4.0f} grip{n} open{spread:.2f}"

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
