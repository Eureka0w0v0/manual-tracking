"""floatcube 行为自检 —— 用合成手势喂 FloatCube, 断言几条手感底线.

为什么要有这个: 玻璃盒的翻转驱动是 cos 型, 一直往一个方向翻会自己转回来
(docs/GLASS_BOX_GEOMETRY.md §6)。悬浮立方体换成拖动增量就是为了根治它,
所以"连续拖能无限翻"必须有可复现的证据, 不能靠肉眼看.

用法: PYTHONPATH=src python tools/cube_check.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from manual_tracking.effects import BOX_FACES  # noqa: E402
from manual_tracking.floatcube import (  # noqa: E402
    CUBE_SIZE_MAX,
    CUBE_SIZE_MIN,
    DRAG_MAX_PX,
    ORBIT_GAIN,
    FloatCube,
)
from manual_tracking.landmarks import (  # noqa: E402
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    THUMB_TIP,
    WRIST,
)
from manual_tracking.tracker import HandPose  # noqa: E402

SHAPE = (720, 1280)
PALM = 160.0


# 掌心用的是 PALM_RING 六个点的平均, 所以合成手必须把这六个点全设上 ——
# 少设一个, 那个 (0,0) 就会把掌心往画面左上角拽, 位移被稀释, 测出来的
# "跟手程度"是假的。
_PALM_LAYOUT = {  # landmark → 相对手中心的偏移(掌宽的倍数)
    WRIST: (0.0, 0.9),
    THUMB_CMC: (-0.55, 0.7),
    INDEX_MCP: (-0.5, 0.0),
    MIDDLE_MCP: (-0.15, -0.05),
    RING_MCP: (0.18, 0.0),
    PINKY_MCP: (0.5, 0.1),
}


def hand(cx: float, cy: float, *, pinch: bool, tid: int = 0) -> HandPose:
    """一只合成手: 整体刚性平移, 捏合时拇指尖贴到食指尖, 松开时拉开 0.9 掌宽."""
    p = np.zeros((21, 3), np.float32)
    for lm, (fx, fy) in _PALM_LAYOUT.items():
        p[lm] = (cx + PALM * fx, cy + PALM * fy, 0)
    p[INDEX_TIP] = (cx, cy - PALM * 0.6, 0)
    off = 0.0 if pinch else PALM * 0.9
    p[THUMB_TIP] = (cx - off, cy - PALM * 0.6, 0)
    return HandPose(handedness="Right", score=1.0, points=p, track_id=tid)


def pair2(cx: float, half: float = 200.0, cy: float = 400.0) -> list[HandPose]:
    """一对都捏住的手, 中点在 cx, 相距 2*half."""
    return [hand(cx - half, cy, pinch=True, tid=0), hand(cx + half, cy, pinch=True, tid=1)]


def visible(cube: FloatCube) -> str:
    """当前朝向镜头的面, 面积最大的那个."""
    scr, cam, focal = cube.project(SHAPE)
    eye = np.array([0, 0, focal], np.float32)
    ctr = cam.mean(0)
    best, best_a = "-", 0.0
    for face in BOX_FACES:
        q = cam[list(face.verts)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        fc = q.mean(0)
        if float(n @ (fc - ctr)) < 0:
            n = -n
        if float(n @ (eye - fc)) <= 0:
            continue
        s = scr[list(face.verts)]
        x, y = s[:, 0], s[:, 1]  # 鞋带公式(numpy 2 没有 2D cross)
        a = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5
        if a > best_a:
            best, best_a = face.tag, a
    return best


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'✓' if ok else '✗'} {name}: {detail}")
    return ok


def main() -> int:
    good = True

    # 1) 一个方向连续拖 —— 面必须一轮一轮循环, 不能翻到某个面就回头。
    #    横拖只绕竖轴转, 走的是四个侧面; 竖拖绕横轴, 走前/顶/背/底。两条各测一遍。
    for axis, expect in (("横拖", {"前", "左", "背", "右"}), ("竖拖", {"前", "顶", "背", "底"})):
        c = FloatCube()
        seq = []
        for i in range(300):
            d = 200.0 + (i * 9) % 900  # 拖到边就抬手回起点(_grab 断了, 不算增量)
            c.update([hand(d, 400, pinch=True) if axis == "横拖" else hand(640, d, pinch=True)], SHAPE)
            t = visible(c)
            if not seq or seq[-1] != t:
                seq.append(t)
        laps = sum(1 for a, b in zip(seq, seq[1:]) if a == seq[0] and b == seq[1])
        good &= check(
            f"{axis}能一直翻不回头",
            set(seq) == expect and laps >= 3,
            f"经过 {sorted(set(seq))}, 循环 {laps} 轮, 前 6 段 {seq[:6]}",
        )

    # 2) 长时间跑下来旋转矩阵要保持正交(否则立方体会被剪切成平行六面体)
    c = FloatCube()
    for i in range(10000):
        c.update([hand(300 + (i * 7) % 600, 400, pinch=True)], SHAPE)
    err = float(np.abs(c.rot.T @ c.rot - np.eye(3)).max())
    good &= check("旋转矩阵 10000 帧后仍正交", err < 1e-4, f"max|RᵀR−I| = {err:.2e}")

    # 3) 捏合滞回: 在阈值附近来回抖, 不能反复误触
    c = FloatCube()
    flips = 0
    prev = False
    for i in range(200):
        r = 0.52 + 0.03 * np.sin(i * 0.7)  # 卡在 PINCH_ON(.42)~PINCH_OFF(.62) 之间
        hd = hand(500, 400, pinch=True, tid=3)
        hd.points[THUMB_TIP] = (500 - PALM * r, 400 - PALM * 0.6, 0)
        c.update([hd], SHAPE)
        now = c._pinch.get(3, False)
        flips += now != prev
        prev = now
    good &= check("捏合滞回不抖", flips <= 1, f"200 帧里状态翻转 {flips} 次")

    # 4) 双手: 拉开=放大 / 收拢=缩小, 且都停在限幅内
    c = FloatCube()
    for i in range(120):
        d = 100 + i * 8
        c.update([hand(640 - d, 400, pinch=True, tid=0), hand(640 + d, 400, pinch=True, tid=1)], SHAPE)
    big = c.size
    for i in range(200):
        d = max(30, 1060 - i * 8)
        c.update([hand(640 - d, 400, pinch=True, tid=0), hand(640 + d, 400, pinch=True, tid=1)], SHAPE)
    small = c.size
    lo, hi = CUBE_SIZE_MIN * min(SHAPE), CUBE_SIZE_MAX * min(SHAPE)
    good &= check(
        "双手缩放跟手且不越界",
        big > small and lo - 1e-3 <= small and big <= hi + 1e-3,
        f"拉开到 {big:.0f}px, 收拢到 {small:.0f}px, 允许 [{lo:.0f}, {hi:.0f}]",
    )

    # 5) 平移/缩放必须 1:1 —— 手走多远它走多远, 不许打折(哥哥嫌慢的就是这个)
    c = FloatCube()
    c.update(pair2(500), SHAPE)  # 建立基线
    p0 = c.pos.copy()
    for i in range(1, 31):
        c.update(pair2(500 + i * 10), SHAPE)  # 两手一起右移 300px
    moved = float(c.pos[0] - p0[0])
    c2 = FloatCube()
    c2.update(pair2(640, half=150), SHAPE)
    s0 = c2.size
    c2.update(pair2(640, half=300), SHAPE)  # 两手拉开一倍
    good &= check(
        "平移/缩放 1:1 跟手",
        abs(moved - 300) < 1.0 and abs(c2.size / s0 - 2.0) < 0.01,
        f"手走 300px 立方体走 {moved:.1f}px; 两手拉开 2.0x 边长变 {c2.size/s0:.2f}x",
    )

    # 6) 双手平移: 立方体不会被推出画面
    c = FloatCube()
    for i in range(300):
        cx = 640 + i * 20
        c.update([hand(cx - 200, 400, pinch=True, tid=0), hand(cx + 200, 400, pinch=True, tid=1)], SHAPE)
    inside = 0 <= c.pos[0] <= SHAPE[1] and 0 <= c.pos[1] <= SHAPE[0]
    good &= check("平移被夹在画面内", inside, f"pos = ({c.pos[0]:.0f}, {c.pos[1]:.0f})")

    # 7) 两只手在 hands[] 里对调顺序, 不该产生假的拖动增量
    c = FloatCube()
    a, b = hand(400, 400, pinch=True, tid=0), hand(900, 400, pinch=False, tid=1)
    for i in range(10):
        c.update([a, b] if i % 2 == 0 else [b, a], SHAPE)
    good &= check("手序对调不炸", float(np.linalg.norm(c._spin)) < 1e-6, f"|spin| = {float(np.linalg.norm(c._spin)):.2e}")

    # 8) 限幅只该挡检测跳变, 不该误伤真实手速(实测真手 p99 = 109 px/帧)
    c = FloatCube()
    c.update([hand(100, 400, pinch=True)], SHAPE)
    c.update([hand(1100, 400, pinch=True)], SHAPE)  # 一帧瞬移 1000px = 误检
    capped = float(np.linalg.norm(c._spin))
    c2 = FloatCube()
    c2.update([hand(400, 400, pinch=True)], SHAPE)
    c2.update([hand(509, 400, pinch=True)], SHAPE)  # 109px = 真手最快的那 1%
    real = float(np.linalg.norm(c2._spin))
    good &= check(
        "限幅挡跳变但不误伤真手",
        abs(capped - DRAG_MAX_PX * ORBIT_GAIN) < 1e-6 and abs(real - 109 * ORBIT_GAIN) < 1e-4,
        f"1000px 跳变被截到 {capped/ORBIT_GAIN:.0f}px, 109px 真手速原样通过({real/ORBIT_GAIN:.0f}px)",
    )

    # 9) 双手模式掉一帧(两手捏在一起会互相遮挡): 立方体该停住, 不该切去转动
    c = FloatCube()
    two = lambda cx: [hand(cx - 200, 400, pinch=True, tid=0), hand(cx + 200, 400, pinch=True, tid=1)]
    for i in range(20):
        c.update(two(400 + i * 6), SHAPE)
    rot_before, pos_before = c.rot.copy(), c.pos.copy()
    c.update([hand(600, 400, pinch=True, tid=0)], SHAPE)  # 右手这帧没测到
    turned = float(np.degrees(np.arccos(np.clip((np.trace(rot_before.T @ c.rot) - 1) / 2, -1, 1))))
    good &= check(
        "双手掉一帧不乱转",
        turned < 1.0 and c.mode == "move",
        f"掉帧后转了 {turned:.2f}° (宽限前 {np.linalg.norm(pos_before - c.pos):.1f}px 位移), 模式仍是 {c.mode}",
    )

    # 10) 松手后的自转: 要慢到不晕但看得出是活的
    c = FloatCube()
    for _ in range(60):
        c.update([], SHAPE)
    deg = float(np.linalg.norm(c._spin)) * np.degrees(1.0) * 30.0
    good &= check("松手自转速度合理", 3.0 <= deg <= 40.0, f"{deg:.1f} °/秒 (30fps)")

    print("\n" + ("全部通过" if good else "有不通过项"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
