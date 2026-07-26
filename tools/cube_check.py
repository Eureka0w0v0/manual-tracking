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
from manual_tracking.floatcube import CUBE_SIZE_MAX, CUBE_SIZE_MIN, SPIN_MAX, FloatCube  # noqa: E402
from manual_tracking.landmarks import (  # noqa: E402
    INDEX_MCP,
    INDEX_TIP,
    PINKY_MCP,
    THUMB_TIP,
)
from manual_tracking.tracker import HandPose  # noqa: E402

SHAPE = (720, 1280)
PALM = 160.0


def hand(cx: float, cy: float, *, pinch: bool, tid: int = 0) -> HandPose:
    """一只合成手: 捏合时拇指尖贴到食指尖, 松开时拉到 0.9 掌宽外."""
    p = np.zeros((21, 3), np.float32)
    p[INDEX_MCP] = (cx - PALM * 0.5, cy, 0)
    p[PINKY_MCP] = (cx + PALM * 0.5, cy, 0)
    p[INDEX_TIP] = (cx, cy - PALM * 0.6, 0)
    off = 0.0 if pinch else PALM * 0.9
    p[THUMB_TIP] = (cx - off, cy - PALM * 0.6, 0)
    return HandPose(handedness="Right", score=1.0, points=p, track_id=tid)


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
        p = np.zeros((21, 3), np.float32)
        p[INDEX_MCP] = (500 - PALM * 0.5, 400, 0)
        p[PINKY_MCP] = (500 + PALM * 0.5, 400, 0)
        p[INDEX_TIP] = (500, 340, 0)
        p[THUMB_TIP] = (500 - PALM * r, 340, 0)
        c.update([HandPose("Right", 1.0, p, 3)], SHAPE)
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

    # 5) 双手平移: 立方体不会被推出画面
    c = FloatCube()
    for i in range(300):
        cx = 640 + i * 20
        c.update([hand(cx - 200, 400, pinch=True, tid=0), hand(cx + 200, 400, pinch=True, tid=1)], SHAPE)
    inside = 0 <= c.pos[0] <= SHAPE[1] and 0 <= c.pos[1] <= SHAPE[0]
    good &= check("平移被夹在画面内", inside, f"pos = ({c.pos[0]:.0f}, {c.pos[1]:.0f})")

    # 6) 两只手在 hands[] 里对调顺序, 不该产生假的拖动增量
    c = FloatCube()
    a, b = hand(400, 400, pinch=True, tid=0), hand(900, 400, pinch=False, tid=1)
    for i in range(10):
        c.update([a, b] if i % 2 == 0 else [b, a], SHAPE)
    good &= check("手序对调不炸", float(np.linalg.norm(c._spin)) < 1e-6, f"|spin| = {float(np.linalg.norm(c._spin)):.2e}")

    # 7) 单帧限幅: 手瞬移半个屏幕也不该把立方体甩飞
    c = FloatCube()
    c.update([hand(100, 400, pinch=True)], SHAPE)
    c.update([hand(1100, 400, pinch=True)], SHAPE)
    good &= check(
        "瞬移被限幅",
        float(np.linalg.norm(c._spin)) <= SPIN_MAX + 1e-6,
        f"|spin| = {float(np.linalg.norm(c._spin)):.3f} rad ≤ {SPIN_MAX}",
    )

    # 8) 松手后的自转: 要慢到不晕但看得出是活的
    c = FloatCube()
    for _ in range(60):
        c.update([], SHAPE)
    deg = float(np.linalg.norm(c._spin)) * np.degrees(1.0) * 30.0
    good &= check("松手自转速度合理", 3.0 <= deg <= 40.0, f"{deg:.1f} °/秒 (30fps)")

    print("\n" + ("全部通过" if good else "有不通过项"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
