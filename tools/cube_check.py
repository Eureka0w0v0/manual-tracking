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
    ORBIT_GAIN,
    TURN_MAX_RAD,
    TURN_RESP,
    FloatCube,
)
from manual_tracking.landmarks import THUMB_TIP  # noqa: E402
from synth import PALM, SHAPE, hand, pair2  # noqa: E402


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


def rot_angle(r0: np.ndarray, r1: np.ndarray) -> float:
    """两个旋转矩阵之间的夹角(度)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))))


def main() -> int:
    good = True

    # 1) 一个方向连续拖 —— 面必须一轮一轮循环, 不能翻到某个面就回头。
    #    横拖只绕竖轴转, 走的是四个侧面; 竖拖绕横轴, 走前/顶/背/底。两条各测一遍。
    for axis, expect in (("横拖", {"前", "左", "背", "右"}), ("竖拖", {"前", "顶", "背", "底"})):
        c = FloatCube()
        seq = []
        for i in range(400):
            d = 200.0 + (i * 9) % 900
            lifted = i > 0 and i % 100 == 0  # 拖到边抬手一帧回起点(增量归零, 不算瞬移)
            pin = not lifted
            c.update([hand(d, 400, pinch=pin) if axis == "横拖" else hand(640, d, pinch=pin)], SHAPE)
            t = visible(c)
            if not seq or seq[-1] != t:
                seq.append(t)
        laps = sum(1 for a, b in zip(seq, seq[1:], strict=False) if a == seq[0] and b == seq[1])
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
    spin = float(np.linalg.norm(c._spin))
    good &= check("手序对调不炸", spin < 1e-6, f"|spin| = {spin:.2e}")

    # 8) 两层限幅 × 惯性: 跳变目标被顶在 90°, 再被质量稀释成 90°×RESP 的角速度;
    #    真实手速一帧只吃 RESP 份, 持续拖 12 帧后必须收敛到全速(不然就是"拖不动")。
    c = FloatCube()
    c.update([hand(100, 400, pinch=True)], SHAPE)
    c.update([hand(1100, 400, pinch=True)], SHAPE)  # 一帧瞬移 1000px = 误检
    capped = float(np.linalg.norm(c._spin))
    c2 = FloatCube()
    for i in range(13):
        c2.update([hand(100 + i * 109, 400, pinch=True)], SHAPE)  # 持续 109px/帧 = 真手最快的 1%
    steady = float(np.linalg.norm(c2._spin))
    good &= check(
        "跳变被质量稀释, 真手持续拖能到全速",
        abs(capped - TURN_MAX_RAD * TURN_RESP) < 1e-6 and steady > 109 * ORBIT_GAIN * 0.99,
        f"1000px 瞬移 → ω 只被拉到 {np.degrees(capped):.0f}°/帧 (目标顶 90°×吸收 {TURN_RESP}), "
        f"109px 持续拖 12 帧 → ω {np.degrees(steady):.0f}°/帧 (全速 {np.degrees(109 * ORBIT_GAIN):.0f}°)",
    )

    # 9) 双手模式掉一帧(两手捏在一起会互相遮挡): 立方体该停住, 不该切去转动
    c = FloatCube()
    for i in range(20):
        c.update(pair2(400 + i * 6), SHAPE)
    rot_before, pos_before = c.rot.copy(), c.pos.copy()
    c.update([hand(600, 400, pinch=True, tid=0)], SHAPE)  # 右手这帧没测到
    turned = rot_angle(rot_before, c.rot)
    good &= check(
        "双手掉一帧不乱转",
        turned < 1.0 and c.mode == "move",
        f"掉帧后转了 {turned:.2f}° "
        f"(宽限前 {np.linalg.norm(pos_before - c.pos):.1f}px 位移), 模式仍是 {c.mode}",
    )

    # 10) 松手后的自转: 要慢到不晕但看得出是活的
    c = FloatCube()
    for _ in range(60):
        c.update([], SHAPE)
    deg = float(np.linalg.norm(c._spin)) * np.degrees(1.0) * 30.0
    good &= check("松手自转速度合理", 3.0 <= deg <= 40.0, f"{deg:.1f} °/秒 (30fps)")

    # 11) 松手滑行 = 物理惯性: 带走的是滤波后的动量, 不是收尾尖峰。
    #     悠闲拖松手滑约半个面; 猛甩收尾也只滑约一个面周期, 不再是旧参数的 300°+。
    #     对照组 = 纯 drift 同样 60 帧; 横拖惯性与 drift 都绕 y 轴, 角度可直接相减。
    ref = FloatCube()
    ref.update([], SHAPE)
    ref0 = ref.rot.copy()
    for _ in range(60):
        ref.update([], SHAPE)
    drift_deg = rot_angle(ref0, ref.rot)

    def coast_after(px_per_frame: float, frames: int) -> float:
        c = FloatCube()
        for i in range(frames):
            c.update([hand(100 + i * px_per_frame, 400, pinch=True)], SHAPE)
        r0 = c.rot.copy()
        for _ in range(60):
            c.update([], SHAPE)
        return rot_angle(r0, c.rot) - drift_deg

    lazy = coast_after(15, 10)  # 悠闲拖散手
    fling = coast_after(109, 8)  # 全速甩出去
    good &= check(
        "松手带惯性但不乱飞",
        25.0 <= lazy <= 60.0 and 70.0 <= fling <= 115.0,
        f"悠闲拖散手滑 {lazy:.0f}° (半个面), 全速甩滑 {fling:.0f}° (一个面周期; 旧参数下是 300°+)",
    )

    # 12) 捏停黏滞: 拖稳后手停住(仍捏着), 立方体要被手"黏"停 —— 5 帧内角速度掉到 10% 以下
    c = FloatCube()
    for i in range(10):
        c.update([hand(100 + i * 30, 400, pinch=True)], SHAPE)
    w0 = float(np.linalg.norm(c._spin))
    for _ in range(5):
        c.update([hand(100 + 9 * 30, 400, pinch=True)], SHAPE)  # 手定住
    w5 = float(np.linalg.norm(c._spin))
    good &= check(
        "捏停 5 帧内停稳",
        w0 > 0 and w5 < w0 * 0.10,
        f"拖稳 ω {np.degrees(w0):.1f}°/帧, 手停 5 帧后剩 {np.degrees(w5):.2f}°/帧 ({w5 / w0 * 100:.0f}%)",
    )

    # 13) 双手抓住 = 锁转: 转起来的立方体被两只手一抓, 3 帧内旋转基本锁死
    c = FloatCube()
    for i in range(10):
        c.update([hand(100 + i * 60, 400, pinch=True, tid=0)], SHAPE)
    w0 = float(np.linalg.norm(c._spin))
    for _ in range(3):
        c.update(pair2(640), SHAPE)
    w3 = float(np.linalg.norm(c._spin))
    good &= check(
        "双手抓住 3 帧锁转",
        w0 > 0 and w3 < w0 * 0.15,
        f"单手拖出 ω {np.degrees(w0):.1f}°/帧, 双手一抓 3 帧后剩 {w3 / w0 * 100:.0f}%",
    )


    print("\n" + ("全部通过" if good else "有不通过项"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
