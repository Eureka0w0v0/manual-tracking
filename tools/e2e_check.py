"""端到端验证器: 真实 _orient 序列 → 完整滤波 → 渲染 → 量可见面积/切换率.

    PYTHONPATH=src .venv/bin/python tools/e2e_check.py           # 断言当前默认档
    PYTHONPATH=src .venv/bin/python tools/e2e_check.py --sweep   # 扫 expo/cap 找参数

改完 screen 的几何/映射/滤波后跑一遍。不通过 = 手感退步, 退出码非 0。
它是**唯一**能发现"可见面每秒切换 5 次"这类体感灾难的手段——纯 ψ 扫描是无噪的,
看不到真实信号驱动下的频闪; 历史上 BOX_ROLL_GAIN 2.0→3.0 那次退步
(切换 3.37→5.29 次/秒)就是靠它才暴露出来的。

指标含义:
  见背/背≥25%/背≥50%  背面投影面积占可见总面积的比例超过阈值的帧占比。
                       "背≥50%" 才代表"真的看得见后面", 只看"见背"会被
                       一条 5% 宽的缝骗过去。
  切换                 可见面组合每秒变化次数。越低越稳, 高于 ~5 就是频闪。
  |Δψ| p95             ψ 的单帧变化, 受 BOX_ROLL_MAX_RATE 限幅约束。
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from manual_tracking.effects import BOX_FACES  # noqa: E402
from manual_tracking.paths import ensure_model  # noqa: E402
from manual_tracking.renderer import VectorOverlayRenderer  # noqa: E402
from manual_tracking.tracker import HandTracker  # noqa: E402

# ---- 手感底线(数字见 docs/GLASS_BOX_GEOMETRY.md §3.6 的 expo×cap 网格) ----
# 卡的是"越过体感红线", 不是"跟上次一模一样": 后两条留了余量, 正常调参不会
# 误报。切换率上限就是 5.0 本身 —— 当前默认档 4.87 只剩 0.13 的余量, 那是
# 有意的, 这条线越过去肉眼直接看得见颜色在闪。
MAX_SWITCH_PER_SEC = 5.0  # 可见面组合每秒变化次数
MIN_BACK_HALF = 0.30  # "背面占可见面积一半以上"的帧占比下限(当前 36.4%)
MIN_BACK_SEEN = 0.45  # "看得见背面"的帧占比下限(当前 54.5%)


def load(path="assets/sample.mp4"):
    c = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fr.append(f)
    tk = HandTracker(model_path=ensure_model(), filter_on=True, infer_max_side=0)
    cache = [tk.process_bgr(f, k, int(k * 1000 / 30)) for k, f in enumerate(fr)]
    tk.close()
    return fr, cache


def face_areas(scr, cam, focal):
    """返回 {面名: 屏幕面积 px}, 只含可见面."""
    eye = np.array([0, 0, focal], np.float32)
    ctr = cam.mean(0)
    out = {}
    for face in BOX_FACES:
        q = cam[list(face.verts)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        fc = q.mean(0)
        if float(n @ (fc - ctr)) < 0:
            n = -n
        if float(n @ (eye - fc)) > 0:
            p = scr[list(face.verts)]
            a = 0.5 * abs(
                float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
            )
            out[face.tag] = a
    return out


def run(fr, cache, label, setup=None, fps=30.0):
    r = VectorOverlayRenderer(style="screen")
    if setup:
        setup(r)
    psis, combos, reds, tops = [], [], [], []
    for fh in cache:
        if len(fh.hands) < 2:
            continue
        left, right = r._ordered(fh.hands)
        geo = r.box.solve(left, right)
        if geo is None:
            continue
        if r.box._ease < 0.999:
            # 收起/出现的过渡帧: 盒子在压扁/长出, 面组合本来就会变 ——
            # 那是刻意的动画, 不是 ψ 频闪。状态已推进(solve 调过), 只跳指标。
            continue
        scr, cam, focal, _ = geo
        ar = face_areas(scr, cam, focal)
        tot = sum(ar.values()) or 1.0
        psis.append(np.degrees(r.box._ema[1]))
        combos.append("+".join(sorted(ar)))
        reds.append(ar.get("背", 0.0) / tot)
        tops.append(ar.get("顶", 0.0) / tot)
    psi = np.array(psis)
    red = np.array(reds)
    top = np.array(tops)
    n = len(psi)
    sw = sum(1 for a, b in zip(combos, combos[1:], strict=False) if a != b)
    d = np.abs(np.diff(psi))
    print(
        f"{label:26} n={n:3d} | 见背 {100 * (red > 0.01).mean():5.1f}% "
        f"背≥25% {100 * (red > 0.25).mean():5.1f}% 背≥50% {100 * (red > 0.5).mean():5.1f}% "
        f"| 背均 {100 * red.mean():4.1f}% 顶均 {100 * top.mean():4.1f}% "
        f"| 切换 {sw / (n / fps):4.2f}/s | |Δψ| p95 {np.percentile(d, 95):5.1f}°"
    )
    return dict(n=n, red=red, sw=sw / (n / fps), psi=psi)


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'✓' if ok else '✗'} {name}: {detail}")
    return ok


def main() -> int:
    sweep = "--sweep" in sys.argv
    fr, cache = load()
    print(f"素材 {len(fr)} 帧, 双手帧 {sum(1 for c in cache if len(c.hands) >= 2)}\n")
    base = run(fr, cache, "默认")

    if sweep:
        for e in (1.5, 2.0, 2.5):
            run(fr, cache, f"expo={e}", lambda r, e=e: setattr(r.box, "roll_expo", e))
        for c in (8.0, 10.0, 15.0, 20.0):
            run(fr, cache, f"cap={c:.0f}°/帧", lambda r, c=c: setattr(r.box, "roll_max_rate", c))
        return 0

    red = base["red"]
    seen = float((red > 0.01).mean())
    half = float((red > 0.5).mean())
    print()
    good = check(
        "可见面不频闪",
        base["sw"] < MAX_SWITCH_PER_SEC,
        f"{base['sw']:.2f} 次/秒 (上限 {MAX_SWITCH_PER_SEC})",
    )
    good &= check(
        "背面读得出来",
        half >= MIN_BACK_HALF,
        f"背≥50% 的帧占 {100 * half:.1f}% (下限 {100 * MIN_BACK_HALF:.0f}%)",
    )
    good &= check(
        "背面露脸够频繁",
        seen >= MIN_BACK_SEEN,
        f"见背的帧占 {100 * seen:.1f}% (下限 {100 * MIN_BACK_SEEN:.0f}%)",
    )
    print("\n全部通过" if good else "\n有退步, 见上面的 ✗")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
