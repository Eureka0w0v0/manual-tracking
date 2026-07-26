"""端到端验证器: 真实 _orient 序列 → 完整滤波 → 渲染 → 量可见面积/切换率.

    ./run.sh 之外单独跑: PYTHONPATH=src .venv/bin/python tools/e2e_check.py

改完 screen 的几何/映射/滤波后跑一遍, 对照 docs/GLASS_BOX_GEOMETRY.md §3.6
里记的基线数字。它是**唯一**能发现"可见面每秒切换 5 次"这类体感灾难的手段——
纯 ψ 扫描是无噪的, 看不到真实信号驱动下的频闪; 历史上 BOX_ROLL_GAIN 2.0→3.0
那次退步(切换 3.37→5.29 次/秒)就是靠它才暴露出来的。

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
import manual_tracking.renderer as R  # noqa: E402
from manual_tracking.renderer import VectorOverlayRenderer  # noqa: E402
from manual_tracking.tracker import HandTracker  # noqa: E402

TAGS = ("前", "背", "底", "顶", "左", "右")


def load(path="assets/sample.mp4"):
    c = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fr.append(f)
    tk = HandTracker(model_path="models/hand_landmarker.task", filter_on=True, infer_max_side=0)
    cache = [tk.process_bgr(f, k, int(k * 1000 / 30)) for k, f in enumerate(fr)]
    tk.close()
    return fr, cache


def face_areas(scr, cam, focal):
    """返回 {面名: 屏幕面积 px}, 只含可见面."""
    eye = np.array([0, 0, focal], np.float32)
    ctr = cam.mean(0)
    out = {}
    for fi, (idx, _) in enumerate(R._BOX_FACES):
        q = cam[list(idx)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        fc = q.mean(0)
        if float(n @ (fc - ctr)) < 0:
            n = -n
        if float(n @ (eye - fc)) > 0:
            p = scr[list(idx)]
            a = 0.5 * abs(
                float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
            )
            out[TAGS[fi]] = a
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
        geo = r._box_geometry(left, right)
        if geo is None:
            continue
        scr, cam, focal, _ = geo
        ar = face_areas(scr, cam, focal)
        tot = sum(ar.values()) or 1.0
        psis.append(np.degrees(r._box_ema[1]))
        combos.append("+".join(sorted(ar)))
        reds.append(ar.get("背", 0.0) / tot)
        tops.append(ar.get("顶", 0.0) / tot)
    psi = np.array(psis)
    red = np.array(reds)
    top = np.array(tops)
    n = len(psi)
    sw = sum(1 for a, b in zip(combos, combos[1:]) if a != b)
    d = np.abs(np.diff(psi))
    print(
        f"{label:26} n={n:3d} | 见背 {100 * (red > 0.01).mean():5.1f}% "
        f"背≥25% {100 * (red > 0.25).mean():5.1f}% 背≥50% {100 * (red > 0.5).mean():5.1f}% "
        f"| 背均 {100 * red.mean():4.1f}% 顶均 {100 * top.mean():4.1f}% "
        f"| 切换 {sw / (n / fps):4.2f}/s | |Δψ| p95 {np.percentile(d, 95):5.1f}°"
    )
    return dict(n=n, red=red, sw=sw / (n / fps), psi=psi)


if __name__ == "__main__":
    fr, cache = load()
    print(f"素材 {len(fr)} 帧, 双手帧 {sum(1 for c in cache if len(c.hands) >= 2)}\n")
    run(fr, cache, "默认")
    for e in (1.0, 1.5, 2.0):
        run(fr, cache, f"expo={e}", lambda r, e=e: setattr(r, "roll_expo", e))
    for c in (10.0, 20.0, 40.0):
        run(fr, cache, f"cap={c:.0f}°/帧", lambda r, c=c: setattr(r, "roll_max_rate", c))
