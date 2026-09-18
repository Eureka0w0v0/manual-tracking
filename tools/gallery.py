"""渲染 docs/img/styles.jpg: 五种风格的示意图, 合成手 + 程序生成的背景.

    PYTHONPATH=src .venv/bin/python tools/gallery.py [输出路径]

不含任何真实素材(没有摄像头画面, 没有原片): 手是 tools/synth.py 的合成手,
背景是几块几何 + 渐变 + 噪声凑出来的"假房间" —— 只为了让六种像素处理有明暗
结构可咬。实机效果贴在你自己的手上, 比这好看得多; 这张图只回答"每种风格
长什么样"。
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tools")

from manual_tracking.floatcube import _rot  # noqa: E402
from manual_tracking.landmarks import THUMB_TIP  # noqa: E402
from manual_tracking.renderer import VectorOverlayRenderer  # noqa: E402
from manual_tracking.tracker import FrameHands, HandPose  # noqa: E402
from synth import hand  # noqa: E402

H, W = 720, 1280
TILE_W, TILE_H = 640, 360
LABEL_H = 34
COLS = 2


def background() -> np.ndarray:
    """有明暗结构的假房间: 渐变墙 + 窗户 + 圆灯 + 地板线 + 细噪声."""
    rng = np.random.default_rng(7)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    base = 70 + 110 * (1 - y) * (0.6 + 0.4 * x)  # 上亮下暗, 右侧更亮
    img = np.clip(np.stack([base * 0.95, base * 0.9, base * 0.85], axis=2), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (80, 120), (420, 520), (200, 190, 170), -1)
    cv2.rectangle(img, (110, 150), (390, 490), (235, 225, 205), -1)
    cv2.circle(img, (1050, 260), 130, (90, 80, 75), -1)
    cv2.circle(img, (1050, 260), 90, (150, 140, 130), -1)
    cv2.rectangle(img, (0, 560), (W, H), (60, 55, 50), -1)
    for i in range(0, W, 64):
        cv2.line(img, (i, 560), (i + 32, H), (85, 78, 70), 2)
    noise = rng.normal(0, 9, (H, W, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def sheet_hand(cx: float, cy: float, tid: int, handedness: str = "Right") -> HandPose:
    """mirror / banner 的角钉在食指尖 + 拇指尖: 拇指尖得在食指尖下方, 纸才有高度."""
    h = hand(cx, cy, pinch=False, tid=tid, handedness=handedness)
    h.points[THUMB_TIP] = (cx - 20.0, cy + 110.0, 0.0)
    return h


def render(bg: np.ndarray, style: str, frames: list[list[HandPose]], setup=None) -> np.ndarray:
    r = VectorOverlayRenderer(style=style, source_dim=0.55)
    if setup is not None:
        setup(r)
    out = bg
    for k, hands in enumerate(frames):
        out = r.render(bg, FrameHands(k, hands))
    return out


def pose_cube(r: VectorOverlayRenderer, size: float) -> None:
    """摆一个能同时看到三个面的姿态(自转要等很久才转到)."""
    r.cube.update([], (H, W))  # 首次放置在画面中央
    r.cube.rot = _rot(np.array([0, 1, 0], np.float32), np.radians(35)) @ _rot(
        np.array([1, 0, 0], np.float32), np.radians(-28)
    )
    r.cube.size = size


def main(out_path: str) -> None:
    bg = background()
    pair = [hand(430, 390, pinch=False, tid=0, handedness="Left"), hand(850, 390, pinch=False, tid=1)]
    sheet = [sheet_hand(430, 330, 0, "Left"), sheet_hand(850, 330, 1)]
    one = [hand(640, 400, pinch=False, tid=0)]
    open_hand = [hand(1000, 560, pinch=False, spread=2.0)]  # 五指张开 → 炸开

    tiles = [
        ("mirror", render(bg, "mirror", [sheet] * 3)),
        ("screen", render(bg, "screen", [pair] * 12)),  # 出现动画要几帧才长到全尺寸
        ("cube", render(bg, "cube", [[]] * 2, lambda r: pose_cube(r, 300.0))),
        ("cube (exploded)", render(bg, "cube", [open_hand] * 40, lambda r: pose_cube(r, 210.0))),
        ("banner", render(bg, "banner", [sheet] * 3)),
        ("wire", render(bg, "wire", [one] * 8)),
    ]

    rows = (len(tiles) + COLS - 1) // COLS
    canvas = np.full((rows * (TILE_H + LABEL_H), COLS * TILE_W, 3), (18, 16, 20), np.uint8)
    for i, (label, img) in enumerate(tiles):
        r_, c_ = divmod(i, COLS)
        y0, x0 = r_ * (TILE_H + LABEL_H), c_ * TILE_W
        canvas[y0 + LABEL_H : y0 + LABEL_H + TILE_H, x0 : x0 + TILE_W] = cv2.resize(
            img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA
        )
        # Hershey 字体只认 ASCII, 中文说明留给 README 的表
        cv2.putText(canvas, label, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out_path} {canvas.shape[1]}x{canvas.shape[0]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/img/styles.jpg")
