"""Hand overlay renderer.

Hands: bone skeleton.
mirror/screen: opaque plate pinned between the two hands' thumb+index tips,
textured with a processed copy of the camera frame (抖音 manualtracking 的
"指间玻璃板"效果):
  mirror — 泛白镜面倒影板(视频1前半风格)
  screen — 黄红横幅夹负片实时画面(TouchDesigner 视频风格)
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import (
    CONNECTIONS,
    INDEX_TIP,
    MIDDLE_TIP,
    PALM_RING,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
)
from .tracker import FrameHands, HandPose

# BGR
GOLD = (40, 170, 255)
ORANGE = (20, 110, 240)
WHITE_HOT = (230, 250, 255)
EDGE = (245, 248, 250)  # plate outline
MIRROR_SIDE = (150, 158, 165)  # plate thickness face (mirror)
SCREEN_SIDE = (24, 18, 140)  # plate thickness face (screen)
BAND_YELLOW = (0, 205, 255)
BAND_RED = (30, 20, 230)

TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_PALM_IDX = np.array(PALM_RING, dtype=np.int32)

GLASS_WHITE = (250, 250, 252)

_STYLE_ALIASES = {
    "fabric": "mirror",
    "frame": "mirror",
    "planes": "mirror",
    "fluid": "mirror",
    "track": "screen",
    "outline": "wire",
}


def _palm_center(hand: HandPose) -> np.ndarray:
    return hand.points[_PALM_IDX, :2].mean(axis=0).astype(np.float32)


def _alpha_fill(canvas: np.ndarray, poly: np.ndarray, color: tuple[int, int, int], a: float) -> None:
    """Translucent convex-poly fill, bbox-local."""
    h, w = canvas.shape[:2]
    p = np.round(poly).astype(np.int32)
    x0, y0 = max(0, p[:, 0].min() - 1), max(0, p[:, 1].min() - 1)
    x1, y1 = min(w, p[:, 0].max() + 2), min(h, p[:, 1].max() + 2)
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.fillConvexPoly(overlay, p - (x0, y0), color, cv2.LINE_AA)
    cv2.addWeighted(overlay, a, roi, 1.0 - a, 0, dst=roi)


class VectorOverlayRenderer:
    """
    styles:
      mirror — skeleton + 指间镜面板
      screen — skeleton + 横幅屏幕板
      wire   — skeleton only
    """

    def __init__(
        self,
        style: str = "mirror",
        *,
        show_source: bool = True,
        source_dim: float = 0.65,
        fast: bool = False,
        vignette: bool | None = None,
        effect: str = "energy",
    ) -> None:
        style = _STYLE_ALIASES.get(style, style)
        self.style = style if style in ("mirror", "screen", "wire") else "mirror"
        self.show_source = show_source
        self.source_dim = float(source_dim)
        self.fast = bool(fast)
        self.vignette = False
        self.effect = effect if effect in ("energy", "calm", "hot") else "energy"

    def reset(self) -> None:
        pass

    def next_effect(self) -> str:
        order = ["energy", "calm", "hot"]
        i = order.index(self.effect) if self.effect in order else 0
        self.effect = order[(i + 1) % len(order)]
        return self.effect

    def render(self, frame_bgr: np.ndarray, frame_hands: FrameHands) -> np.ndarray:
        if self.show_source:
            if self.source_dim >= 0.98:
                out = frame_bgr.copy()
            else:
                out = cv2.convertScaleAbs(frame_bgr, alpha=self.source_dim, beta=0)
        else:
            out = np.empty_like(frame_bgr)
            out[:] = (8, 6, 12)

        hands = frame_hands.hands
        if self.style in ("mirror", "screen") and len(hands) >= 2:
            quad = self._plate_quad(hands)
            if quad is not None:
                self._draw_plate(out, quad)

        for i, h in enumerate(hands):
            self._skeleton(out, h, i)
        return out

    def _skeleton(self, canvas: np.ndarray, hand: HandPose, index: int = 0) -> None:
        """Bone lines + joints only. No filled palm graphics."""
        color = GOLD if index == 0 else ORANGE
        pts = hand.as_int()
        for a, b in CONNECTIONS:
            p0 = (int(pts[a, 0]), int(pts[a, 1]))
            p1 = (int(pts[b, 0]), int(pts[b, 1]))
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
        for i in range(21):
            x, y = int(pts[i, 0]), int(pts[i, 1])
            r = 4 if i in TIP_IDS else 3
            cv2.circle(canvas, (x, y), r, WHITE_HOT, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), r, color, 1, cv2.LINE_AA)

    def _plate_quad(self, hands: list[HandPose]) -> np.ndarray | None:
        """4 corners: L index, R index, R thumb, L thumb (tl, tr, br, bl)."""
        left, right = sorted(hands[:2], key=lambda h: float(_palm_center(h)[0]))
        quad = np.array(
            [
                left.points[INDEX_TIP, :2],
                right.points[INDEX_TIP, :2],
                right.points[THUMB_TIP, :2],
                left.points[THUMB_TIP, :2],
            ],
            np.float32,
        )
        # fingers pinched shut -> plate collapses and disappears
        if abs(cv2.contourArea(quad.reshape(-1, 1, 2))) < 30.0:
            return None
        return quad

    def _draw_plate(self, canvas: np.ndarray, quad: np.ndarray) -> None:
        """Pure translucent glass — no image pasted in, background shows through."""
        k = {"calm": 0.8, "energy": 1.0, "hot": 1.2}[self.effect]

        # thickness: translucent slab side under the bottom edge
        top_len = float(np.linalg.norm(quad[1] - quad[0]))
        d = float(np.clip(top_len * 0.05, 3.0, 14.0))
        side = np.array(
            [quad[3], quad[2], quad[2] + [0.0, d], quad[3] + [0.0, d]],
            np.float32,
        )
        side_col = MIRROR_SIDE if self.style == "mirror" else SCREEN_SIDE
        _alpha_fill(canvas, side, side_col, 0.5)

        tl, tr, br, bl = quad

        def _lerp_row(t: float) -> tuple[np.ndarray, np.ndarray]:
            return tl + (bl - tl) * t, tr + (br - tr) * t

        if self.style == "mirror":
            # white glass sheet + stronger sheen on the top third
            _alpha_fill(canvas, quad, GLASS_WHITE, min(0.9, 0.45 * k))
            l3, r3 = _lerp_row(0.35)
            _alpha_fill(canvas, np.array([tl, tr, r3, l3], np.float32), WHITE_HOT, 0.25)
        else:
            # translucent banner: yellow / clear glass / red
            l1, r1 = _lerp_row(0.16)
            l2, r2 = _lerp_row(0.84)
            _alpha_fill(canvas, np.array([tl, tr, r1, l1], np.float32), BAND_YELLOW, min(0.9, 0.7 * k))
            _alpha_fill(canvas, np.array([l1, r1, r2, l2], np.float32), GLASS_WHITE, min(0.9, 0.3 * k))
            _alpha_fill(canvas, np.array([l2, r2, br, bl], np.float32), BAND_RED, min(0.9, 0.7 * k))

        cv2.polylines(canvas, [np.round(quad).astype(np.int32)], True, EDGE, 2, cv2.LINE_AA)
