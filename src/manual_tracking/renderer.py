"""Hand overlay renderer.

Hands: bone skeleton ONLY (no palm plates / aura fills).
Nothing is drawn between the two hands.
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import (
    CONNECTIONS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PALM_RING,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)
from .tracker import FrameHands, HandPose

# BGR — orange / gold energy
GOLD = (40, 170, 255)
ORANGE = (20, 110, 240)
AMBER = (60, 200, 255)
DEEP = (10, 40, 90)
WHITE_HOT = (230, 250, 255)
SOFT_ORANGE = (30, 90, 200)

TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_PALM_IDX = np.array(PALM_RING, dtype=np.int32)


def _palm_center(hand: HandPose) -> np.ndarray:
    return hand.points[_PALM_IDX, :2].mean(axis=0).astype(np.float32)


def _palm_scale(hand: HandPose) -> float:
    p = hand.points
    a = p[INDEX_MCP, :2]
    b = p[PINKY_MCP, :2]
    c = p[WRIST, :2]
    return float(max(np.linalg.norm(a - b), np.linalg.norm((a + b) * 0.5 - c), 20.0))


def _blend_poly(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    outline: tuple[int, int, int] | None = None,
    outline_w: int = 1,
) -> None:
    """Alpha-fill polygon on bbox only (quality translucent faces)."""
    if len(pts) < 3 or alpha <= 0:
        return
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    h, w = canvas.shape[:2]
    xs, ys = poly[:, 0, 0], poly[:, 0, 1]
    x0, x1 = max(0, int(xs.min()) - 2), min(w, int(xs.max()) + 3)
    y0, y1 = max(0, int(ys.min()) - 2), min(h, int(ys.max()) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local = poly.copy()
    local[:, 0, 0] -= x0
    local[:, 0, 1] -= y0
    cv2.fillPoly(mask, [local], 255, lineType=cv2.LINE_AA)
    roi = canvas[y0:y1, x0:x1]
    a = float(np.clip(alpha, 0.0, 1.0))
    m = (mask > 0)[:, :, None].astype(np.float32)
    col = np.array(color, np.float32)
    roi[:] = (roi.astype(np.float32) * (1.0 - a * m) + col * (a * m)).astype(np.uint8)
    if outline is not None and outline_w > 0:
        cv2.polylines(canvas, [poly.reshape(-1, 2)], True, outline, outline_w, cv2.LINE_AA)


class VectorOverlayRenderer:
    """
    styles:
      fabric — skeleton only (between-hand energy removed)
      track  — skeleton + few tip faces
      wire   — skeleton only
    """

    def __init__(
        self,
        style: str = "fabric",
        *,
        show_source: bool = True,
        source_dim: float = 0.65,
        fast: bool = False,
        vignette: bool | None = None,
        effect: str = "energy",
    ) -> None:
        if style in ("frame", "planes", "fluid"):
            style = "fabric"
        if style == "outline":
            style = "wire"
        self.style = style if style in ("fabric", "track", "wire") else "fabric"
        self.show_source = show_source
        self.source_dim = float(source_dim)
        self.fast = bool(fast)
        self.vignette = False
        self.effect = effect if effect in ("energy", "calm", "hot") else "energy"
        self._pull_ema = 0.0

    def reset(self) -> None:
        self._pull_ema = 0.0

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
        if self.style == "wire":
            for i, h in enumerate(hands):
                self._skeleton(out, h, i)
            return out
        if self.style == "track":
            self._draw_track(out, hands)
            return out

        self._draw_fabric(out, hands)
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

    def _pull_state(
        self, hands: list[HandPose]
    ) -> tuple[float, float, HandPose, HandPose] | None:
        if len(hands) < 2:
            self._pull_ema *= 0.85
            return None
        left, right = sorted(hands[:2], key=lambda h: float(_palm_center(h)[0]))
        dist = float(np.linalg.norm(_palm_center(right) - _palm_center(left)))
        scale = 0.5 * (_palm_scale(left) + _palm_scale(right))
        pull = float(np.clip((dist - scale * 0.6) / max(scale * 2.4, 1.0), 0.0, 1.0))
        self._pull_ema = pull if self._pull_ema < 1e-6 else self._pull_ema * 0.7 + pull * 0.3
        return self._pull_ema, scale, left, right

    def _draw_fabric(self, canvas: np.ndarray, hands: list[HandPose]) -> None:
        """Skeleton only — between-hand energy removed by request."""
        for i, h in enumerate(hands):
            self._skeleton(canvas, h, i)

    def _draw_track(self, canvas: np.ndarray, hands: list[HandPose]) -> None:
        got = self._pull_state(hands)
        if got is not None:
            pull, scale, left, right = got
            if pull > 0.08:
                for tip, col in (
                    (INDEX_TIP, (245, 245, 250)),
                    (MIDDLE_TIP, GOLD),
                    (RING_TIP, (200, 200, 205)),
                ):
                    p0 = left.points[tip, :2].astype(np.float32)
                    p1 = right.points[tip, :2].astype(np.float32)
                    v = p1 - p0
                    ln = float(np.linalg.norm(v))
                    if ln < scale * 0.35:
                        continue
                    n = np.array([-v[1], v[0]], np.float32) / (ln + 1e-6)
                    thick = scale * (0.08 + 0.18 * pull)
                    mid = (p0 + p1) * 0.5
                    dia = np.array([p0, mid + n * thick, p1, mid - n * thick], np.float32)
                    _blend_poly(canvas, dia, col, 0.28 + 0.4 * pull, outline=DEEP, outline_w=1)
        for i, h in enumerate(hands):
            self._skeleton(canvas, h, i)
