"""Vector-style hand overlay renderer (AM manual-tracking look).

Live path avoids full-frame copies / alpha blends every stroke.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .landmarks import (
    CONNECTIONS,
    FINGERS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PALM_RING,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    STYLE_A,
    STYLE_B,
    THUMB_TIP,
    WRIST,
)
from .tracker import FrameHands, HandPose


def _finger_ribbon(hand: HandPose, indices: tuple[int, ...], width: float) -> np.ndarray:
    """Build a tapered ribbon polygon along a finger chain."""
    pts = hand.points[list(indices), :2]
    if len(pts) < 2:
        return np.zeros((0, 2), dtype=np.float32)

    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        norm = float(np.linalg.norm(d))
        if norm < 1e-3:
            normal = np.array([0.0, 1.0], dtype=np.float32)
        else:
            d = d / norm
            normal = np.array([-d[1], d[0]], dtype=np.float32)
        t = 1.0 - (i / max(n - 1, 1)) * 0.65
        half = width * t * 0.5
        left.append(pts[i] + normal * half)
        right.append(pts[i] - normal * half)
    return np.vstack([left, right[::-1]]).astype(np.float32)


def _palm_scale(hand: HandPose) -> float:
    p = hand.points
    a = p[INDEX_MCP, :2]
    b = p[PINKY_MCP, :2]
    c = p[WRIST, :2]
    return float(max(np.linalg.norm(a - b), np.linalg.norm((a + b) * 0.5 - c), 20.0))


def _style_for(hand: HandPose, index: int) -> dict:
    if hand.handedness == "Left":
        return STYLE_B
    if hand.handedness == "Right":
        return STYLE_A
    return STYLE_A if index % 2 == 0 else STYLE_B


def _i2(pts: np.ndarray) -> np.ndarray:
    return np.round(pts).astype(np.int32)


class VectorOverlayRenderer:
    """
    styles:
      - fluid: filled palm + finger ribbons + glow
      - wire:  skeleton + joints
      - outline: convex hull silhouette

    performance:
      - fast=True  (live default): no per-stroke full-frame alpha, light trail
      - fast=False (export): richer glow + optional vignette
    """

    def __init__(
        self,
        style: str = "fluid",
        *,
        show_source: bool = True,
        source_dim: float = 0.35,
        trail: int = 6,
        fast: bool = False,
        vignette: bool | None = None,
    ) -> None:
        self.style = style
        self.show_source = show_source
        self.source_dim = source_dim
        self.trail = max(0, trail)
        self.fast = fast
        self.vignette = (not fast) if vignette is None else vignette
        self._history: list[list[HandPose]] = []
        self._vignette_mask: np.ndarray | None = None
        self._vignette_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        self._history.clear()

    def render(self, frame_bgr: np.ndarray, frame_hands: FrameHands) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.show_source:
            if self.source_dim >= 0.98:
                out = frame_bgr.copy()
            else:
                # cheaper than convertScaleAbs float path for live
                out = cv2.convertScaleAbs(frame_bgr, alpha=self.source_dim, beta=0)
        else:
            out = np.empty((h, w, 3), dtype=np.uint8)
            out[:] = (12, 10, 14)

        hands = frame_hands.hands
        self._history.append(hands)
        max_hist = self.trail + 1
        if len(self._history) > max_hist:
            del self._history[:-max_hist]

        # ghost trails — only tip trails in fast mode
        if self.trail > 0 and len(self._history) > 1:
            if self.fast:
                self._draw_tip_trails(out)
            else:
                ghosts = self._history[:-1]
                for gi, gh in enumerate(ghosts):
                    age = len(ghosts) - gi
                    alpha = 0.12 * (1.0 - (age - 1) / max(self.trail, 1))
                    self._draw_hands(out, gh, alpha_scale=alpha, ghost=True)

        self._draw_hands(out, hands, alpha_scale=1.0, ghost=False)

        if self.vignette:
            self._apply_vignette(out, strength=0.28)
        return out

    def _draw_tip_trails(self, canvas: np.ndarray) -> None:
        """Cheap motion trails: only fingertip polylines from history."""
        tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
        # group by hand index over history
        hist = self._history
        if len(hist) < 2:
            return
        # for each hand slot 0/1, draw tip paths
        for hand_i in range(2):
            color = STYLE_A["stroke"] if hand_i == 0 else STYLE_B["stroke"]
            for tip in tips:
                path: list[tuple[int, int]] = []
                for frame_hands in hist:
                    if hand_i >= len(frame_hands):
                        continue
                    x, y = frame_hands[hand_i].points[tip, :2]
                    path.append((int(x), int(y)))
                if len(path) >= 2:
                    arr = np.array(path, dtype=np.int32)
                    cv2.polylines(canvas, [arr], False, color, 2, cv2.LINE_AA)

    def _draw_hands(
        self,
        canvas: np.ndarray,
        hands: list[HandPose],
        alpha_scale: float,
        ghost: bool,
    ) -> None:
        for i, hand in enumerate(hands):
            style = _style_for(hand, i)
            if self.style == "wire":
                self._draw_wire(canvas, hand, style)
            elif self.style == "outline":
                self._draw_outline(canvas, hand, style)
            else:
                self._draw_fluid(canvas, hand, style, ghost=ghost)

    def _draw_fluid(
        self,
        canvas: np.ndarray,
        hand: HandPose,
        style: dict,
        *,
        ghost: bool = False,
    ) -> None:
        palm = _palm_scale(hand)
        stroke_w = max(2, int(palm * 0.06))
        joint_r = max(3, int(palm * 0.05))
        palm_pts = _i2(hand.points[list(PALM_RING), :2])

        if not self.fast and not ghost:
            # soft glow once, offline quality only
            glow = np.zeros_like(canvas)
            cv2.fillPoly(glow, [palm_pts], style["glow"], lineType=cv2.LINE_AA)
            for name, idxs in FINGERS.items():
                ribbon = _finger_ribbon(hand, idxs, width=palm * (0.28 if name != "thumb" else 0.22))
                if len(ribbon) >= 3:
                    cv2.fillPoly(glow, [_i2(ribbon)], style["glow"], lineType=cv2.LINE_AA)
            cv2.addWeighted(glow, 0.22, canvas, 1.0, 0, canvas)

        # palm fill + outline (direct draw — no full-frame copy)
        cv2.fillPoly(canvas, [palm_pts], style["fill"], lineType=cv2.LINE_AA)
        cv2.polylines(
            canvas,
            [palm_pts],
            True,
            style["stroke"],
            max(2, stroke_w // 2),
            cv2.LINE_AA,
        )

        for name, idxs in FINGERS.items():
            width = palm * (0.30 if name == "middle" else 0.26 if name != "thumb" else 0.20)
            ribbon = _finger_ribbon(hand, idxs, width=width)
            if len(ribbon) < 3:
                continue
            ri = _i2(ribbon)
            cv2.fillPoly(canvas, [ri], style["fill"], lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [ri], True, style["stroke"], max(2, stroke_w // 2), cv2.LINE_AA)

        # fingertip web
        tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
        tip_pts = [tuple(map(int, hand.points[t, :2])) for t in tips]
        for a, b in zip(tip_pts, tip_pts[1:]):
            cv2.line(canvas, a, b, style["stroke"], max(1, stroke_w // 3), cv2.LINE_AA)

        # joints — direct circles
        tip_set = set(tips)
        for i in range(21):
            x, y = int(hand.points[i, 0]), int(hand.points[i, 1])
            r = joint_r + (2 if i in tip_set else 0)
            cv2.circle(canvas, (x, y), r, style["joint"], -1, cv2.LINE_AA)

        center = tuple(map(int, hand.points[list(PALM_RING), :2].mean(axis=0)))
        cv2.circle(canvas, center, max(4, joint_r), style["stroke"], -1, cv2.LINE_AA)

    def _draw_wire(self, canvas: np.ndarray, hand: HandPose, style: dict) -> None:
        palm = _palm_scale(hand)
        thick = max(2, int(palm * 0.05))
        pts = hand.as_int()
        for a, b in CONNECTIONS:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), style["stroke"], thick, cv2.LINE_AA)
        r = max(3, thick)
        for i in range(21):
            cv2.circle(canvas, tuple(pts[i]), r, style["joint"], -1, cv2.LINE_AA)

    def _draw_outline(self, canvas: np.ndarray, hand: HandPose, style: dict) -> None:
        palm = _palm_scale(hand)
        idxs = [
            WRIST,
            THUMB_TIP,
            INDEX_TIP,
            MIDDLE_TIP,
            RING_TIP,
            PINKY_TIP,
            PINKY_MCP,
            RING_MCP,
            MIDDLE_MCP,
            INDEX_MCP,
        ]
        raw = hand.points[idxs, :2].astype(np.float32)
        hull = cv2.convexHull(raw.reshape(-1, 1, 2))
        cv2.fillPoly(canvas, [hull.astype(np.int32)], style["fill"], lineType=cv2.LINE_AA)
        cv2.polylines(
            canvas,
            [hull.astype(np.int32)],
            True,
            style["stroke"],
            max(3, int(palm * 0.08)),
            cv2.LINE_AA,
        )

    def _apply_vignette(self, img: np.ndarray, strength: float = 0.28) -> None:
        h, w = img.shape[:2]
        if self._vignette_mask is None or self._vignette_shape != (h, w):
            y, x = np.ogrid[:h, :w]
            cy, cx = h / 2.0, w / 2.0
            dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
            mask = np.clip(1.0 - (dist - 0.4).clip(0, None) * strength * 1.4, 0.45, 1.0)
            self._vignette_mask = mask.astype(np.float32)
            self._vignette_shape = (h, w)
        # single multiply; still offline-only by default
        img[:] = (img.astype(np.float32) * self._vignette_mask[..., None]).astype(np.uint8)
