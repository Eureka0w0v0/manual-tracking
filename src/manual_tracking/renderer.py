"""Space-fabric hand effect.

Hands: bone skeleton ONLY — no palm plates / aura fills.
Between hands when pulled: sparse energy filaments + one membrane.
"""

from __future__ import annotations

import math
import random

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
WHITE_HOT = (220, 245, 255)

TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def _palm_center(hand: HandPose) -> np.ndarray:
    return hand.points[list(PALM_RING), :2].mean(axis=0).astype(np.float32)


def _palm_scale(hand: HandPose) -> float:
    p = hand.points
    a = p[INDEX_MCP, :2]
    b = p[PINKY_MCP, :2]
    c = p[WRIST, :2]
    return float(max(np.linalg.norm(a - b), np.linalg.norm((a + b) * 0.5 - c), 20.0))


def _blend_poly(canvas, pts, color, alpha, outline=None, outline_w=1):
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
    a = float(np.clip(alpha, 0, 1))
    m = (mask > 0)[:, :, None].astype(np.float32)
    col = np.array(color, np.float32)
    roi[:] = (roi.astype(np.float32) * (1 - a * m) + col * (a * m)).astype(np.uint8)
    if outline is not None and outline_w > 0:
        cv2.polylines(canvas, [poly.reshape(-1, 2)], True, outline, outline_w, cv2.LINE_AA)


class VectorOverlayRenderer:
    """
    styles:
      fabric — skeleton + between-hand energy (default)
      track  — skeleton + few tip faces
      wire   — skeleton only
      outline — skeleton only (alias)
    """

    def __init__(
        self,
        style: str = "fabric",
        *,
        show_source: bool = True,
        source_dim: float = 0.55,
        trail: int = 0,
        fast: bool = True,
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
        self.trail = 0
        self.fast = fast
        self.vignette = False
        self.effect = effect if effect in ("energy", "calm", "hot") else "energy"
        self._pull_ema = 0.0
        self._t = 0.0
        self._sparks: list[list[float]] = []
        self._rng = random.Random(7)

    def reset(self) -> None:
        self._pull_ema = 0.0
        self._sparks.clear()

    def next_effect(self) -> str:
        order = ["energy", "calm", "hot"]
        i = order.index(self.effect) if self.effect in order else 0
        self.effect = order[(i + 1) % len(order)]
        return self.effect

    def render(self, frame_bgr: np.ndarray, frame_hands: FrameHands) -> np.ndarray:
        self._t += 0.05
        if self.show_source:
            out = (
                frame_bgr.copy()
                if self.source_dim >= 0.98
                else cv2.convertScaleAbs(frame_bgr, alpha=self.source_dim, beta=0)
            )
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

    def _intensity(self) -> float:
        if self.effect == "calm":
            return 0.65
        if self.effect == "hot":
            return 1.25
        return 1.0

    def _skeleton(self, canvas: np.ndarray, hand: HandPose, index: int = 0) -> None:
        """Bone lines + joints only. No filled palm graphics."""
        color = GOLD if index == 0 else ORANGE
        pts = hand.as_int()
        for a, b in CONNECTIONS:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for i in range(21):
            r = 4 if i in TIP_IDS else 3
            cv2.circle(canvas, tuple(pts[i]), r, WHITE_HOT, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[i]), r, color, 1, cv2.LINE_AA)

    def _draw_fabric(self, canvas: np.ndarray, hands: list[HandPose]) -> None:
        inten = self._intensity()
        glow = np.zeros_like(canvas)

        if len(hands) >= 2:
            left, right = sorted(hands[:2], key=lambda h: float(_palm_center(h)[0]))
            dist = float(np.linalg.norm(_palm_center(right) - _palm_center(left)))
            scale = 0.5 * (_palm_scale(left) + _palm_scale(right))
            pull = float(np.clip((dist - scale * 0.6) / max(scale * 2.4, 1.0), 0.0, 1.0))
            self._pull_ema = pull if self._pull_ema < 1e-6 else self._pull_ema * 0.7 + pull * 0.3
            pull = self._pull_ema

            # between hands only — never paint over the hand surface with plates
            self._membrane(glow, canvas, left, right, pull, scale, inten)
            self._filaments(glow, left, right, pull, inten)
            self._update_sparks(canvas, left, right, pull, inten)
        else:
            self._pull_ema *= 0.85

        cv2.addWeighted(canvas, 1.0, glow, 0.85 * inten, 0, canvas)

        for i, h in enumerate(hands):
            self._skeleton(canvas, h, i)

    def _draw_track(self, canvas: np.ndarray, hands: list[HandPose]) -> None:
        if len(hands) >= 2:
            left, right = sorted(hands[:2], key=lambda h: float(_palm_center(h)[0]))
            dist = float(np.linalg.norm(_palm_center(right) - _palm_center(left)))
            scale = 0.5 * (_palm_scale(left) + _palm_scale(right))
            pull = float(np.clip((dist - scale * 0.85) / max(scale * 2.0, 1.0), 0.0, 1.0))
            self._pull_ema = pull if self._pull_ema < 1e-6 else self._pull_ema * 0.75 + pull * 0.25
            pull = self._pull_ema
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
                    thick = scale * (0.08 + 0.15 * pull)
                    mid = (p0 + p1) * 0.5
                    dia = np.array([p0, mid + n * thick, p1, mid - n * thick], np.float32)
                    _blend_poly(canvas, dia, col, 0.35 * pull + 0.15, outline=(20, 18, 20), outline_w=1)
        else:
            self._pull_ema *= 0.9

        for i, h in enumerate(hands):
            self._skeleton(canvas, h, i)

    def _filaments(
        self,
        glow: np.ndarray,
        left: HandPose,
        right: HandPose,
        pull: float,
        inten: float,
    ) -> None:
        if pull < 0.05:
            return
        pairs = [INDEX_TIP, MIDDLE_TIP, RING_TIP]
        if pull > 0.4:
            pairs = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        for tip in pairs:
            p0 = left.points[tip, :2].astype(np.float32)
            p1 = right.points[tip, :2].astype(np.float32)
            self._energy_curve(glow, p0, p1, pull, inten)

    def _energy_curve(
        self,
        glow: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        pull: float,
        inten: float,
    ) -> None:
        v = p1 - p0
        ln = float(np.linalg.norm(v))
        if ln < 12:
            return
        n = np.array([-v[1], v[0]], np.float32) / (ln + 1e-6)
        wob = math.sin(self._t * 2.3 + ln * 0.01) * (8 + 28 * pull)
        wob2 = math.cos(self._t * 1.7 + ln * 0.02) * (6 + 18 * pull)
        c1 = p0 * 0.65 + p1 * 0.35 + n * wob
        c2 = p0 * 0.35 + p1 * 0.65 - n * wob2

        pts = []
        steps = 16 if self.fast else 24
        for i in range(steps + 1):
            t = i / steps
            u = 1 - t
            p = (u**3) * p0 + 3 * (u**2) * t * c1 + 3 * u * (t**2) * c2 + (t**3) * p1
            pts.append(p)
        arr = np.round(np.array(pts)).astype(np.int32)

        cv2.polylines(glow, [arr], False, DEEP, max(6, int(10 * pull * inten)), cv2.LINE_AA)
        cv2.polylines(glow, [arr], False, ORANGE, max(3, int(5 * pull * inten)), cv2.LINE_AA)
        cv2.polylines(glow, [arr], False, GOLD, max(1, int(2 * inten)), cv2.LINE_AA)
        if pull > 0.45:
            cv2.polylines(glow, [arr], False, WHITE_HOT, 1, cv2.LINE_AA)

    def _membrane(
        self,
        glow: np.ndarray,
        canvas: np.ndarray,
        left: HandPose,
        right: HandPose,
        pull: float,
        scale: float,
        inten: float,
    ) -> None:
        if pull < 0.08:
            return
        cL, cR = _palm_center(left), _palm_center(right)
        v = cR - cL
        ln = float(np.linalg.norm(v))
        if ln < 10:
            return
        n = np.array([-v[1], v[0]], np.float32) / (ln + 1e-6)
        half = scale * (0.25 + 0.55 * pull)
        wave = math.sin(self._t * 1.8) * scale * 0.08 * pull
        wave2 = math.cos(self._t * 2.1) * scale * 0.06 * pull

        segs = 8
        top, bot = [], []
        for i in range(segs + 1):
            t = i / segs
            p = cL * (1 - t) + cR * t
            bulge = math.sin(math.pi * t) * half
            wiggle = math.sin(self._t * 2.5 + t * 4.0) * wave
            top.append(p + n * (bulge + wiggle))
            bot.append(p - n * (bulge * 0.85 + wave2 * math.sin(t * 3 + self._t)))
        poly = np.array(top + bot[::-1], dtype=np.float32)

        alpha = (0.12 + 0.22 * pull) * inten
        _blend_poly(canvas, poly, ORANGE, alpha, outline=None)
        _blend_poly(canvas, poly, GOLD, alpha * 0.45, outline=None)

        pr = np.round(poly).astype(np.int32)
        cv2.polylines(glow, [pr], True, GOLD, 2, cv2.LINE_AA)
        cv2.polylines(glow, [pr], True, WHITE_HOT, 1, cv2.LINE_AA)

        for k, t in enumerate((0.3, 0.5, 0.7)):
            if pull < 0.2 and k != 1:
                continue
            a = top[int(t * segs)]
            b = bot[int(t * segs)]
            cv2.line(
                glow,
                tuple(map(int, a)),
                tuple(map(int, b)),
                AMBER if k == 1 else ORANGE,
                1,
                cv2.LINE_AA,
            )

    def _update_sparks(
        self,
        canvas: np.ndarray,
        left: HandPose,
        right: HandPose,
        pull: float,
        inten: float,
    ) -> None:
        if pull < 0.15:
            self._sparks.clear()
            return
        cL, cR = _palm_center(left), _palm_center(right)
        spawn = 2 if self.fast else 4
        for _ in range(spawn):
            t = self._rng.random()
            p = cL * (1 - t) + cR * t
            p = p + np.array(
                [self._rng.uniform(-12, 12), self._rng.uniform(-18, 18)],
                np.float32,
            )
            self._sparks.append(
                [
                    float(p[0]),
                    float(p[1]),
                    self._rng.uniform(-1.5, 1.5),
                    self._rng.uniform(-2.5, -0.5),
                    self._rng.uniform(0.4, 1.0),
                ]
            )
        if len(self._sparks) > 40:
            self._sparks = self._sparks[-40:]

        alive = []
        for x, y, vx, vy, life in self._sparks:
            life -= 0.04
            if life <= 0:
                continue
            x += vx
            y += vy
            r = max(1, int(3 * life * inten))
            col = WHITE_HOT if life > 0.6 else GOLD if life > 0.3 else ORANGE
            cv2.circle(canvas, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
            alive.append([x, y, vx, vy, life])
        self._sparks = alive
