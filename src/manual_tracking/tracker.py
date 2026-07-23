"""MediaPipe Hand Landmarker wrapper + light temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class HandPose:
    """One detected hand in pixel space."""

    handedness: str  # "Left" | "Right"
    score: float
    # shape (21, 3) -> x_px, y_px, z_norm
    points: np.ndarray

    def as_int(self) -> np.ndarray:
        return np.round(self.points[:, :2]).astype(np.int32)


@dataclass
class FrameHands:
    index: int
    hands: list[HandPose] = field(default_factory=list)


class HandTracker:
    """Stateful tracker that smooths landmarks across frames."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.4,
        min_presence_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        smooth: float = 0.55,
        infer_max_side: int = 0,
    ) -> None:
        """
        infer_max_side:
          0 = run on full frame
          e.g. 640 = downscale longest side for MediaPipe (much faster for live)
        """
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.infer_max_side = max(0, int(infer_max_side))
        self._prev: dict[str, np.ndarray] = {}

        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def process_bgr(self, frame_bgr: np.ndarray, frame_index: int, timestamp_ms: int) -> FrameHands:
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        infer = frame_bgr
        if self.infer_max_side > 0:
            long_side = max(h, w)
            if long_side > self.infer_max_side:
                scale = self.infer_max_side / float(long_side)
                infer = cv2.resize(
                    frame_bgr,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

        ih, iw = infer.shape[:2]
        rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
        # contiguous helps MediaPipe / TFLite a bit
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[HandPose] = []
        if not result.hand_landmarks:
            self._prev.clear()
            return FrameHands(index=frame_index, hands=hands)

        inv = 1.0 / scale if scale > 0 else 1.0
        n = len(result.hand_landmarks)
        for i in range(n):
            lms = result.hand_landmarks[i]
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                label = cat.category_name
                score = float(cat.score)
            else:
                label, score = "Unknown", 0.0

            # landmarks are normalized to infer image → map to full frame pixels
            pts = np.array(
                [[lm.x * iw * inv, lm.y * ih * inv, lm.z] for lm in lms],
                dtype=np.float32,
            )
            key = label if label in ("Left", "Right") else f"hand{i}"
            if key in self._prev and self.smooth > 0:
                pts = self.smooth * self._prev[key] + (1.0 - self.smooth) * pts
            self._prev[key] = pts.copy()
            hands.append(HandPose(handedness=label, score=score, points=pts))

        order = {"Right": 0, "Left": 1}
        hands.sort(key=lambda hp: order.get(hp.handedness, 9))
        return FrameHands(index=frame_index, hands=hands)

    def iter_video(self, video_path: str | Path) -> Iterable[tuple[np.ndarray, FrameHands, float]]:
        """Yield (frame_bgr, hands, fps)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1e-3:
            fps = 30.0

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = int(round(idx * 1000.0 / fps))
                hands = self.process_bgr(frame, idx, ts)
                yield frame, hands, fps
                idx += 1
        finally:
            cap.release()
