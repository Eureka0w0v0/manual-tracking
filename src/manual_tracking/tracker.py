"""MediaPipe Hand Landmarker wrapper + 持久轨迹 One Euro 时域滤波.

稳定性设计(全部有实测数据背书, 见 git 历史里的 5 份审查报告):
- 手用持久 slot 跟踪(track_id), 帧间按手腕距离做 2x2 最优指派——
  handedness 标签翻转/输出顺序变化不再互换两只手的滤波历史
- One Euro 替代固定 EMA: 静止残噪不劣化, 快速运动滞后 23.5→9px(角点级)
- 短暂丢检测保留 slot 250ms(TTL, 按墙钟计), 不再单帧清史导致恢复帧裸输出瞬移
- 只滤 xy; z 保留当帧原始观测(AE 导出数据不被跨参考系混合污染)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# One Euro 参数(像素单位, 30fps 实测甜点)
OE_MIN_CUTOFF = 1.0  # Hz, 主调静止残噪(嫌抖降 0.6, 嫌拖影升 1.5)
OE_BETA = 0.04  # 速度增益, 主调运动滞后(甜点区 0.03-0.05)
OE_D_CUTOFF = 1.0  # Hz, 速度估计低通
# 手短暂丢失时滤波历史保留多久。按**墙钟**而非检测 tick 计: 检测率 = 1/max(D,P),
# 30Hz 时 6 tick 是 200ms, 机器负载高掉到 10Hz 就变成 600ms —— 同一个常数漂 3 倍。
SLOT_TTL_MS = 250.0
MATCH_PALM_SCALE = 1.5  # 配对门限 = 该值 × 掌宽(|MCP5-MCP17|)
MATCH_MIN_PX = 80.0  # 掌宽异常小时的门限下限


@dataclass
class HandPose:
    """One detected hand in pixel space."""

    handedness: str  # "Left" | "Right"
    score: float
    # shape (21, 3) -> x_px, y_px, z_norm
    points: np.ndarray
    track_id: int = -1  # 跨帧持久身份(-1 = 滤波关闭/未跟踪)

    def as_int(self) -> np.ndarray:
        return np.round(self.points[:, :2]).astype(np.int32)


@dataclass
class FrameHands:
    index: int
    hands: list[HandPose] = field(default_factory=list)


class _Slot:
    """一条手部轨迹: track_id + One Euro 滤波状态 + 寿命."""

    __slots__ = ("sid", "x", "dx", "t_ms")

    def __init__(self, sid: int, pts: np.ndarray, ts_ms: float) -> None:
        self.sid = sid
        self.x = pts.copy()
        self.dx = np.zeros((21, 2), np.float32)
        self.t_ms = float(ts_ms)  # 最后一次匹配上的时刻; TTL 由它算, 不用 tick 计数

    def filt(self, pts: np.ndarray, ts_ms: float) -> np.ndarray:
        """One Euro: 截止频率随速度自适应; 只滤 xy, z 直通."""
        dt = float(np.clip((float(ts_ms) - self.t_ms) / 1000.0, 1.0 / 120.0, 0.25))
        self.t_ms = float(ts_ms)
        xy, prev = pts[:, :2], self.x[:, :2]
        ad = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * OE_D_CUTOFF * dt))
        self.dx = ad * (xy - prev) / dt + (1.0 - ad) * self.dx
        cutoff = OE_MIN_CUTOFF + OE_BETA * np.abs(self.dx)
        a = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))
        out = pts.copy()
        out[:, :2] = a * xy + (1.0 - a) * prev
        self.x = out.copy()
        return out


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
        smooth:
          <=0 关闭时域滤波(裸输出); >0 启用 One Euro(数值本身不再是 EMA 系数)
        infer_max_side:
          0 = 全帧推理(实测 Apple Silicon 上最快且尾部误差最小)
          >0 = 按最长边降采样(仅在推理确实过慢的机器上使用)
        """
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.infer_max_side = max(0, int(infer_max_side))
        self._slots: list[_Slot] = []
        self._next_id = 0

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

    def _expire(self, ts_ms: float) -> None:
        """丢掉超过 TTL 没再匹配上的轨迹(墙钟计时, 与检测率无关)."""
        self._slots = [s for s in self._slots if ts_ms - s.t_ms <= SLOT_TTL_MS]

    def _track(self, pts_list: list[np.ndarray], ts_ms: float) -> list[tuple[np.ndarray, int]]:
        """配对 + 滤波: 返回 [(filtered_pts, track_id)], 与输入同序.

        两手时做 2x2 最优指派(总距离最小), 消除贪心的顺序依赖;
        门限随掌宽自适应, 杜绝 300px 级跨手误配。
        """
        if self.smooth <= 0:
            return [(p, -1) for p in pts_list]

        old = self._slots
        pairs: list[tuple[int, int]] = []
        if pts_list and old:

            def d(k: int, j: int) -> float:
                return float(np.linalg.norm(pts_list[k][0, :2] - old[j].x[0, :2]))

            def lim(j: int) -> float:
                palm = float(np.linalg.norm(old[j].x[5, :2] - old[j].x[17, :2]))
                return max(MATCH_PALM_SCALE * palm, MATCH_MIN_PX)

            if len(pts_list) == 2 and len(old) == 2:
                keep = d(0, 0) + d(1, 1) <= d(0, 1) + d(1, 0)
                cand = ((0, 0), (1, 1)) if keep else ((0, 1), (1, 0))
                pairs = [(k, j) for k, j in cand if d(k, j) < lim(j)]
            else:
                order = sorted(
                    (d(k, j), k, j) for k in range(len(pts_list)) for j in range(len(old))
                )
                uk: set[int] = set()
                uj: set[int] = set()
                for dd, k, j in order:
                    if k in uk or j in uj or dd >= lim(j):
                        continue
                    pairs.append((k, j))
                    uk.add(k)
                    uj.add(j)

        out: list[tuple[np.ndarray, int] | None] = [None] * len(pts_list)
        matched = {j for _, j in pairs}
        keep_slots: list[_Slot] = []
        for k, j in pairs:
            s = old[j]
            out[k] = (s.filt(pts_list[k], ts_ms), s.sid)  # filt 会把 t_ms 推到当前
            keep_slots.append(s)
        for j, s in enumerate(old):
            if j not in matched and ts_ms - s.t_ms <= SLOT_TTL_MS:
                keep_slots.append(s)
        for k in range(len(pts_list)):
            if out[k] is None:
                s = _Slot(self._next_id, pts_list[k], ts_ms)
                self._next_id += 1
                keep_slots.append(s)
                out[k] = (pts_list[k], s.sid)
        self._slots = keep_slots
        return out  # type: ignore[return-value]

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
            self._expire(float(timestamp_ms))  # 短暂丢检测不清史, TTL 内恢复仍有平滑
            return FrameHands(index=frame_index, hands=hands)

        inv = 1.0 / scale if scale > 0 else 1.0
        labels: list[tuple[str, float]] = []
        pts_list: list[np.ndarray] = []
        for i, lms in enumerate(result.hand_landmarks):
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                labels.append((cat.category_name, float(cat.score)))
            else:
                labels.append(("Unknown", 0.0))
            # landmarks are normalized to infer image → map to full frame pixels
            pts_list.append(
                np.array([[lm.x * iw * inv, lm.y * ih * inv, lm.z] for lm in lms], dtype=np.float32)
            )

        for (label, score), (pts, sid) in zip(labels, self._track(pts_list, float(timestamp_ms))):
            hands.append(HandPose(handedness=label, score=score, points=pts, track_id=sid))

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
