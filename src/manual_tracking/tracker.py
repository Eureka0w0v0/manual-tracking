"""MediaPipe Hand Landmarker wrapper + 持久轨迹 One Euro 时域滤波.

稳定性设计(全部有实测数据背书, 见 git 历史里的 5 份审查报告):
- 手用持久 slot 跟踪(track_id), 帧间按手腕距离做 2x2 最优指派——
  handedness 标签翻转/输出顺序变化不再互换两只手的滤波历史
- handedness 也锁在轨迹上(连续 5 帧改判才切), 单帧翻转不再让 orient 反号
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

from .landmarks import INDEX_MCP, PINKY_MCP, WRIST

# One Euro 参数(像素单位, 30fps 实测甜点)
OE_MIN_CUTOFF = 1.0  # Hz, 主调静止残噪(嫌抖降 0.6, 嫌拖影升 1.5)
OE_BETA = 0.04  # 速度增益, 主调运动滞后(甜点区 0.03-0.05)
OE_D_CUTOFF = 1.0  # Hz, 速度估计低通
# 手短暂丢失时滤波历史保留多久。按**墙钟**而非检测 tick 计: 检测率 = 1/max(D,P),
# 30Hz 时 6 tick 是 200ms, 机器负载高掉到 10Hz 就变成 600ms —— 同一个常数漂 3 倍。
SLOT_TTL_MS = 250.0
MATCH_PALM_SCALE = 1.5  # 配对门限 = 该值 × 掌宽(|MCP5-MCP17|)
MATCH_MIN_PX = 80.0  # 掌宽异常小时的门限下限
# handedness 改判需要连续这么多帧确认(与 floatcube 的 PINCH_OFF_FRAMES 同一个
# 手法: 建立瞬时, 推翻保守)。5 帧 = 167ms@30fps, 真的换手感觉不出延迟。
LABEL_FLIP_FRAMES = 5


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
    """一条手部轨迹: track_id + One Euro 滤波状态 + handedness + 寿命."""

    __slots__ = ("sid", "x", "dx", "t_ms", "label", "_dissent")

    def __init__(self, sid: int, pts: np.ndarray, ts_ms: float, label: str) -> None:
        self.sid = sid
        self.x = pts.copy()
        self.dx = np.zeros((21, 2), np.float32)
        self.t_ms = float(ts_ms)  # 最后一次匹配上的时刻; TTL 由它算, 不用 tick 计数
        self.label = label  # 锁在轨迹上的 handedness(见 vote)
        # 起手就差一票 —— 首帧标签**没有信心**, 下一个不同的观测立刻推翻它。
        # MediaPipe 刚认出一只手时 handedness 最不准: 实测 assets/sample.mp4 的
        # tid5 裸标签是 Right×1 | Left×28(首帧判错, 之后 28 帧全对)。把首帧当
        # 可信基准的话, 错标签会活满 LABEL_FLIP_FRAMES 帧, 比根本不锁存还差。
        # 锁存要挡的是轨迹**中段**的单帧噪声, 不是给首帧背书。
        self._dissent = LABEL_FLIP_FRAMES - 1

    def vote(self, label: str) -> str:
        """这条轨迹的 handedness: 带滞回, 不直接用当帧的裸标签.

        MediaPipe 的 handedness 是**每帧独立**判的, 翻腕/遮挡时会单帧翻转——
        实测 assets/sample.mp4 的 408 个手帧里翻 1 次。而 handgeom.orient()
        的符号直接取它(sign = +1 if Right else −1), 翻一次 orient 就整个反号
        (实测那一帧 |Δorient| = 0.55), mirror 的半张纸亮度跟着闪一下
        (B_SWING 0.65 下 ≈36% 的亮度跳)。

        同一条轨迹就是同一只手, 所以标签该跟着轨迹走; 轨迹断了(TTL 过期)才
        开新 slot 重新取标签, 因此锁存不会把"真的换了只手"锁死。

        滞回只对**已经确认过至少一帧**的标签生效 —— 首帧标签起手就差一票,
        见 __init__ 里 _dissent 的初值。
        """
        if label == self.label:
            self._dissent = 0
        else:
            self._dissent += 1
            if self._dissent >= LABEL_FLIP_FRAMES:
                self.label = label
                self._dissent = 0
        return self.label

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
        filter_on: bool = True,
        infer_max_side: int = 0,
    ) -> None:
        """
        filter_on:
          One Euro 时域滤波开关。滤波强度由模块级 OE_* 常量决定, 不是每次调用
          传进来的——早先这里是个 float `smooth`, 但自从换成 One Euro 之后
          数值就只被拿来跟 0 比大小了(One Euro 的截止频率随速度自适应, 没有
          "一个 EMA 系数"可调)。留着 float 只会让人以为能调强度。
        infer_max_side:
          0 = 全帧推理(实测 Apple Silicon 上最快且尾部误差最小)
          >0 = 按最长边降采样(仅在推理确实过慢的机器上使用)
        """
        self.filter_on = bool(filter_on)
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

    def _track(
        self, pts_list: list[np.ndarray], labels: list[str], ts_ms: float
    ) -> list[tuple[np.ndarray, int, str]]:
        """配对 + 滤波: 返回 [(filtered_pts, track_id, handedness)], 与输入同序.

        两手时做 2x2 最优指派(总距离最小), 消除贪心的顺序依赖;
        门限随掌宽自适应, 杜绝 300px 级跨手误配。

        handedness 也在这里定: 配上轨迹的走 _Slot.vote 的滞回, 新轨迹直接取
        当帧标签。关滤波 = 没有轨迹, 裸标签直通(--no-filter 本来就是"给我原始
        观测"的意思)。
        """
        if not self.filter_on:
            return [(p, -1, lb) for p, lb in zip(pts_list, labels, strict=True)]

        old = self._slots
        pairs: list[tuple[int, int]] = []
        if pts_list and old:

            def d(k: int, j: int) -> float:
                return float(np.linalg.norm(pts_list[k][WRIST, :2] - old[j].x[WRIST, :2]))

            def lim(j: int) -> float:
                palm = float(np.linalg.norm(old[j].x[INDEX_MCP, :2] - old[j].x[PINKY_MCP, :2]))
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

        out: dict[int, tuple[np.ndarray, int, str]] = {}
        matched = {j for _, j in pairs}
        keep_slots: list[_Slot] = []
        for k, j in pairs:
            s = old[j]
            # filt 会把 t_ms 推到当前; vote 给出这条轨迹锁定的 handedness
            out[k] = (s.filt(pts_list[k], ts_ms), s.sid, s.vote(labels[k]))
            keep_slots.append(s)
        for j, s in enumerate(old):
            if j not in matched and ts_ms - s.t_ms <= SLOT_TTL_MS:
                keep_slots.append(s)
        for k in range(len(pts_list)):
            if k not in out:  # 没配上任何旧轨迹 → 开一条新的, 当帧不滤波
                s = _Slot(self._next_id, pts_list[k], ts_ms, labels[k])
                self._next_id += 1
                keep_slots.append(s)
                out[k] = (pts_list[k], s.sid, s.label)
        self._slots = keep_slots
        return [out[k] for k in range(len(pts_list))]

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
        labels: list[str] = []
        scores: list[float] = []
        pts_list: list[np.ndarray] = []
        for i, lms in enumerate(result.hand_landmarks):
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                labels.append(cat.category_name)
                scores.append(float(cat.score))
            else:
                labels.append("Unknown")
                scores.append(0.0)
            # landmarks are normalized to infer image → map to full frame pixels
            pts_list.append(
                np.array([[lm.x * iw * inv, lm.y * ih * inv, lm.z] for lm in lms], dtype=np.float32)
            )

        # _track 保证"与输入同序等长", strict=True 把这条契约钉成断言。
        # handedness 取 _track 给的**轨迹级**标签, 不是上面那份逐帧裸标签。
        tracked = self._track(pts_list, labels, float(timestamp_ms))
        for score, (pts, sid, label) in zip(scores, tracked, strict=True):
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
