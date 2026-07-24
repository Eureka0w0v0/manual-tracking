"""Realtime webcam live loop.

Architecture (after review of async negative opts):
- MediaPipe on a worker thread, but only submit when worker is IDLE
- Submit a pre-downscaled frame for inference (not full 1280x720 copy)
- Main thread: camera → draw → imshow; never blocked by detect
- No discarded full-res memcpy when worker is busy
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .pipeline import default_model_path
from .renderer import STYLES, VectorOverlayRenderer
from .tracker import FrameHands, HandTracker


def _open_camera(camera: int, width: int, height: int) -> cv2.VideoCapture:
    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(cv2.CAP_ANY)

    last_err = None
    for backend in backends:
        cap = cv2.VideoCapture(camera, backend)
        for _ in range(20):
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap
            time.sleep(0.15)
        last_err = f"backend={backend}"
        cap.release()

    raise RuntimeError(
        f"打不开摄像头 #{camera} ({last_err}).\n"
        "macOS: 系统设置 → 隐私与安全性 → 摄像头 → 打开终端\n"
        "  cd ~/Claudecode/manual-tracking && ./run.sh live"
    )


class _AsyncHandDetector:
    """
    Worker holds HandTracker.
    submit() only accepted when idle — no discarded copies.
    Main passes a full-res frame ownership transfer ONLY when idle;
    worker resizes inside process_bgr (main never INTER_AREA).
    """

    def __init__(self, tracker: HandTracker) -> None:
        self._tracker = tracker
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: np.ndarray | None = None
        self._pending_meta: tuple[int, int] | None = None  # idx, ts
        self._busy = False
        self._hands = FrameHands(index=-1, hands=[])
        self._detect_ms = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="hand-detect", daemon=True)
        self._thread.start()

    def idle(self) -> bool:
        with self._lock:
            return (not self._busy) and self._pending is None

    def submit(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> bool:
        """Return False if worker busy (caller should skip copy)."""
        with self._cond:
            if self._busy or self._pending is not None:
                return False
            # frame ownership transfers; caller must not reuse this array
            self._pending = frame_bgr
            self._pending_meta = (frame_index, timestamp_ms)
            self._busy = True
            self._cond.notify()
            return True

    def latest(self) -> tuple[FrameHands, float]:
        with self._lock:
            return self._hands, self._detect_ms

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait(timeout=0.05)
                if not self._running and self._pending is None:
                    return
                frame = self._pending
                meta = self._pending_meta
                self._pending = None
                self._pending_meta = None
            if frame is None or meta is None:
                with self._lock:
                    self._busy = False
                continue

            fi, ts = meta
            t0 = time.perf_counter()
            try:
                # tracker.infer_max_side handles downscale + maps coords to full-res
                hands = self._tracker.process_bgr(frame, fi, ts)
            except Exception:
                hands = FrameHands(index=fi, hands=[])
            dt = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._hands = hands
                self._detect_ms = dt
                self._busy = False


def run_live(
    *,
    camera: int = 0,
    model_path: str | Path | None = None,
    style: str = "mirror",
    show_source: bool = True,
    source_dim: float = 0.65,
    smooth: float = 0.35,
    mirror: bool = True,
    width: int = 1280,
    height: int = 720,
    infer_size: int = 480,
    record: str | Path | None = None,
    window_name: str = "Manual Tracking Live  |  Q退出 S风格 D暗底 R录制",
) -> None:
    model = Path(model_path) if model_path else default_model_path()
    if not model.exists():
        raise FileNotFoundError(f"model missing: {model}")

    # renderer 是风格名/别名的唯一规范化入口
    renderer = VectorOverlayRenderer(
        style=style,
        show_source=show_source,
        source_dim=source_dim,
    )
    styles = list(STYLES)
    style_idx = styles.index(renderer.style)

    cap = _open_camera(camera, width, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)

    writer: cv2.VideoWriter | None = None
    recording = False
    record_path: Path | None = Path(record) if record else None

    print("=" * 56)
    print("  MANUAL TRACKING LIVE — 折纸镜面 / 彩色玻璃盒 / TD横幅")
    print("  拇指+食指捏纸；翻转一只手拧麻花；捏死压成细线")
    print(f"  采集 {actual_w}x{actual_h} (req {width}x{height})  推理边 {infer_size}")
    print("  Q退出 | S风格 | D暗底 | R录制")
    print("=" * 56)

    frame_index = 0
    t0 = time.perf_counter()
    fps_ema = 0.0
    last_t = t0
    draw_ms_ema = 0.0
    last_ts = -1

    # Worker tracker downscales internally (main only copies when idle)
    tracker = HandTracker(
        model,
        smooth=smooth,
        num_hands=2,
        infer_max_side=infer_size,
        min_detection_confidence=0.45,
        min_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    detector = _AsyncHandDetector(tracker)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("摄像头读帧失败，退出")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            if abs(fw - width) > 8 or abs(fh - height) > 8:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                fh, fw = frame.shape[:2]

            timestamp_ms = int((time.perf_counter() - t0) * 1000)
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms

            # Only when worker idle: one full-res ownership copy for detect.
            # Never copy when busy (no discarded memcpy). Worker resizes.
            if detector.idle():
                detector.submit(frame.copy(), frame_index, timestamp_ms)

            hands, detect_ms = detector.latest()

            t_draw0 = time.perf_counter()
            out = renderer.render(frame, hands)
            draw_ms = (time.perf_counter() - t_draw0) * 1000.0
            draw_ms_ema = draw_ms if draw_ms_ema <= 1e-3 else draw_ms_ema * 0.85 + draw_ms * 0.15

            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-4)
            last_t = now
            fps_ema = inst if fps_ema <= 1e-3 else fps_ema * 0.85 + inst * 0.15

            # 录制在画 HUD 之前，成片不带黑条和状态文字
            if recording and writer is not None:
                writer.write(out)

            n_hands = len(hands.hands)
            busy = "busy" if not detector.idle() else "idle"
            hud = (
                f"FPS {fps_ema:5.1f}  det {detect_ms:5.1f}ms  "
                f"draw {draw_ms_ema:4.1f}ms  hands:{n_hands}  "
                f"{styles[style_idx]}  {busy}"
                f"{'  REC' if recording else ''}"
            )
            cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(
                out,
                hud,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200) if recording else (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            if recording:
                cv2.circle(out, (out.shape[1] - 24, 16), 7, (0, 0, 255), -1)

            cv2.imshow(window_name, out)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                style_idx = (style_idx + 1) % len(styles)
                renderer.style = styles[style_idx]
                print(f"style → {styles[style_idx]}")
            if key in (ord("d"), ord("D")):
                renderer.show_source = not renderer.show_source
                print(f"show_source → {renderer.show_source}")
            if key in (ord("+"), ord("=")):
                renderer.source_dim = float(min(1.0, renderer.source_dim + 0.05))
            if key in (ord("-"), ord("_")):
                renderer.source_dim = float(max(0.05, renderer.source_dim - 0.05))
            if key in (ord("r"), ord("R")):
                if not recording:
                    if record_path is None:
                        out_dir = Path(__file__).resolve().parents[2] / "output"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        record_path = out_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    hh, ww = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # 用实测 FPS 录制，避免快放/慢放（无实测值时退回 30）
                    fps_rec = float(np.clip(fps_ema, 10.0, 60.0)) if fps_ema > 1e-3 else 30.0
                    writer = cv2.VideoWriter(str(record_path), fourcc, fps_rec, (ww, hh))
                    if not writer.isOpened():
                        print("无法开始录制")
                        writer = None
                    else:
                        recording = True
                        print(f"REC start → {record_path}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"REC stop → {record_path}")
                    record_path = None

            frame_index += 1
    finally:
        detector.close()
        tracker.close()
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
