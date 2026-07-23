"""Realtime webcam hand tracking + vector overlay.

You are the subject — camera in, effect out, live.
Optimized for 30fps-ish on laptop webcam.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .pipeline import default_model_path
from .renderer import VectorOverlayRenderer
from .tracker import HandTracker


def _open_camera(camera: int, width: int, height: int) -> cv2.VideoCapture:
    """Open webcam; on macOS first launch may need Privacy prompt."""
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
                    # reduce internal buffer lag on some drivers
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
        "macOS: 系统设置 → 隐私与安全性 → 摄像头\n"
        "  → 打开「终端」或你用来跑命令的 App 的摄像头权限\n"
        "也可试: --camera 1\n"
        "然后在本机终端运行:\n"
        "  cd ~/Claudecode/manual-tracking && ./run.sh live"
    )


def run_live(
    *,
    camera: int = 0,
    model_path: str | Path | None = None,
    style: str = "fluid",
    show_source: bool = True,
    source_dim: float = 0.55,
    trail: int = 4,
    smooth: float = 0.45,
    mirror: bool = True,
    width: int = 960,
    height: int = 540,
    infer_size: int = 480,
    record: str | Path | None = None,
    window_name: str = "Manual Tracking Live  |  Q退出 S风格 D暗底 R录制 F性能",
) -> None:
    """
    Open webcam, track YOUR hands, draw the vector effect in realtime.

    Keys:
      q / ESC  quit
      s        cycle style: fluid → wire → outline
      d        toggle source dim / black bg
      r        start/stop recording
      f        toggle quality/perf (infer size / trail)
      + / -    source brightness
    """
    model = Path(model_path) if model_path else default_model_path()
    if not model.exists():
        raise FileNotFoundError(f"model missing: {model}")

    styles = ["fluid", "wire", "outline"]
    if style not in styles:
        style = "fluid"
    style_idx = styles.index(style)

    # live defaults: fast path on
    perf_mode = True
    cur_infer = infer_size
    cur_trail = trail

    renderer = VectorOverlayRenderer(
        style=styles[style_idx],
        show_source=show_source,
        source_dim=source_dim,
        trail=cur_trail,
        fast=True,
        vignette=False,
    )

    cap = _open_camera(camera, width, height)
    # read actual capture size
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)

    writer: cv2.VideoWriter | None = None
    recording = False
    record_path: Path | None = Path(record) if record else None

    print("=" * 56)
    print("  MANUAL TRACKING LIVE (性能模式)")
    print(f"  采集 {actual_w}x{actual_h}  推理最长边 {cur_infer}px")
    print("  把手伸到镜头前，特效贴在你手上")
    print("  Q退出 | S风格 | D暗底 | R录制 | F切换性能/画质")
    print("=" * 56)

    frame_index = 0
    t0 = time.perf_counter()
    fps_ema = 0.0
    last_t = t0
    # if a frame takes too long, skip display backlog by grabbing latest
    detect_ms_ema = 0.0
    draw_ms_ema = 0.0

    try:
        tracker = HandTracker(
            model,
            smooth=smooth,
            num_hands=2,
            infer_max_side=cur_infer,
            min_detection_confidence=0.45,
            min_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        with tracker:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("摄像头读帧失败，退出")
                    break

                # if falling behind, drop buffered frames (keep newest)
                # buffersize=1 helps; extra grab if detect was slow
                if detect_ms_ema > 45:
                    for _ in range(2):
                        cap.grab()

                if mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.perf_counter() - t0) * 1000)
                # ensure strictly increasing for MediaPipe VIDEO mode
                if frame_index > 0 and timestamp_ms <= getattr(run_live, "_last_ts", -1):
                    timestamp_ms = getattr(run_live, "_last_ts") + 1
                run_live._last_ts = timestamp_ms  # type: ignore[attr-defined]

                t_det0 = time.perf_counter()
                hands = tracker.process_bgr(frame, frame_index, timestamp_ms)
                t_det1 = time.perf_counter()
                out = renderer.render(frame, hands)
                t_draw1 = time.perf_counter()

                det_ms = (t_det1 - t_det0) * 1000
                draw_ms = (t_draw1 - t_det1) * 1000
                detect_ms_ema = det_ms if detect_ms_ema <= 1e-3 else detect_ms_ema * 0.85 + det_ms * 0.15
                draw_ms_ema = draw_ms if draw_ms_ema <= 1e-3 else draw_ms_ema * 0.85 + draw_ms * 0.15

                now = time.perf_counter()
                inst = 1.0 / max(now - last_t, 1e-3)
                last_t = now
                fps_ema = inst if fps_ema <= 1e-3 else (fps_ema * 0.85 + inst * 0.15)

                n_hands = len(hands.hands)
                mode = "PERF" if perf_mode else "QUAL"
                hud = (
                    f"FPS {fps_ema:4.1f}  det {detect_ms_ema:4.0f}ms  "
                    f"draw {draw_ms_ema:4.0f}ms  hands:{n_hands}  "
                    f"{styles[style_idx]}  {mode}"
                    f"{'  REC' if recording else ''}"
                )
                cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(
                    out,
                    hud,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 200) if recording else (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                if n_hands == 0:
                    cv2.putText(
                        out,
                        "Show your hands to the camera",
                        (10, out.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if recording and writer is not None:
                    writer.write(out)
                    cv2.circle(out, (out.shape[1] - 24, 16), 7, (0, 0, 255), -1)

                cv2.imshow(window_name, out)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    style_idx = (style_idx + 1) % len(styles)
                    renderer.style = styles[style_idx]
                    renderer.reset()
                    print(f"style → {styles[style_idx]}")
                if key in (ord("d"), ord("D")):
                    renderer.show_source = not renderer.show_source
                    print(f"show_source → {renderer.show_source}")
                if key in (ord("+"), ord("=")):
                    renderer.source_dim = float(min(1.0, renderer.source_dim + 0.05))
                if key in (ord("-"), ord("_")):
                    renderer.source_dim = float(max(0.05, renderer.source_dim - 0.05))
                if key in (ord("f"), ord("F")):
                    # toggle performance / quality
                    perf_mode = not perf_mode
                    if perf_mode:
                        cur_infer = 480
                        cur_trail = 4
                        renderer.fast = True
                        renderer.trail = cur_trail
                        tracker.infer_max_side = cur_infer
                        print("→ 性能模式 infer=480 trail=4")
                    else:
                        cur_infer = 720
                        cur_trail = 8
                        renderer.fast = False
                        renderer.trail = cur_trail
                        tracker.infer_max_side = cur_infer
                        print("→ 画质模式 infer=720 trail=8")
                    renderer.reset()
                if key in (ord("r"), ord("R")):
                    if not recording:
                        if record_path is None:
                            out_dir = Path(__file__).resolve().parents[2] / "output"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            record_path = out_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        h, w = out.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(record_path), fourcc, 30.0, (w, h))
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
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
