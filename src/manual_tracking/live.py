"""Realtime webcam: two-hand stretch frame fill (hand-frame-glitch structure)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .pipeline import default_model_path
from .renderer import VectorOverlayRenderer
from .tracker import HandTracker


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


def run_live(
    *,
    camera: int = 0,
    model_path: str | Path | None = None,
    style: str = "fabric",
    show_source: bool = True,
    source_dim: float = 0.75,
    trail: int = 0,
    smooth: float = 0.45,
    mirror: bool = True,
    width: int = 960,
    height: int = 540,
    infer_size: int = 480,
    record: str | Path | None = None,
    window_name: str = "Manual Tracking Live  |  Q退出 E强度 S风格 D暗底 R录制",
) -> None:
    """
    Keys:
      q / ESC  quit
      e        cycle fill effect inside stretch box
      s        cycle style frame/wire/outline
      d        toggle camera bg / black
      r        record
      + / -    brightness
    """
    model = Path(model_path) if model_path else default_model_path()
    if not model.exists():
        raise FileNotFoundError(f"model missing: {model}")

    styles = ["fabric", "track", "wire", "outline"]
    if style not in styles:
        style = "frame"
    style_idx = styles.index(style)

    renderer = VectorOverlayRenderer(
        style=styles[style_idx],
        show_source=show_source,
        source_dim=source_dim,
        trail=trail,
        fast=True,
        effect="energy",
    )

    cap = _open_camera(camera, width, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)

    writer: cv2.VideoWriter | None = None
    recording = False
    record_path: Path | None = Path(record) if record else None

    print("=" * 56)
    print("  MANUAL TRACKING LIVE — space fabric energy")
    print("  橙金空间布：手部光晕 + 指尖能量丝 + 掌间薄膜")
    print(f"  采集 {actual_w}x{actual_h}  推理边 {infer_size}")
    print("  两手入镜并拉开 — 橙金能量膜/丝线")
    print("  Q退出 | E切换 energy/calm/hot | S风格 | D暗底 | R录制")
    print("=" * 56)

    frame_index = 0
    t0 = time.perf_counter()
    fps_ema = 0.0
    last_t = t0
    detect_ms_ema = 0.0
    last_ts = -1

    try:
        with HandTracker(
            model,
            smooth=smooth,
            num_hands=2,
            infer_max_side=infer_size,
            min_detection_confidence=0.45,
            min_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        ) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("摄像头读帧失败，退出")
                    break

                if detect_ms_ema > 45:
                    for _ in range(2):
                        cap.grab()

                if mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.perf_counter() - t0) * 1000)
                if timestamp_ms <= last_ts:
                    timestamp_ms = last_ts + 1
                last_ts = timestamp_ms

                t_det0 = time.perf_counter()
                hands = tracker.process_bgr(frame, frame_index, timestamp_ms)
                t_det1 = time.perf_counter()
                out = renderer.render(frame, hands)
                t_draw1 = time.perf_counter()

                det_ms = (t_det1 - t_det0) * 1000
                draw_ms = (t_draw1 - t_det1) * 1000
                detect_ms_ema = det_ms if detect_ms_ema <= 1e-3 else detect_ms_ema * 0.85 + det_ms * 0.15

                now = time.perf_counter()
                inst = 1.0 / max(now - last_t, 1e-3)
                last_t = now
                fps_ema = inst if fps_ema <= 1e-3 else fps_ema * 0.85 + inst * 0.15

                n_hands = len(hands.hands)
                hud = (
                    f"FPS {fps_ema:4.1f}  det {detect_ms_ema:4.0f}ms  "
                    f"draw {draw_ms:4.0f}ms  hands:{n_hands}  "
                    f"{styles[style_idx]}/{renderer.effect}"
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

                if recording and writer is not None:
                    writer.write(out)
                    cv2.circle(out, (out.shape[1] - 24, 16), 7, (0, 0, 255), -1)

                cv2.imshow(window_name, out)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("e"), ord("E")):
                    name = renderer.next_effect()
                    print(f"energy → {name}")
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
                if key in (ord("r"), ord("R")):
                    if not recording:
                        if record_path is None:
                            out_dir = Path(__file__).resolve().parents[2] / "output"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            record_path = out_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        hh, ww = out.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(record_path), fourcc, 30.0, (ww, hh))
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
