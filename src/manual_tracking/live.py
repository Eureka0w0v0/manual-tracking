"""Realtime webcam live loop.

Architecture (5 份稳定性审查后的时序设计):
- MediaPipe on a worker thread; latest-wins 邮箱: 每帧零拷贝提交,
  worker 醒来只取最新帧(检测率 = 1/max(D, P), 不再被 idle 门量化)
- 主线程按当前帧墙钟对最近两次检测结果做速度外推——
  消除"检测率<显示率"造成的角点阶跃/顿挫(乱飘主因)
- Main thread: camera → extrapolate → draw → imshow; never blocked
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .pipeline import default_model_path
from .renderer import STYLES, VectorOverlayRenderer
from .tracker import FrameHands, HandPose, HandTracker

EXTRAP_CAP_MS = 80.0  # 外推最多补偿这么多毫秒的检测延迟
EXTRAP_DAMP = 0.7  # 外推阻尼(压过冲)
EXTRAP_MAX_PX = 40.0  # 单点外推位移上限(翻转等乱抖速度下防甩飞)


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
    latest-wins 邮箱: submit 永远接受并覆盖旧待处理帧(零拷贝——每次迭代的
    frame 都是新分配数组, 下游只读), worker 醒来只处理最新一帧。
    保留最近两次结果(带时间戳)供主线程做速度外推。
    """

    def __init__(self, tracker: HandTracker) -> None:
        self._tracker = tracker
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: np.ndarray | None = None
        self._pending_meta: tuple[int, int] | None = None  # idx, ts
        self._busy = False
        self._res_last: tuple[FrameHands, int] | None = None  # (hands, ts_ms)
        self._res_prev: tuple[FrameHands, int] | None = None
        self._detect_ms = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="hand-detect", daemon=True)
        self._thread.start()

    def busy(self) -> bool:
        with self._lock:
            return self._busy or self._pending is not None

    def submit(self, frame_bgr: np.ndarray, frame_index: int, timestamp_ms: int) -> None:
        with self._cond:
            self._pending = frame_bgr  # latest wins; 旧待处理帧直接被替换
            self._pending_meta = (frame_index, timestamp_ms)
            self._cond.notify()

    def latest_pair(
        self,
    ) -> tuple[tuple[FrameHands, int] | None, tuple[FrameHands, int] | None, float]:
        with self._lock:
            return self._res_prev, self._res_last, self._detect_ms

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
                self._busy = True
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
                self._res_prev = self._res_last
                self._res_last = (hands, ts)
                self._detect_ms = dt
                self._busy = False


def _extrapolate(
    prev: tuple[FrameHands, int] | None,
    last: tuple[FrameHands, int] | None,
    now_ms: int,
) -> FrameHands:
    """按 track_id 配对最近两次检测, 把 landmark 速度外推到当前显示时刻.

    消除检测率(15-30Hz)低于显示率(30fps)时的零阶保持阶跃——乱飘主因。
    任何配不上的情况原样返回最新结果, 永不比不外推更差。
    """
    if last is None:
        return FrameHands(index=-1, hands=[])
    fh1, t1 = last
    if prev is None:
        return fh1
    fh0, t0 = prev
    dt = float(t1 - t0)
    lead = min(float(now_ms - t1), EXTRAP_CAP_MS) * EXTRAP_DAMP
    if dt <= 1.0 or lead <= 0.0:
        return fh1
    by_id = {h.track_id: h for h in fh0.hands if h.track_id >= 0}
    out: list[HandPose] = []
    for h1 in fh1.hands:
        h0 = by_id.get(h1.track_id)
        if h0 is None:
            out.append(h1)
            continue
        disp = (h1.points[:, :2] - h0.points[:, :2]) * (lead / dt)
        m = float(np.max(np.linalg.norm(disp, axis=1)))
        if m > EXTRAP_MAX_PX:  # 翻转等乱抖速度下限幅, 防外推甩飞
            disp *= EXTRAP_MAX_PX / m
        pts = h1.points.copy()
        pts[:, :2] += disp
        out.append(HandPose(h1.handedness, h1.score, pts, h1.track_id))
    return FrameHands(index=fh1.index, hands=out)


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
    infer_size: int = 0,
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
    infer_txt = "全帧" if infer_size <= 0 else str(infer_size)
    print(f"  采集 {actual_w}x{actual_h} (req {width}x{height})  推理边 {infer_txt}")
    print("  Q退出 | S风格 | D暗底 | R录制")
    print("  screen 调参: [ ] 翻转灵敏度  ; ' 挂多高  , . 旋转轴  9 0 旋转跟手程度")
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
        min_presence_confidence=0.35,  # 实测: 更快的手部重入, 无副作用
        min_tracking_confidence=0.45,
    )
    detector = _AsyncHandDetector(tracker)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("摄像头读帧失败，退出")
                break

            # 取时贴近真实捕获时刻(外推的 dt 必须是真时间)
            timestamp_ms = int((time.perf_counter() - t0) * 1000)
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms

            if mirror:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            if abs(fw - width) > 8 or abs(fh - height) > 8:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                fh, fw = frame.shape[:2]

            # latest-wins 提交(零拷贝: frame 此后只读), 结果按显示时刻外推
            detector.submit(frame, frame_index, timestamp_ms)
            prev_res, last_res, detect_ms = detector.latest_pair()
            hands = _extrapolate(prev_res, last_res, timestamp_ms)

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
            busy = "busy" if detector.busy() else "idle"
            hud = (
                f"FPS {fps_ema:5.1f}  det {detect_ms:5.1f}ms  "
                f"draw {draw_ms_ema:4.1f}ms  hands:{n_hands}  "
                f"{styles[style_idx]}  {busy}"
                f"{'  REC' if recording else ''}"
            )
            if renderer.style == "screen" and renderer.box_debug:
                # psi = 盒子绕长轴的角, oL/oR = 双手掌面朝向(驱动 roll 的原始信号)
                hud += (
                    f"  roll {renderer.roll_gain:.1f}  lift {renderer.anchor_lift:.2f}"
                    f"  bias {renderer.depth_bias:.2f}  resp {renderer.roll_resp:.2f}"
                    f"  {renderer.box_debug}"
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
            if key in (ord("["), ord("]")):  # screen: 翻转倍率(1.0=1:1, 负值反向)
                renderer.roll_gain = float(
                    np.clip(renderer.roll_gain + (0.25 if key == ord("]") else -0.25), -3.0, 3.0)
                )
                print(f"roll_gain → {renderer.roll_gain:.2f}")
            if key in (ord(";"), ord("'")):  # screen: 实时调盒子挂多高(掌心↔指弧)
                renderer.anchor_lift = float(
                    np.clip(renderer.anchor_lift + (0.05 if key == ord("'") else -0.05), 0.0, 1.4)
                )
                print(f"anchor_lift → {renderer.anchor_lift:.2f}")
            if key in (ord(","), ord(".")):  # screen: 旋转不动点 前面(0)↔体心(0.5)↔后面(1)
                renderer.depth_bias = float(
                    np.clip(renderer.depth_bias + (0.05 if key == ord(".") else -0.05), 0.0, 1.0)
                )
                print(f"depth_bias → {renderer.depth_bias:.2f}")
            if key in (ord("9"), ord("0")):  # screen: 旋转跟手程度(大=跟手, 小=顺滑)
                renderer.roll_resp = float(
                    np.clip(renderer.roll_resp + (0.05 if key == ord("0") else -0.05), 0.1, 0.9)
                )
                print(f"roll_resp → {renderer.roll_resp:.2f}")
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
