"""Realtime webcam live loop.

Architecture (5 份稳定性审查后的时序设计):
- MediaPipe on a worker thread; latest-wins 邮箱: 每帧零拷贝提交,
  worker 醒来只取最新帧(检测率 = 1/max(D, P), 不再被 idle 门量化)
- 主线程按当前帧墙钟对最近两次检测结果做速度外推——
  消除"检测率<显示率"造成的角点阶跃/顿挫(乱飘主因)
- Main thread: camera → extrapolate → draw → imshow; never blocked
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .paths import default_output_dir, ensure_model
from .renderer import STYLES, VectorOverlayRenderer
from .tracker import FrameHands, HandPose, HandTracker

# 实拍底的压暗系数 —— **live 场景的策略, 所以归 live 管**(离线的那份在
# pipeline.SOURCE_DIM)。live 要亮一点: 手得看得见, 特效贴在自己手上才对得准。
# CLI 与 run_live() 都从这里取, 别在各自的默认参数里手抄 —— 早先 CLI 写 0.55
# 而 run_live() 的签名默认是 0.65, 命令行和直接 import 调用出来的画面亮度不
# 一样, 两边还都"看着没错"。
SOURCE_DIM = 0.55

FPS_WARMUP_FRAMES = 5  # 前 N 帧不计入 fps_ema(冷启动: 首帧 read/首次推理远慢于稳态)
EXTRAP_CAP_MS = 80.0  # 外推最多补偿这么多毫秒的检测延迟
EXTRAP_DAMP = 0.7  # 外推阻尼(压过冲)
EXTRAP_MAX_PX = 40.0  # 单点外推位移上限(翻转等乱抖速度下防甩飞)


def _builtin_camera_index() -> int:
    """本机内置摄像头的索引; 认不出来就退回 0.

    macOS 的"连续互通相机"会把 iPhone 也列成一个摄像头设备, 被选中时手机会
    亮屏接管——不是我们要的。system_profiler 的列举顺序与 AVFoundation(OpenCV
    用的后端)一致, 所以按顺序找第一个不是 iPhone/iPad 的设备即可。
    """
    try:
        raw = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
        items = json.loads(raw).get("SPCameraDataType", [])
    except Exception:
        return 0
    for i, dev in enumerate(items):
        blob = " ".join(str(v) for v in dev.values()).lower()
        if not any(k in blob for k in ("iphone", "ipad", "continuity", "desk view")):
            return i
    return 0


def _refps(path: Path, fps: float) -> bool:
    """按真实平均帧率重封装录像(不重编码)。没有 ffmpeg 就返回 False."""
    import shutil

    if shutil.which("ffmpeg") is None or not path.exists():
        return False
    tmp = path.with_suffix(".refps.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-r", f"{fps:.4f}", "-i", str(path), "-c", "copy", str(tmp)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        tmp.replace(path)
        return True
    except (subprocess.SubprocessError, OSError):
        tmp.unlink(missing_ok=True)
        return False



class _Recorder:
    """录制状态机: 开/关、写帧、停录后按真实帧率修容器.

    从 run_live 里抽出来的六个局部变量(writer/recording/record_path/rec_frames/
    rec_t0/fps_rec)——它们跟"实时循环"没关系, 只是恰好被 R 键触发。

    帧率有两道坎, 都踩过:
      1. 开录时 fps_ema 可能还没热身好(首帧含模型加载 133ms, 瞬时 FPS 只有 ~6,
         而它会播种 EMA)。实测启动第 1 帧按 R 会把容器帧率写成 10fps → 成片
         慢放 67%。所以未热身时直接退回 30。
      2. 容器帧率开录时就写死了, 但真实平均帧率只有录完才知道(低光下摄像头会
         自动降到 15fps → 成片快放 2x)。停录时实测偏差 >5% 就用 ffmpeg 重封装。
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path: Path | None = Path(path) if path else None
        self.writer: cv2.VideoWriter | None = None
        self.on = False
        self._frames = 0
        self._t0 = 0.0
        self._fps = 30.0

    def write(self, frame: np.ndarray) -> None:
        if self.on and self.writer is not None:
            self.writer.write(frame)
            self._frames += 1

    def toggle(self, frame: np.ndarray, fps_ema: float, warm: bool) -> None:
        self.stop() if self.on else self.start(frame, fps_ema, warm)

    def start(self, frame: np.ndarray, fps_ema: float, warm: bool) -> None:
        if self.path is None:
            out_dir = default_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            self.path = out_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        h, w = frame.shape[:2]
        self._fps = float(np.clip(fps_ema, 10.0, 60.0)) if warm else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            print("无法开始录制")
            return
        self.writer = writer
        self.on = True
        self._frames = 0
        self._t0 = time.perf_counter()
        print(f"REC start → {self.path}  ({self._fps:.1f} fps)")

    def stop(self) -> None:
        if not self.on:
            return
        self.on = False
        path, frames = self.path, self._frames
        self.release()
        dur = time.perf_counter() - self._t0
        real = frames / max(dur, 1e-6)
        # 太短的录制不做帧率修正: 样本不足时 real 会算出荒谬值(实测 20 帧
        # 瞬间写完 → 1359 fps), 拿它去改容器只会把好文件改坏。
        if path and dur >= REC_MIN_SEC and abs(real - self._fps) / self._fps > 0.05:
            fixed = _refps(path, real)
            print(
                f"REC 帧率修正 {self._fps:.1f} → {real:.1f} fps"
                f"{'' if fixed else ' (需要 ffmpeg, 已跳过)'}"
            )
        print(f"REC stop → {path}  ({frames} 帧)")
        self.path = None

    def release(self) -> None:
        """只放句柄, 不做帧率修正 —— finally 里用, 那时不该再跑子进程."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None



REC_MIN_SEC = 1.0  # 短于此的录制不做帧率修正(样本不足, 实测值会荒谬)
HUD_BAR_H = 34  # HUD 黑条高度(px)
HUD_REC_DOT = 7  # 录制红点半径(px)


def _draw_hud(
    out: np.ndarray,
    renderer: VectorOverlayRenderer,
    rec: "_Recorder",
    fps_ema: float,
    detect_ms: float,
    draw_ms_ema: float,
    hands: FrameHands,
    detector: "_AsyncHandDetector",
) -> None:
    """顶部状态条. 纯展示, 不改任何状态 —— 数值计算留在主循环里."""
    n_err = detector.errors()
    hud = (
        f"FPS {fps_ema:5.1f}  det {detect_ms:5.1f}ms  "
        f"draw {draw_ms_ema:4.1f}ms  hands:{len(hands.hands)}  "
        f"{renderer.style}  {'busy' if detector.busy() else 'idle'}"
        f"{f'  ERR{n_err}' if n_err else ''}"
        f"{'  REC' if rec.on else ''}"
    )
    if renderer.style == "cube":
        hud += f"  {renderer.cube.debug}"  # 操作提示画在立方体旁边, 不挤 HUD
    elif renderer.box_debug:  # 只有 screen 会写它, 不必判 style
        # psi = 盒子绕长轴的角, oL/oR = 双手掌面朝向(驱动 roll 的原始信号)
        b = renderer.box
        hud += (
            f"  expo {b.roll_expo:.1f}  lift {b.anchor_lift:.2f}  bias {b.depth_bias:.2f}"
            f"  resp {b.roll_resp:.2f}  cap {b.roll_max_rate:.0f}  {renderer.box_debug}"
        )
    cv2.rectangle(out, (0, 0), (out.shape[1], HUD_BAR_H), (0, 0, 0), -1)
    cv2.putText(
        out,
        hud,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 200) if rec.on else (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    if rec.on:
        cv2.circle(out, (out.shape[1] - 24, 16), HUD_REC_DOT, (0, 0, 255), -1)


def _open_camera(camera: int, width: int, height: int, fps: int = 0) -> cv2.VideoCapture:
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
                    # fps 是**请求值**, 设备按能力/光线自行决定实际值(低光会自动
                    # 降帧)。0 = 沿用 30 的历史默认; 实际值开机横幅里打出来。
                    cap.set(cv2.CAP_PROP_FPS, fps if fps > 0 else 30)
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
    latest-wins 邮箱: submit 永远接受并覆盖旧待处理帧, worker 醒来只处理最新
    一帧。保留最近两次结果(带时间戳)供主线程做速度外推。

    为什么不拷贝也安全: `cap.read()` 的缓冲区在**没人持有引用时会被复用**
    (实测: 持有引用则 8 次返回 8 个不同地址, 不持有则全部复用同一地址)。
    这里 `_pending` 和 worker 的局部 `frame` 接力持有引用, refcount 全程 ≥1,
    所以那块内存不会被下一次 read 覆盖。下游确实只读: renderer 三条路径都
    新建输出画布, HUD 画在副本上(实测 358 帧提交前后校验和 0 次改写)。
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
        self._errors = 0  # 检测异常累计(HUD 显示; 静默失败会伪装成"没检测到手")
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

    def errors(self) -> int:
        with self._lock:
            return self._errors

    def close(self) -> bool:
        """停 worker 并等它退出; 返回是否干净退出.

        调用方**必须**检查返回值: worker 卡住时 native landmarker 可能正在
        detect 里, 此时销毁它是未定义行为(实测 mediapipe 0.10.35 下没崩,
        但没有任何保证)。正常路径检测中位 6.9ms/p95 9.4ms, 离 3s 有数百倍余量。
        """
        with self._cond:
            self._running = False
            self._pending = None  # 丢掉待处理帧, 别让 worker 退出前又跑一次完整检测
            self._pending_meta = None
            self._cond.notify()
        self._thread.join(timeout=3.0)
        return not self._thread.is_alive()

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
            err: str | None = None
            try:
                # tracker.infer_max_side handles downscale + maps coords to full-res
                hands = self._tracker.process_bgr(frame, fi, ts)
            except Exception as exc:  # noqa: BLE001 - worker 不能死, 但要留下痕迹
                hands = FrameHands(index=fi, hands=[])
                # 静默吞掉的话, 检测持续炸只表现为 HUD 一直 hands:0, 分不清
                # "真没手"和"检测挂了"。首次打印 + 计数, HUD 上显示 errN。
                err = f"{type(exc).__name__}: {exc}"
            dt = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._res_prev = self._res_last
                self._res_last = (hands, ts)
                self._detect_ms = dt
                self._busy = False
                if err is not None:
                    if self._errors == 0:
                        print(f"检测异常(后续同类只计数): {err}")
                    self._errors += 1


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
    camera: int = -1,
    model_path: str | Path | None = None,
    style: str = "mirror",
    show_source: bool = True,
    source_dim: float = SOURCE_DIM,
    filter_on: bool = True,
    mirror: bool = True,
    width: int = 1920,
    height: int = 1080,
    fps: int = 0,
    infer_size: int = 0,
    window_scale: float = 1.0,
    record: str | Path | None = None,
    window_name: str = "Manual Tracking Live  |  Q退出 S风格 D暗底 R录制",
) -> None:
    if camera < 0:  # 自动: 挑本机内置摄像头, 绕开 iPhone 连续互通
        camera = _builtin_camera_index()
    model = ensure_model(model_path)

    # renderer 是风格名/别名的唯一规范化入口
    renderer = VectorOverlayRenderer(
        style=style,
        show_source=show_source,
        source_dim=source_dim,
    )
    styles = list(STYLES)
    style_idx = styles.index(renderer.style)

    # 先建 tracker 再开摄像头/开窗: 模型损坏之类的失败在这里抛, 此时还没有任何
    # 资源需要回收(否则摄像头会被占着直到进程退出)。
    tracker = HandTracker(
        model,
        filter_on=filter_on,
        num_hands=2,
        infer_max_side=infer_size,
        min_detection_confidence=0.45,
        min_presence_confidence=0.35,  # 实测: 更快的手部重入, 无副作用
        min_tracking_confidence=0.45,
    )
    try:
        cap = _open_camera(camera, width, height, fps)
    except Exception:
        tracker.close()
        raise
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    # 可缩放窗口: 默认 AUTOSIZE 会把窗口钉死在采集分辨率上, Retina 屏(3456x2234)
    # 下一个 1280x720 的窗口很小。WINDOW_NORMAL 允许拖拽边角任意放大, 初始尺寸
    # 按 window_scale 给。放大只影响显示, 不改采集/检测/录制分辨率。
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(
        window_name, int(actual_w * window_scale), int(actual_h * window_scale)
    )

    rec = _Recorder(record)

    print("=" * 56)
    print("  MANUAL TRACKING LIVE — 折纸镜面 / 彩色玻璃盒 / 悬浮立方体 / TD横幅")
    print("  拇指+食指捏纸；翻转一只手拧麻花；捏死压成细线")
    infer_txt = "全帧" if infer_size <= 0 else str(infer_size)
    fps_txt = f"{actual_fps:.0f}" if actual_fps > 0 else "?"
    print(
        f"  采集 {actual_w}x{actual_h} (req {width}x{height})"
        f"  帧率 {fps_txt} (req {fps if fps > 0 else 30})  推理边 {infer_txt}"
    )
    print(f"  窗口 {int(actual_w * window_scale)}x{int(actual_h * window_scale)} (可拖拽边角缩放)")
    print("  Q退出 | S风格 | D暗底 | R录制")
    print("  screen 调参: [ ] 翻转曲线  ; ' 挂多高  , . 旋转轴  7 8 角速度上限")
    print("               9 0 旋转跟手  - = 锚点跟手  < > 收起距离  { } 棱线")
    print("  cube  操作: 捏在盒上拖=转 | 双手捏住=移动+缩放+拧 | 张开手=炸开")
    print("              F 重力开关 | X 归位")
    print("  通用调参: g h 通透  o p 底亮度")
    print("=" * 56)

    frame_index = 0
    fps_ema = 0.0
    draw_ms_ema = 0.0
    last_ts = -1

    detector = _AsyncHandDetector(tracker)

    # 计时基准必须在模型加载(实测 133ms)之后取, 否则第一帧的瞬时 FPS 只有 ~6,
    # 而它会直接播种 fps_ema(α=0.15 要 ~30 帧才收敛)。启动 1 秒内按 R 录制,
    # fps_rec 就会冻结在这个坏值上 —— 实测第 1 帧按 R 得到 10fps, 成片慢放 67%。
    t0 = time.perf_counter()
    last_t = t0

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
            # 跳过冷启动帧: 首帧 cap.read()/首次推理都远慢于稳态, 拿它播种 EMA 会
            # 让 fps_ema 花 ~1 秒才爬到真值, 期间按 R 会把坏值冻进录制帧率。
            if frame_index >= FPS_WARMUP_FRAMES:
                fps_ema = inst if fps_ema <= 1e-3 else fps_ema * 0.85 + inst * 0.15

            # 录制在画 HUD 之前，成片不带黑条和状态文字
            rec.write(out)

            _draw_hud(out, renderer, rec, fps_ema, detect_ms, draw_ms_ema, hands, detector)

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
            if key in (ord("o"), ord("O")):
                renderer.source_dim = float(max(0.05, renderer.source_dim - 0.05))
            if key in (ord("p"), ord("P")):
                renderer.source_dim = float(min(1.0, renderer.source_dim + 0.05))
            if key in (ord("["), ord("]")):  # screen: 翻转曲线陡度(小=灵敏, 大=中心钝但稳)
                renderer.box.roll_expo = float(
                    np.clip(renderer.box.roll_expo + (0.1 if key == ord("]") else -0.1), 0.6, 3.0)
                )
                print(f"roll_expo → {renderer.box.roll_expo:.2f}")
            if key in (ord(";"), ord("'")):  # screen: 实时调盒子挂多高(掌心↔指弧)
                renderer.box.anchor_lift = float(
                    np.clip(renderer.box.anchor_lift + (0.05 if key == ord("'") else -0.05), 0.0, 1.4)
                )
                print(f"anchor_lift → {renderer.box.anchor_lift:.2f}")
            if key in (ord(","), ord(".")):  # screen: 旋转不动点 前面(0)↔体心(0.5)↔后面(1)
                renderer.box.depth_bias = float(
                    np.clip(renderer.box.depth_bias + (0.05 if key == ord(".") else -0.05), 0.0, 1.0)
                )
                print(f"depth_bias → {renderer.box.depth_bias:.2f}")
            if key in (ord("-"), ord("_"), ord("="), ord("+")):  # screen: 锚点跟手程度
                up = key in (ord("="), ord("+"))
                renderer.box.anchor_resp = float(
                    np.clip(renderer.box.anchor_resp + (0.05 if up else -0.05), 0.05, 1.0)
                )
                print(f"anchor_resp → {renderer.box.anchor_resp:.2f}")
            if key in (ord("g"), ord("G"), ord("h"), ord("H")):
                # 玻璃通透度(正向面 alpha; 内壁按 0.42 倍跟着走)。不能用 a/s ——
                # s 已经是切风格键, 同一次按键会先切风格再改 alpha。
                up = key in (ord("h"), ord("H"))
                renderer.face_alpha = float(
                    np.clip(renderer.face_alpha + (0.05 if up else -0.05), 0.25, 1.0)
                )
                renderer.back_alpha = round(renderer.face_alpha * 0.42, 3)
                print(f"face_alpha → {renderer.face_alpha:.2f} (back {renderer.back_alpha:.2f})")
            if key in (ord("{"), ord("}")):  # screen: 棱线粗细
                renderer.box_edge_w = int(
                    np.clip(renderer.box_edge_w + (1 if key == ord("}") else -1), 0, 8)
                )
                print(f"box_edge_w → {renderer.box_edge_w}{' (无缝)' if renderer.box_edge_w == 0 else ''}")
            if key in (ord("<"), ord(">")):  # screen: 双手多近才收起
                up = key == ord(">")
                renderer.box.gap_shut = float(
                    np.clip(renderer.box.gap_shut + (0.05 if up else -0.05), 0.05, 1.2)
                )
                print(f"gap_shut → {renderer.box.gap_shut:.2f}")
            if key in (ord("7"), ord("8")):  # screen: 角速度上限(度/帧), 小=更稳但更钝
                renderer.box.roll_max_rate = float(
                    np.clip(renderer.box.roll_max_rate + (5.0 if key == ord("8") else -5.0), 5.0, 90.0)
                )
                print(f"roll_max_rate → {renderer.box.roll_max_rate:.0f}°/帧")
            if key in (ord("9"), ord("0")):  # 旋转跟手程度(大=跟手, 小=顺滑)
                if renderer.style == "cube":  # cube: 拖 1px 转多少
                    renderer.cube.orbit_gain = float(
                        np.clip(
                            renderer.cube.orbit_gain * (1.15 if key == ord("0") else 1 / 1.15),
                            0.002,
                            0.06,
                        )
                    )
                    print(f"orbit_gain → {renderer.cube.orbit_gain:.4f}")
                else:
                    renderer.box.roll_resp = float(
                        np.clip(renderer.box.roll_resp + (0.05 if key == ord("0") else -0.05), 0.1, 0.9)
                    )
                    print(f"roll_resp → {renderer.box.roll_resp:.2f}")
            if key in (ord("f"), ord("F")):  # cube: 重力模式(扔出去走抛物线)
                renderer.cube.gravity = not renderer.cube.gravity
                print(f"gravity → {'ON (F 关闭)' if renderer.cube.gravity else 'OFF'}")
            if key in (ord("x"), ord("X")):  # cube: 位姿归位(飘出画面 / 转乱了时用)
                renderer.cube.reset()
                print("cube reset")
            if key in (ord("r"), ord("R")):
                rec.toggle(out, fps_ema, frame_index >= FPS_WARMUP_FRAMES + 10)

            frame_index += 1
    finally:
        # 释放顺序按"用户可感知的损失"排: 录像文件和摄像头最先, 因为 MediaPipe
        # graph 关闭万一抛异常, 后面的语句就都不执行了 —— 那会留下一个没写
        # moov box 的坏 mp4(实测 未 release 1.31MB 打不开 / release 后 1.56MB 正常)。
        # 每个 release 各自 try, 一个失败不拖累其余。
        for label, fn in (
            ("writer", rec.release),
            ("camera", cap.release),
            ("window", cv2.destroyAllWindows),
        ):
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - 清理阶段, 记录后继续
                print(f"释放 {label} 失败: {exc!r}")
        for _ in range(5):
            cv2.waitKey(1)
        # worker 卡住时不销毁 native landmarker(否则是未定义行为), 宁可泄漏
        # 一个句柄 —— 进程随后就退出了, OS 会回收。
        if detector.close():
            tracker.close()
        else:
            print("警告: 检测线程未在 3s 内退出, 跳过 landmarker 销毁")
