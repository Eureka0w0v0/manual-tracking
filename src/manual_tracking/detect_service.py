"""检测服务: MediaPipe 跑在 worker 线程, 主线程按显示时刻取外推后的结果.

从 live.py 整块抽出来的 —— 它跟"摄像头 / 录制 / HUD / 键位"没有任何耦合,
只是恰好和主循环住在一起。分开之后这一块能单独测(tests/test_detect_service.py),
live.py 也就只剩"读帧 → 画 → 显示 → 按键"一条线。

时序设计(5 份稳定性审查后的结论):
- latest-wins 邮箱: 每帧零拷贝提交, worker 醒来只取最新那一帧 —— 检测率
  = 1/max(D, P), 不再被 idle 门量化
- 主线程按当前帧的墙钟, 对最近两次检测结果做速度外推 —— 消除"检测率 <
  显示率"造成的角点阶跃/顿挫(乱飘主因)
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .tracker import FrameHands, HandPose, HandTracker

EXTRAP_CAP_MS = 80.0  # 外推最多补偿这么多毫秒的检测延迟
EXTRAP_DAMP = 0.7  # 外推阻尼(压过冲)
EXTRAP_MAX_PX = 40.0  # 单点外推位移上限(翻转等乱抖速度下防甩飞)


class AsyncHandDetector:
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


def extrapolate(
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
