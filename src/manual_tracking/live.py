"""Realtime webcam live loop: camera → extrapolate → draw → imshow.

主线程只做这一条线, 且永不阻塞在检测上 —— MediaPipe 与速度外推整块住在
detect_service.py(那里有 latest-wins 邮箱和外推的时序设计)。这里剩下的是
摄像头、录制、HUD、键位, 以及把它们串起来的主循环。
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detect_service import AsyncHandDetector, extrapolate
from .paths import default_output_dir, ensure_model
from .renderer import STYLES, VectorOverlayRenderer
from .tracker import FrameHands, HandTracker

# 实拍底的压暗系数 —— **live 场景的策略, 所以归 live 管**(离线的那份在
# pipeline.SOURCE_DIM)。live 要亮一点: 手得看得见, 特效贴在自己手上才对得准。
# CLI 与 run_live() 都从这里取, 别在各自的默认参数里手抄 —— 早先 CLI 写 0.55
# 而 run_live() 的签名默认是 0.65, 命令行和直接 import 调用出来的画面亮度不
# 一样, 两边还都"看着没错"。
SOURCE_DIM = 0.55

# live 不传 --style 时进哪个风格 —— **唯一权威**。CLI 里 `live` 子命令的默认值、
# run_live() 的签名默认、全部一键入口(双击 / IDE ▶ / Cmd+Shift+B / 裸 run.sh, 它们
# 走 __main__.DEFAULT_ARGV = ["live"])都从这里取。早先一键路径写死 cube 而 `live`
# 子命令的 argparse 默认还是 mirror: `./run.sh` 进立方体、`./run.sh live` 进折纸
# 镜面, 差一个词两种结果, 而 README 第一条示例就是后者。
DEFAULT_STYLE = "cube"

FPS_WARMUP_FRAMES = 5  # 前 N 帧不计入 fps_ema(冷启动: 首帧 read/首次推理远慢于稳态)


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



REC_MIN_SEC = 1.0  # 短于此的录制不做帧率修正(样本不足, 实测值会荒谬)


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
        # --record 给的路径**只管第一次**录制: stop() 里把它清成 None, 之后再
        # 按 R 就自动命名。这是有意的 —— 一次会话里录第二条不该把第一条盖掉。
        # (原先没写这句, 看代码的人会把 stop() 里那行当成漏了复位而"修"回去。)
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
        self.path = None  # 下次按 R 自动命名, 不覆盖刚录完的这条(见 __init__)

    def release(self) -> None:
        """只放句柄, 不做帧率修正 —— finally 里用, 那时不该再跑子进程."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None



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
    detector: "AsyncHandDetector",
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


# ---- 键位 ----
# 全部键位字面量**只在这里出现一次**。原先是 17 条平铺 `if key in (ord(..), ..)`
# (一条 elif 都没有, 每帧 17 条全跑)加 50 处散落的 ord(), 而且一个键要在守卫和
# 分支体里各写一遍(`0.1 if key == ord("]") else -0.1`)。
#
# 代价已经兑现过: glassbox.py 的 gap_shut 注释一度写着 "live ( )", 真实绑定却是
# "< >" —— 键位说明散在窗口标题 / 开机横幅 / 分支体三处, 各改各的。现在横幅由
# _ACTIONS 与 _KNOBS 两张表生成(_banner_lines), 这类漂移在结构上不再可能。
#
# 撞键: 平铺 if 时代是**静默双触发**(一次按键跑两条分支), 现在建表时当场抛。


@dataclass
class _Tick:
    """一帧里 handler 会用到、而 renderer 身上没有的那几样.

    复用同一个实例, 只在真的按了键时才刷字段 —— 没按键的帧零开销。
    """

    renderer: VectorOverlayRenderer
    rec: "_Recorder"
    out: np.ndarray
    fps_ema: float = 0.0
    warm: bool = False
    quit: bool = False


_Handler = Callable[["_Tick", str], None]


def _off_style(t: "_Tick", ch: str, styles: tuple[str, ...]) -> bool:
    """当前风格用不上这个键 → 打一句提示并返回 True. styles 空 = 到处都管用.

    原先这些键在**任何**风格下都照改不误: cube 里按 `[` `]` 改的是 screen 的
    roll_expo, 终端还认真打印新值, 而画面纹丝不动; 反过来在 screen 里按 F
    切的是 cube 的重力, 一样没有任何反馈。README 的键位表早就分成 screen 和
    cube 两节了, 运行时却没有这一层 —— 只能靠记, 记错了还查不出来。
    """
    if not styles or t.renderer.style in styles:
        return False
    print(f"{ch} 只对 {'/'.join(styles)} 有用 (当前 {t.renderer.style})")
    return True


@dataclass(frozen=True)
class _Knob:
    """按一下 ±一步、钳进 [lo, hi] 的旋钮 —— 七个调参键的共同形状.

    dec/inc 各可写多个字符: `-`/`_` 与 `=`/`+` 是同一档的 shift 变体。
    owner="" 指 renderer 自己, 否则是它的子对象名(box / cube)。
    label="" 表示改完不打印(source_dim 原先就不打)。
    styles 空 = 所有风格; 否则只在列出的风格里生效(见 _off_style)。
    """

    dec: str
    inc: str
    owner: str
    attr: str
    step: float
    lo: float
    hi: float
    hint: str  # 开机横幅由它生成
    label: str = ""
    unit: str = ""
    fmt: str = "{:.2f}"
    styles: tuple[str, ...] = ()

    @property
    def keys(self) -> str:
        """这个旋钮占的全部字符 —— 建键表 / 查撞键时与 _Action 同一个接口."""
        return self.dec + self.inc

    def apply(self, t: "_Tick", ch: str) -> None:
        if _off_style(t, ch, self.styles):
            return
        obj = getattr(t.renderer, self.owner) if self.owner else t.renderer
        step = self.step if ch in self.inc else -self.step
        val = float(np.clip(getattr(obj, self.attr) + step, self.lo, self.hi))
        setattr(obj, self.attr, val)
        if self.label:
            print(f"{self.label} → {self.fmt.format(val)}{self.unit}")


# 除了底亮度, 其余六个旋钮改的都是 GlassBox 的字段 —— 只有 screen 读它们。
_SCREEN = ("screen",)
_KNOBS: tuple[_Knob, ...] = (
    _Knob("o", "p", "", "source_dim", 0.05, 0.05, 1.0, "o p 底亮度"),
    _Knob("[", "]", "box", "roll_expo", 0.1, 0.6, 3.0, "[ ] 翻转曲线", "roll_expo",
          styles=_SCREEN),
    _Knob(";", "'", "box", "anchor_lift", 0.05, 0.0, 1.4, "; ' 挂多高", "anchor_lift",
          styles=_SCREEN),
    _Knob(",", ".", "box", "depth_bias", 0.05, 0.0, 1.0, ", . 旋转轴", "depth_bias",
          styles=_SCREEN),
    _Knob("-_", "=+", "box", "anchor_resp", 0.05, 0.05, 1.0, "- = 锚点跟手", "anchor_resp",
          styles=_SCREEN),
    _Knob("<", ">", "box", "gap_shut", 0.05, 0.05, 1.2, "< > 收起距离", "gap_shut",
          styles=_SCREEN),
    _Knob("7", "8", "box", "roll_max_rate", 5.0, 5.0, 90.0, "7 8 角速度上限",
          "roll_max_rate", "°/帧", "{:.0f}", styles=_SCREEN),
)


def _act_quit(t: _Tick, ch: str) -> None:
    t.quit = True


def _act_style(t: _Tick, ch: str) -> None:
    """S: 循环切风格.

    下一个风格直接从 STYLES 算 —— 原先另存了一个 style_idx 跟着它走, 那是可
    推导的冗余状态: STYLES 是元组、canon_style 保证成员资格, 所以当前风格恒在
    表内, 而 renderer.style 的 setter 还要顺带清盒子状态, 它才是唯一事实源。
    """
    nxt = STYLES[(STYLES.index(t.renderer.style) + 1) % len(STYLES)]
    t.renderer.style = nxt
    print(f"style → {nxt}")


def _act_bg(t: _Tick, ch: str) -> None:
    t.renderer.show_source = not t.renderer.show_source
    print(f"show_source → {t.renderer.show_source}")


def _act_alpha(t: _Tick, ch: str) -> None:
    """g/h: 玻璃通透度(正向面 alpha; 内壁按 0.42 倍跟着走).

    不能用 a/s —— s 已经是切风格键, 同一次按键会先切风格再改 alpha。
    联动 back_alpha 是它进不了 _KNOBS 的唯一原因。
    """
    step = 0.05 if ch == "h" else -0.05
    t.renderer.face_alpha = float(np.clip(t.renderer.face_alpha + step, 0.25, 1.0))
    t.renderer.back_alpha = round(t.renderer.face_alpha * 0.42, 3)
    print(f"face_alpha → {t.renderer.face_alpha:.2f} (back {t.renderer.back_alpha:.2f})")


def _act_edge(t: _Tick, ch: str) -> None:
    """{ }: 盒子棱线粗细. 整数档 + 0 要额外提示"无缝", 所以不走 _KNOBS."""
    step = 1 if ch == "}" else -1
    t.renderer.box_edge_w = int(np.clip(t.renderer.box_edge_w + step, 0, 8))
    w = t.renderer.box_edge_w
    print(f"box_edge_w → {w}{' (无缝)' if w == 0 else ''}")


def _act_spin(t: _Tick, ch: str) -> None:
    """9/0: 旋转跟手程度 —— 两种风格调的不是同一个量, 所以按风格分派.

    cube 的 orbit_gain 跨一个数量级(0.002-0.06), 只能**乘性**步进;
    screen 的 roll_resp 就在 0.1-0.9 之间, 加性即可。
    """
    if t.renderer.style == "cube":
        k = 1.15 if ch == "0" else 1 / 1.15
        t.renderer.cube.orbit_gain = float(np.clip(t.renderer.cube.orbit_gain * k, 0.002, 0.06))
        print(f"orbit_gain → {t.renderer.cube.orbit_gain:.4f}")
    else:
        step = 0.05 if ch == "0" else -0.05
        t.renderer.box.roll_resp = float(np.clip(t.renderer.box.roll_resp + step, 0.1, 0.9))
        print(f"roll_resp → {t.renderer.box.roll_resp:.2f}")


def _act_gravity(t: _Tick, ch: str) -> None:
    t.renderer.cube.gravity = not t.renderer.cube.gravity
    print(f"gravity → {'ON (F 关闭)' if t.renderer.cube.gravity else 'OFF'}")


def _act_cube_reset(t: _Tick, ch: str) -> None:
    t.renderer.cube.reset()
    print("cube reset")


def _act_record(t: _Tick, ch: str) -> None:
    t.rec.toggle(t.out, t.fps_ema, t.warm)


_BOXY = ("screen", "cube")  # 两种有盒子的风格共用: 通透度 / 棱线 / 旋转跟手
_CUBE = ("cube",)


@dataclass(frozen=True)
class _Action:
    """形状各异、进不了 _KNOBS 的键: 一个或几个字符 → 同一个 handler.

    keys 里多个字符共用一个 handler(g/h、{/}、9/0 各是一对, q 与 Esc 是同义键),
    方向由 handler 自己看 ch 决定。hint 进开机横幅, styles 与 _Knob.styles 同一条
    规则、同一句提示(_off_style)。原先靠 _only() 包一层, 而横幅上这几个键的那两
    行是手抄的 —— _KNOBS 那半生成、这半手抄, 会漂的只剩这半。
    """

    keys: str
    fn: _Handler
    hint: str  # 开机横幅由它生成
    styles: tuple[str, ...] = ()

    def apply(self, t: "_Tick", ch: str) -> None:
        if not _off_style(t, ch, self.styles):
            self.fn(t, ch)


# 理由写在各自 handler 的 docstring 里。这里的顺序就是横幅里的顺序。
_ACTIONS: tuple[_Action, ...] = (
    _Action("q\x1b", _act_quit, "Q退出"),  # Esc 同义
    _Action("s", _act_style, "S风格"),
    _Action("d", _act_bg, "D暗底"),
    _Action("r", _act_record, "R录制"),
    _Action("90", _act_spin, "9 0 旋转跟手", _BOXY),
    _Action("gh", _act_alpha, "g h 通透", _BOXY),
    _Action("{}", _act_edge, "{ } 棱线", _BOXY),
    _Action("f", _act_gravity, "F 重力开关", _CUBE),
    _Action("x", _act_cube_reset, "X 归位", _CUBE),
)


def _hints(styles: tuple[str, ...]) -> str:
    """开机横幅里属于某一档风格的那行键位说明 —— 两张表一起生成, 不手抄.

    手抄过, 漂过: glassbox 的注释一度写着 "live ( )" 而实际绑的是 "< >"。
    **按风格分行**是为了和 _off_style 的运行时提示对得上 —— 横幅说这个键属于
    哪个风格, 按下去时就该得到同一个说法。
    """
    return "  ".join(e.hint for e in (*_ACTIONS, *_KNOBS) if e.styles == styles)


# 横幅的骨架: (行首标签, 这一行装哪一档风格的键)。键位表里出现了不在这里的
# styles 组合, test_live 会报 —— 那个键存在、能用、但没人知道。
_BANNER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ()),
    ("screen 调参: ", _SCREEN),
    ("screen/cube: ", _BOXY),
    ("cube  键位: ", _CUBE),
)


def _banner_lines() -> list[str]:
    """开机横幅的键位部分, 每档风格一行."""
    return [label + _hints(styles) for label, styles in _BANNER_GROUPS]


def _build_keymap() -> dict[str, _Handler]:
    """键 → handler. 撞键在这里当场炸 —— 平铺 if 时代它是静默双触发."""
    m: dict[str, _Handler] = {}
    for entry in (*_ACTIONS, *_KNOBS):
        for ch in entry.keys:
            if ch in m:
                raise AssertionError(f"键 {ch!r} 被绑了两次")
            m[ch] = entry.apply
    return m


def _key_char(key: int) -> str:
    """cv2.waitKey 的返回值 → 查表用的字符; 没按键/不可打印返回 "".

    字母的大小写在这里统一一次 —— 原先 10 个字母键各写一对 ord("s"), ord("S")。
    只对**字母**生效: ","/"<" 与 "."/">" 是两档不同的旋钮, lower() 不会把它们
    并到一起。
    """
    if key == 27:
        return "\x1b"
    return chr(key).lower() if 32 <= key < 127 else ""


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


def run_live(
    *,
    camera: int = -1,
    model_path: str | Path | None = None,
    style: str = DEFAULT_STYLE,
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
    keymap = _build_keymap()  # 撞键在这里炸, 不进主循环

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
    # 按键 handler 的上下文; out/fps_ema/warm 在真按了键的那一帧才刷
    tick = _Tick(renderer, rec, np.empty((1, 1, 3), np.uint8))

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
    for line in _banner_lines():
        print(f"  {line}")
    print("  cube  手势: 捏在盒上拖=转 | 双手捏住=移动+缩放+拧 | 张开手=炸开")
    print("=" * 56)

    frame_index = 0
    fps_ema = 0.0
    draw_ms_ema = 0.0
    last_ts = -1

    detector = AsyncHandDetector(tracker)

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
            hands = extrapolate(prev_res, last_res, timestamp_ms)

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

            # 没按键时 waitKey 返回 -1(&0xFF = 255), 查表 miss —— 直接跳过, 不
            # 像原先那样每帧空跑 17 条 if。
            handler = keymap.get(_key_char(key)) if key != 255 else None
            if handler is not None:
                tick.out = out  # handler 要用的当帧量, 只在真按了键时才刷
                tick.fps_ema = fps_ema
                tick.warm = frame_index >= FPS_WARMUP_FRAMES + 10
                handler(tick, _key_char(key))
                if tick.quit:
                    break

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
