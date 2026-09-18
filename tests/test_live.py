"""live.py 里那几段**纯逻辑**的契约 —— 不开摄像头, 不进主循环.

`run_live()` 本身要摄像头, 测不了; 但它周围这几样都是可以离线钉死的, 而且
每一样都对应一次踩过的坑(注释里写着实测数字), 在这之前却一条断言都没有
(速度外推的那份跟着代码去了 test_detect_service.py):

  _Recorder     录制帧率的两道坎(未热身 / 停录后按实测重封装)
  _build_keymap 撞键当场炸(平铺 if 时代它是**静默双触发**)
  _key_char     字母统一小写, 但 ","/"<" 是两档不同的旋钮, 不能一起并掉

这些坏掉的时候, ruff / pytest / cube_check 全都照样绿, 症状要等到"录出来的
片子慢放 67%"或者"按一个键跑了两条分支"才浮出来 —— 正是最该有断言的地方。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from manual_tracking import live
from manual_tracking.renderer import STYLES, VectorOverlayRenderer

# ---------- _Recorder ----------


class _FakeWriter:
    """替掉 cv2.VideoWriter: 这里测的是录制状态机, 不是 OpenCV 的编码能力."""

    def __init__(self) -> None:
        self.frames = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        self.frames += 1

    def release(self) -> None:
        self.released = True


@pytest.fixture
def fake_writer(monkeypatch):
    made: list[_FakeWriter] = []

    def _make(*a, **k):
        made.append(_FakeWriter())
        return made[-1]

    monkeypatch.setattr(live.cv2, "VideoWriter", _make)
    return made


@pytest.fixture
def refps_calls(monkeypatch):
    """记录停录时有没有真去重封装, 以及用了什么帧率."""
    calls: list[float] = []
    monkeypatch.setattr(live, "_refps", lambda path, fps: calls.append(fps) or True)
    return calls


FRAME = np.zeros((8, 8, 3), np.uint8)


def test_cold_start_does_not_freeze_a_bad_fps_into_the_container(fake_writer, tmp_path):
    """未热身时不许拿 fps_ema 写容器帧率.

    首帧含模型加载(实测 133ms), 瞬时 FPS 只有 ~6 而它会播种 EMA —— 实测启动
    第 1 帧按 R, 容器帧率被写成 10fps, 成片慢放 67%。所以 warm=False 退回 30。
    """
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.start(FRAME, fps_ema=6.0, warm=False)
    assert rec._fps == 30.0


def test_a_warmed_up_fps_is_used_but_kept_in_a_sane_range(fake_writer, tmp_path):
    """热身之后用实测值, 但仍钳进 [10, 60] —— 容器帧率不接受荒谬值."""
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.start(FRAME, fps_ema=59.4, warm=True)
    assert rec._fps == pytest.approx(59.4)

    rec2 = live._Recorder(tmp_path / "b.mp4")
    rec2.start(FRAME, fps_ema=1359.0, warm=True)  # 实测 20 帧瞬间写完能算出这种值
    assert rec2._fps == 60.0


def test_only_written_frames_are_counted(fake_writer, tmp_path):
    """没开录时 write() 必须是空操作 —— 否则停录时的实测帧率会被稀释."""
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.write(FRAME)
    rec.start(FRAME, fps_ema=30.0, warm=True)
    for _ in range(5):
        rec.write(FRAME)
    assert rec._frames == 5 and fake_writer[0].frames == 5


def test_a_drifted_real_fps_triggers_a_remux(fake_writer, refps_calls, tmp_path):
    """容器帧率开录时就写死了, 真实平均帧率只有录完才知道(低光下摄像头会自动
    降到 15fps → 成片快放 2x)。偏差 >5% 就按实测重封装。"""
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.start(FRAME, fps_ema=30.0, warm=True)
    rec._frames = 120
    rec._t0 = time.perf_counter() - 2.0  # 2 秒 120 帧 = 真实 60fps, 容器写的是 30
    rec.stop()
    assert refps_calls == [pytest.approx(60.0, abs=0.5)]


def test_a_matching_real_fps_leaves_the_file_alone(fake_writer, refps_calls, tmp_path):
    """实测与容器一致时不要去动文件 —— 重封装是有损风险的操作, 没必要就别做."""
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.start(FRAME, fps_ema=30.0, warm=True)
    rec._frames = 60
    rec._t0 = time.perf_counter() - 2.0  # 2 秒 60 帧 = 30fps, 与容器相同
    rec.stop()
    assert refps_calls == []


def test_a_too_short_take_is_never_remuxed(fake_writer, refps_calls, tmp_path):
    """太短的录制不做修正: 样本不足时实测值会算出荒谬数(实测 20 帧瞬间写完
    → 1359 fps), 拿它去改容器只会把好文件改坏。"""
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.start(FRAME, fps_ema=30.0, warm=True)
    rec._frames = 20
    rec._t0 = time.perf_counter() - 0.1  # 远短于 REC_MIN_SEC
    rec.stop()
    assert refps_calls == []


def test_release_never_runs_a_subprocess(fake_writer, refps_calls, tmp_path):
    """finally 里走的是 release(): 那时不该再去起 ffmpeg 子进程."""
    rec = live._Recorder(tmp_path / "a.mp4")
    rec.start(FRAME, fps_ema=30.0, warm=True)
    rec._frames = 120
    rec._t0 = time.perf_counter() - 2.0
    rec.release()
    assert refps_calls == [] and fake_writer[0].released


# ---------- 键位表 ----------


def test_keymap_covers_every_declared_key():
    """_ACTIONS 与 _KNOBS 声明的键必须都能查到 handler."""
    km = live._build_keymap()
    for ch in live._ACTIONS:
        assert ch in km
    for kb in live._KNOBS:
        for ch in kb.dec + kb.inc:
            assert km[ch].__self__ is kb, f"键 {ch!r} 没落在它自己的旋钮上"


def test_a_double_bound_key_blows_up_at_build_time(monkeypatch):
    """撞键必须**当场炸**. 平铺 if 时代它是静默双触发: 一次按键跑两条分支,
    而且两边看着都对, 只有画面不对劲。"""
    clash = live._Knob("s", "\x00", "box", "roll_expo", 0.1, 0.6, 3.0, "撞 s 键")
    monkeypatch.setattr(live, "_KNOBS", (*live._KNOBS, clash))
    with pytest.raises(AssertionError, match="被绑了两次"):
        live._build_keymap()


def test_every_knob_points_at_an_attribute_that_exists():
    """旋钮表与 renderer 的字段名不许漂 —— 漂了要等哥哥按下那个键才炸."""
    r = VectorOverlayRenderer(style="screen")
    for kb in live._KNOBS:
        obj = getattr(r, kb.owner) if kb.owner else r
        assert hasattr(obj, kb.attr), f"{kb.hint}: {kb.owner or 'renderer'}.{kb.attr} 不存在"


def test_a_knob_steps_and_clamps():
    """按一下 ±一步, 到头就停在边界上(不越界、不回绕)."""
    kb = next(k for k in live._KNOBS if k.attr == "roll_expo")
    r = VectorOverlayRenderer(style="screen")
    t = live._Tick(r, live._Recorder(None), FRAME)
    r.box.roll_expo = 1.5
    kb.apply(t, kb.inc[0])
    assert r.box.roll_expo == pytest.approx(1.5 + kb.step)
    for _ in range(100):
        kb.apply(t, kb.inc[0])
    assert r.box.roll_expo == pytest.approx(kb.hi)
    for _ in range(200):
        kb.apply(t, kb.dec[0])
    assert r.box.roll_expo == pytest.approx(kb.lo)


# ---------- 键位按风格分流 ----------


def test_a_screen_only_knob_does_nothing_elsewhere(capsys):
    """cube 里按 `[` `]` 原先照样改 screen 的 roll_expo, 终端还打印新值 ——
    画面纹丝不动, 看着像"这个参数没用", 其实是按错了地方。"""
    r = VectorOverlayRenderer(style="cube")
    t = live._Tick(r, live._Recorder(None), FRAME)
    kb = next(k for k in live._KNOBS if k.attr == "roll_expo")
    before = r.box.roll_expo
    kb.apply(t, kb.inc[0])
    assert r.box.roll_expo == before
    assert "只对 screen 有用" in capsys.readouterr().out


def test_a_universal_knob_still_works_in_every_style():
    """底亮度是所有风格共用的, 不许被风格分流误伤."""
    kb = next(k for k in live._KNOBS if k.attr == "source_dim")
    for style in STYLES:
        r = VectorOverlayRenderer(style=style)
        t = live._Tick(r, live._Recorder(None), FRAME)
        before = r.source_dim
        kb.apply(t, kb.inc[0])
        assert r.source_dim > before, f"{style} 下底亮度键失灵了"


def test_gravity_and_reset_are_cube_only(capsys):
    """F/X 在 screen 下改的是 cube 的状态, 一样没有任何反馈."""
    km = live._build_keymap()
    r = VectorOverlayRenderer(style="screen")
    t = live._Tick(r, live._Recorder(None), FRAME)
    km["f"](t, "f")
    assert r.cube.gravity is False
    assert "只对 cube 有用" in capsys.readouterr().out
    r.style = "cube"
    km["f"](t, "f")
    assert r.cube.gravity is True


def test_glass_keys_cover_both_boxy_styles():
    """通透度/棱线是 screen 与 cube 共用的字段, 两边都得能调."""
    km = live._build_keymap()
    for style in ("screen", "cube"):
        r = VectorOverlayRenderer(style=style)
        t = live._Tick(r, live._Recorder(None), FRAME)
        before = r.face_alpha
        km["h"](t, "h")
        assert r.face_alpha > before, f"{style} 下 g/h 失灵"


def test_every_knob_shows_up_in_the_banner():
    """新加一个旋钮却忘了给它一行横幅, 结果是这个键存在、能用、但没人知道."""
    shown = live._knob_hints(()) + "  " + live._knob_hints(live._SCREEN)
    for kb in live._KNOBS:
        assert kb.hint in shown, f"旋钮「{kb.hint}」没出现在开机横幅里"


def test_a_second_take_never_overwrites_the_first(fake_writer, refps_calls, monkeypatch, tmp_path):
    """--record 给的路径只管第一次录制, 第二次按 R 自动命名."""
    monkeypatch.setattr(live, "default_output_dir", lambda: tmp_path)
    target = tmp_path / "take.mp4"
    rec = live._Recorder(target)
    rec.start(FRAME, fps_ema=30.0, warm=True)
    rec._frames, rec._t0 = 60, time.perf_counter() - 2.0
    rec.stop()
    rec.start(FRAME, fps_ema=30.0, warm=True)
    assert rec.path != target and rec.path.name.startswith("live_")


# ---------- _key_char ----------


def test_no_key_pressed_maps_to_nothing():
    """cv2.waitKey 没按键时返回 -1, &0xFF 之后是 255 —— 查表必须 miss."""
    assert live._key_char(255) == ""


def test_escape_is_spelled_out():
    assert live._key_char(27) == "\x1b"
    assert live._key_char(27) in live._ACTIONS


def test_letters_are_case_folded():
    """10 个字母键原先各写一对 ord("s")/ord("S"), 现在统一在这里折一次."""
    assert live._key_char(ord("S")) == live._key_char(ord("s")) == "s"


def test_shifted_punctuation_is_not_folded():
    """`,`/`<` 与 `.`/`>` 是**两档不同的旋钮**(旋转轴 vs 收起距离)。
    lower() 只对字母生效, 把它们并到一起就等于两个旋钮互相打架。"""
    assert live._key_char(ord("<")) == "<"
    assert live._key_char(ord(",")) == ","
    knobs = {ch: kb.attr for kb in live._KNOBS for ch in kb.dec + kb.inc}
    assert knobs["<"] != knobs[","], "这两个键必须落在不同的旋钮上"


def test_unprintable_keys_are_dropped():
    """方向键之类的返回值不该撞进查表."""
    assert live._key_char(0) == ""
    assert live._key_char(200) == ""
