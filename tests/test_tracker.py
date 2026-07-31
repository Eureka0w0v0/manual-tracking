"""轨迹层的纯逻辑契约: handedness 滞回.

这里**不碰 MediaPipe** —— `_Slot` 是个纯状态机, 拿它单独测就够。真实检测下
标签翻多勤见 `_Slot.vote` 的 docstring(实测 assets/sample.mp4 411 手帧翻 1 次,
且那一次是首帧判错)。
"""

import numpy as np

from manual_tracking.tracker import LABEL_FLIP_FRAMES, _Slot


def fresh(label: str = "Right") -> _Slot:
    """刚建立的轨迹: 首帧标签还没有信心."""
    return _Slot(sid=0, pts=np.zeros((21, 3), np.float32), ts_ms=0.0, label=label)


def warm(label: str = "Right") -> _Slot:
    """已经确认过一帧的轨迹 —— 滞回从这里才开始生效."""
    s = fresh(label)
    s.vote(label)
    return s


def test_first_frame_label_can_be_overturned_immediately():
    """首帧标签起手就差一票: MediaPipe 刚认出手时 handedness 最不准.

    实测 sample.mp4 的 tid5 裸标签是 Right×1 | Left×28。给首帧背书的话错标签
    会活满 LABEL_FLIP_FRAMES 帧, 比根本不锁存(只错 1 帧)还差。
    """
    assert fresh("Right").vote("Left") == "Left"


def test_new_slot_takes_the_observed_label():
    """新轨迹没有历史, 直接认当帧标签(响应优先, 推翻才保守)."""
    assert fresh("Left").vote("Left") == "Left"


def test_single_frame_flip_is_ignored():
    """确认过之后, 单帧翻转 = 误检, 不是换手。挡不住它 orient 就会整个反号."""
    s = warm("Right")
    assert s.vote("Left") == "Right"
    assert s.vote("Right") == "Right"


def test_sustained_flip_eventually_wins():
    """持续改判要跟上 —— 锁存是滤噪声, 不是把标签焊死."""
    s = warm("Right")
    seen = [s.vote("Left") for _ in range(LABEL_FLIP_FRAMES)]
    assert seen[:-1] == ["Right"] * (LABEL_FLIP_FRAMES - 1), "不该提前翻"
    assert seen[-1] == "Left", f"连续 {LABEL_FLIP_FRAMES} 帧后必须翻过来"


def test_dissent_counter_resets_on_agreement():
    """反对票必须**连续**才算数: 断续的噪声攒不出一次翻转."""
    s = warm("Right")
    for _ in range(LABEL_FLIP_FRAMES * 3):
        s.vote("Left")  # 一票反对
        s.vote("Right")  # 立刻被打断
    assert s.label == "Right"


def test_flip_is_stable_after_switching():
    """翻过去之后就是新标签的天下, 而且新标签立刻带信心(不会一帧回弹)."""
    s = warm("Right")
    for _ in range(LABEL_FLIP_FRAMES):
        s.vote("Left")
    assert s.label == "Left"
    assert s.vote("Right") == "Left", "翻回来同样要连帧确认"
