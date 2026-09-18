"""速度外推的契约 —— 检测率低于显示率时, 靠它把 landmark 补到当前显示时刻.

这是"乱飘"的主要解法, 却是个**纯函数**: 喂两次带时间戳的检测结果, 要什么
摄像头都不需要。它有四条早退路径(没结果 / 只有一次 / 时间戳重合 / 显示时刻
更早)和两道护栏(时间封顶 EXTRAP_CAP_MS、单点位移限幅 EXTRAP_MAX_PX), 每一条
走错都只表现为"手抖了一下", 排查起来要命 —— 所以逐条钉住。

AsyncHandDetector 那半要起真线程 + 真 MediaPipe, 留给 tools/e2e_check.py。
"""

from __future__ import annotations

import pytest

from manual_tracking import detect_service as ds
from manual_tracking.landmarks import INDEX_TIP
from manual_tracking.tracker import FrameHands
from synth import hand


def _pair(x0: float, x1: float, t0: int, t1: int, tid: int = 0):
    """同一只手在 t0/t1 两次检测里的位置 → (prev, last) 两个带时间戳的结果."""
    return (
        (FrameHands(0, [hand(x0, 400, pinch=False, tid=tid)]), t0),
        (FrameHands(1, [hand(x1, 400, pinch=False, tid=tid)]), t1),
    )


def _cx(fh: FrameHands, i: int = 0) -> float:
    """一只手的参考横坐标(食指尖 —— synth 把它放在手中心正上方)."""
    return float(fh.hands[i].points[INDEX_TIP, 0])


def test_no_detection_yet_yields_an_empty_frame():
    """一次检测都还没回来: 给个空的, index=-1 —— 下游按"没有手"处理."""
    out = ds.extrapolate(None, None, 1000)
    assert out.hands == [] and out.index == -1


def test_a_single_detection_is_passed_through():
    """只有一次结果时无从算速度, 原样返回 —— 外推**永不比不外推更差**."""
    _, last = _pair(100.0, 100.0, 0, 33)
    assert ds.extrapolate(None, last, 66) is last[0]


def test_zero_gap_between_detections_is_not_extrapolated():
    """两次结果时间戳几乎相同: dt 做除数会炸出天文数字的速度, 直接放弃."""
    prev, last = _pair(100.0, 200.0, 33, 33)
    assert ds.extrapolate(prev, last, 100) is last[0]


def test_display_time_before_the_detection_does_not_rewind():
    """显示时刻早于最后一次检测(lead<=0): 不许往回倒放."""
    prev, last = _pair(100.0, 200.0, 0, 33)
    assert ds.extrapolate(prev, last, 30) is last[0]


def test_landmarks_advance_along_the_measured_velocity():
    """正常配对: 位移 = 帧间位移 × (lead/dt), lead 带 EXTRAP_DAMP 阻尼.

    30px/帧 的手, 检测间隔 33ms, 显示时刻落后检测 33ms:
        lead = min(33, CAP) × 0.7 = 23.1ms → 位移 30 × 23.1/33 = 21px
    """
    prev, last = _pair(100.0, 130.0, 0, 33)
    out = ds.extrapolate(prev, last, 66)
    lead = min(33.0, ds.EXTRAP_CAP_MS) * ds.EXTRAP_DAMP
    assert _cx(out) == pytest.approx(130.0 + 30.0 * lead / 33.0, abs=1e-3)
    # 外推**不许**就地改上一帧的结果 —— 那份还挂在 detector 的 _res_last 上,
    # 下一帧还要拿它当基准, 改了就会一帧比一帧飘得远。
    assert _cx(last[0]) == pytest.approx(130.0, abs=1e-3)


def test_extrapolation_lead_is_capped_in_time():
    """检测卡了很久(显示落后 500ms): lead 封在 EXTRAP_CAP_MS, 不按 500ms 外推.

    手速取 10px/帧 是**有意的**: 封顶后的位移 17px 落在 EXTRAP_MAX_PX 以内,
    所以这条测的确实是时间封顶。不封顶的话位移会是 106px, 那时先撞上的是
    位移限幅(40px), 两道护栏混在一起就分不清哪道在生效了。
    """
    prev, last = _pair(100.0, 110.0, 0, 33)
    far = ds.extrapolate(prev, last, 33 + 500)
    lead = ds.EXTRAP_CAP_MS * ds.EXTRAP_DAMP
    shift = 10.0 * lead / 33.0
    assert shift < ds.EXTRAP_MAX_PX, "用例失效: 位移限幅抢在时间封顶前面了"
    assert _cx(far) == pytest.approx(110.0 + shift, abs=1e-3)


def test_a_detection_jump_cannot_fling_the_hand_away():
    """误检瞬移 1000px: 单点位移被 EXTRAP_MAX_PX 限住, 不许被外推放大着甩出去."""
    prev, last = _pair(100.0, 1100.0, 0, 33)
    out = ds.extrapolate(prev, last, 66)
    assert _cx(out) - 1100.0 == pytest.approx(ds.EXTRAP_MAX_PX, abs=1e-3)


def test_a_hand_without_a_prior_track_is_left_alone():
    """这一帧新出现的手(上一次检测里没有这个 track_id): 没有速度可用, 原样传下去."""
    prev, last = _pair(100.0, 130.0, 0, 33, tid=0)
    fresh = FrameHands(1, [hand(700.0, 400, pinch=False, tid=9)])
    out = ds.extrapolate(prev, (fresh, 33), 66)
    assert _cx(out) == pytest.approx(700.0, abs=1e-3)


def test_unfiltered_hands_are_never_extrapolated():
    """--no-filter 时 track_id 全是 -1: 配不成对(它不是身份), 一律原样."""
    prev, last = _pair(100.0, 130.0, 0, 33, tid=-1)
    out = ds.extrapolate(prev, last, 66)
    assert _cx(out) == pytest.approx(130.0, abs=1e-3)
