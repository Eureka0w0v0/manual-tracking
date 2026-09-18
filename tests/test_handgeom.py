"""handgeom: 手 → 少数几个低噪标量. 它是 renderer 和 glassbox 的共享叶子."""

import numpy as np
import pytest

from manual_tracking.handgeom import grip, orient, palm_center, pinch
from manual_tracking.landmarks import INDEX_MCP, PINKY_MCP
from synth import PALM, hand


def test_palm_center_inside_hand():
    h = hand(500, 400, pinch=False)
    c = palm_center(h)
    xy = h.points[:, :2]
    assert xy[:, 0].min() <= c[0] <= xy[:, 0].max()
    assert xy[:, 1].min() <= c[1] <= xy[:, 1].max()


def test_palm_center_follows_translation():
    """整只手平移 d, 掌心就该平移 d —— 不多不少."""
    a = palm_center(hand(300, 300, pinch=False))
    b = palm_center(hand(300 + 250, 300 - 80, pinch=False))
    assert np.allclose(b - a, (250, -80), atol=1e-3)


def test_orient_flips_sign_with_handedness():
    """同样的点位, 左右手朝向必须反号.

    这是 `sign = +1 if Right else -1` 那行的契约: 镜像的两只手做同一个动作,
    掌心/掌背的判定要相反, 否则 mirror 风格两半张纸会同亮同暗。
    """
    r = orient(hand(500, 400, pinch=False, handedness="Right"))
    left = orient(hand(500, 400, pinch=False, handedness="Left"))
    assert r == pytest.approx(-left)
    assert r != 0.0, "合成手不该正好侧立, 否则这条测了个寂寞"


def test_orient_saturates_within_unit_range():
    """掌宽放大 8 倍(手贴到镜头上)也不能冲出 [-1, 1] —— 下游按比例用它."""
    for palm in (40.0, PALM, PALM * 8):
        o = orient(hand(500, 400, pinch=False, palm=palm))
        assert -1.0 <= o <= 1.0, (palm, o)


def test_pinch_gap_tracks_thumb():
    """捏合 = 拇指尖贴食指尖; 松开 = 拉开. gap 是 mirror 判"压成细线"的依据."""
    _, _, _, shut = pinch(hand(500, 400, pinch=True))
    _, _, _, open_ = pinch(hand(500, 400, pinch=False))
    assert shut == pytest.approx(0.0, abs=1e-3)
    assert open_ > shut + 100


def test_grip_palm_width_matches_mcp_span():
    """掌宽必须是 |MCP5−MCP17| —— 单目深度线索全靠它, 换成别的点就不是同一个量."""
    h = hand(600, 350, pinch=False)
    *_, palm, _ = grip(h)
    expect = np.linalg.norm(h.points[INDEX_MCP, :2] - h.points[PINKY_MCP, :2])
    assert palm == pytest.approx(expect)
    assert palm > 0


def test_grip_scales_with_hand_size():
    """手离镜头近(掌宽翻倍) → 指弧展开量同比例翻倍, 比值才与远近无关."""
    _, _, spread1, palm1, _ = grip(hand(500, 400, pinch=False, palm=PALM))
    _, _, spread2, palm2, _ = grip(hand(500, 400, pinch=False, palm=PALM * 2))
    assert palm2 == pytest.approx(palm1 * 2)
    assert spread2 / spread1 == pytest.approx(2.0, rel=1e-3)
