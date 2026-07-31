"""GlassBox: 双手参数 → 刚体长方体的 8 个顶点.

「刚性」是这套参数化存在的全部理由(见 docs/GLASS_BOX_GEOMETRY.md §4: 旧的
「四指各握一角」架构里两条本该相等的棱比值中位 1.56)。所以这里逐条量它:
长方体在 3D 里造好再投影, 相机系里三组棱必须组内完全等长。
"""

import numpy as np
import pytest
from synth import PALM, hand

from manual_tracking.glassbox import GlassBox

# 顶点索引 = x*4 + u*2 + w, 所以沿三个轴的棱就是差 4 / 2 / 1 的顶点对
EDGE_GROUPS = {
    "长轴": [(0, 4), (1, 5), (2, 6), (3, 7)],
    "高": [(0, 2), (1, 3), (4, 6), (5, 7)],
    "进深": [(0, 1), (2, 3), (4, 5), (6, 7)],
}


def solved(box: GlassBox, left, right):
    geo = box.solve(left, right)
    assert geo is not None, "这对手应该能解出盒子"
    return geo


def test_returns_none_when_hands_overlap():
    """锚点几乎重合 → 长轴没方向, 几何本身无解, 必须返回 None 而不是 NaN."""
    box = GlassBox()
    h = hand(500, 400, pinch=False, tid=0)
    assert box.solve(h, hand(505, 400, pinch=False, tid=1)) is None


def test_box_is_rigid(two_hands):
    """三组棱各自组内等长 —— 长方体在 3D 里造好, 手只给参数, 不钉顶点."""
    _scr, cam, _f, _len = solved(GlassBox(), *two_hands)
    for name, edges in EDGE_GROUPS.items():
        lens = [float(np.linalg.norm(cam[a] - cam[b])) for a, b in edges]
        assert max(lens) - min(lens) < 1e-3, f"{name}四条棱不等长: {lens}"


def test_box_edges_are_mutually_perpendicular(two_hands):
    """三个轴两两垂直 —— 塌了就成了平行六面体, 面上的像素处理会被剪切拉花."""
    _scr, cam, _f, _len = solved(GlassBox(), *two_hands)
    axes = [cam[4] - cam[0], cam[2] - cam[0], cam[1] - cam[0]]
    axes = [a / np.linalg.norm(a) for a in axes]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert abs(float(axes[i] @ axes[j])) < 1e-4, f"轴 {i}/{j} 不垂直"


def test_focal_keeps_projection_finite(two_hands):
    """f 至少 1.1×(高+深) → f−z 恒为正, 双手贴近时不会被透视除爆."""
    _scr, cam, focal, _len = solved(GlassBox(), *two_hands)
    assert focal > float(cam[:, 2].max()) + 1e-3
    scr, *_ = solved(GlassBox(), *two_hands)
    assert np.isfinite(scr).all()


def test_length_tracks_hand_span():
    """两手拉开一倍, 盒长跟着翻倍 —— 盒子是挂在两手之间的, 不是固定尺寸."""
    a = solved(GlassBox(), hand(400, 400, pinch=False, tid=0), hand(700, 400, pinch=False, tid=1))
    b = solved(GlassBox(), hand(400, 400, pinch=False, tid=0), hand(1000, 400, pinch=False, tid=1))
    assert b[3] / a[3] == pytest.approx(2.0, rel=0.02)


def test_scale_invariance_to_camera_distance():
    """整个人靠近镜头(手更大、间距同比例更大) → 盒子等比放大, 形状不变.

    这是「一切按掌宽归一」能成立的前提: 远处的小盒子和近处的大盒子在参数空间
    里是同一个盒子。只放大手不拉开间距是**另一回事** —— 那会让两手的最近距离
    相对掌宽变小, 正确行为是触发收起(solve 返回 None), 不是长出个大盒子。
    """

    def dims(k: float) -> tuple[float, float, float]:
        pair = (
            hand(640 - 250 * k, 400, pinch=False, tid=0, palm=PALM * k),
            hand(640 + 250 * k, 400, pinch=False, tid=1, palm=PALM * k),
        )
        _scr, cam, _f, length = solved(GlassBox(), *pair)
        return length, float(np.linalg.norm(cam[0] - cam[2])), float(np.linalg.norm(cam[0] - cam[1]))

    for near, far in zip(dims(1.0), dims(2.0), strict=True):
        assert far / near == pytest.approx(2.0, rel=0.02)


def test_hands_closing_in_retracts_the_box():
    """手放大但不拉开 → 两手最近距离/掌宽 掉到阈值下 → 收起(None).

    上一条的反面。判据按掌宽归一, 所以"手变大"和"手靠近"对它是同一件事。
    """
    box = GlassBox()
    big = (
        hand(400, 400, pinch=False, tid=0, palm=PALM * 2),
        hand(900, 400, pinch=False, tid=1, palm=PALM * 2),
    )
    assert box.solve(*big) is None


def test_reset_clears_cross_frame_state(two_hands):
    """手离场后要当冷启动 —— 留着上一次的 ψ, 回来时盒子得边转边追."""
    box = GlassBox()
    solved(box, *two_hands)
    assert box._ema is not None
    box.reset()
    assert box._ema is None
    assert box._b0_prev is None
    assert box._anchor_prev is None
    assert box._psi_rate == 0.0


def test_reset_keeps_the_shut_hysteresis(two_hands):
    """reset() **必须留着**靠拢滞回 —— solve() 在收起动画走完后每帧都调它.

    清掉的话, 下一帧会按 gap_shut(而不是 gap_shut+GAP_HYST)重判, 滞回等于没有,
    双手停在阈值附近时盒子就会逐帧闪现。
    """
    box = GlassBox()
    solved(box, *two_hands)
    box._shut = True
    box.reset()
    assert box._shut is True


def test_hands_left_also_clears_the_shut_hysteresis(two_hands):
    """手全离场 ≠ 收起动画走完: 前者连滞回一起归零, 回来重新判.

    留着的话, 手在画面外"合拢"过, 回来第一帧会用宽阈值误判成收起。
    这条契约原先由 renderer 伸手写 box._shut 维持, 没有任何测试覆盖。
    """
    box = GlassBox()
    solved(box, *two_hands)
    box._shut = True
    box.hands_left()
    assert box._shut is False
    assert box._ema is None, "hands_left 也该清掉 reset 那套滤波历史"


def test_static_pose_converges_not_oscillates(two_hands):
    """手不动时 ψ 必须收敛到定值, 不能在两个值之间来回跳(那就是画面在抖)."""
    box = GlassBox()
    psis = [solved(box, *two_hands)[1] for _ in range(40)]
    last = np.stack(psis[-5:])
    assert float(np.abs(last - last[0]).max()) < 1e-3
