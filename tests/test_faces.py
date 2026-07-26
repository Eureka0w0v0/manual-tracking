"""BOX_FACES 的拓扑契约.

这张表是「六个面 → 六种像素处理」的唯一权威: renderer 的可见性判定、描边去重、
顶面 glitch 全靠它的下标对齐。写错一个顶点索引**不会抛异常**——盒子只是看起来
怪怪的, 而且怪在哪要盯着实时画面猜。下面几条把长方体的拓扑性质钉死。
"""

from collections import Counter

from manual_tracking.effects import BOX_FACES


def test_six_faces():
    assert len(BOX_FACES) == 6


def test_each_face_has_four_distinct_valid_verts():
    for f in BOX_FACES:
        assert len(f.verts) == 4, f"{f.tag} 面顶点数不是 4"
        assert len(set(f.verts)) == 4, f"{f.tag} 面有重复顶点: {f.verts}"
        assert all(0 <= v < 8 for v in f.verts), f"{f.tag} 面顶点越界: {f.verts}"


def test_all_eight_corners_used():
    """8 个角一个都不能漏 —— 漏了就是有个面贴错位置。"""
    used = {v for f in BOX_FACES for v in f.verts}
    assert used == set(range(8)), f"没用到的顶点: {set(range(8)) - used}"


def test_every_edge_shared_by_exactly_two_faces():
    """长方体 6 面 12 棱, 每条棱属于且仅属于 2 个面.

    这条最灵: 它只看拓扑不看坐标, 顶点索引写错时第一个炸的就是它。
    renderer 的描边去重(`drawn` 集合)也正是建立在这个性质上。
    """
    edges: Counter[tuple[int, int]] = Counter()
    for f in BOX_FACES:
        v = f.verts
        for a, b in zip(v, v[1:] + v[:1], strict=True):
            edges[(min(a, b), max(a, b))] += 1
    assert len(edges) == 12, f"棱数应为 12, 实为 {len(edges)}: {sorted(edges)}"
    bad = {e: c for e, c in edges.items() if c != 2}
    assert not bad, f"这些棱不是恰好被 2 个面共享: {bad}"


def test_face_edges_are_axis_aligned():
    """相邻顶点只能差一个轴位.

    顶点索引 = x*4 + u*2 + w, 所以一条棱的两端异或后必须是 4/2/1 之一。
    差两个轴 = 画了条对角线, 那个面会是折的。
    """
    for f in BOX_FACES:
        v = f.verts
        for a, b in zip(v, v[1:] + v[:1], strict=True):
            assert (a ^ b) in (1, 2, 4), f"{f.tag} 面的 {a}-{b} 是对角线, 不是棱"


def test_tags_unique_and_single_top():
    tags = [f.tag for f in BOX_FACES]
    assert len(set(tags)) == 6, f"HUD 靠 tag 区分朝向, 不能重名: {tags}"
    assert sum(f.is_top for f in BOX_FACES) == 1, "横条 glitch 只认一个顶面"


def test_each_face_has_its_own_effect():
    """六个面六种处理 —— 复用同一个 effect 对象说明表填串了。"""
    fx = [id(f.fx) for f in BOX_FACES]
    assert len(set(fx)) == 6, [f.tag for f in BOX_FACES]
