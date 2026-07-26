"""风格注册表与骨架拓扑 —— 两张"别处手抄一份就会漂"的表."""

import pytest

from manual_tracking.landmarks import CONNECTIONS, PALM_RING
from manual_tracking.renderer import STYLE_ALIASES, STYLES, canon_style


@pytest.mark.parametrize("name", STYLES)
def test_canonical_names_are_fixed_points(name):
    assert canon_style(name) == name


@pytest.mark.parametrize(("alias", "target"), sorted(STYLE_ALIASES.items()))
def test_aliases_resolve_into_the_registry(alias, target):
    assert target in STYLES, f"别名 {alias} 指向了不存在的风格 {target}"
    assert canon_style(alias) == target


def test_unknown_style_falls_back_to_mirror():
    """live 的 --style 是自由文本, 打错字要退回默认而不是崩在渲染循环里."""
    assert canon_style("没有这个风格") == "mirror"
    assert canon_style("") == "mirror"


def test_alias_never_shadows_a_real_style():
    """别名和正名撞车的话, 正名就被劫持了, 而且不报错."""
    assert not (set(STYLE_ALIASES) & set(STYLES))


def test_connections_reference_real_landmarks():
    for a, b in CONNECTIONS:
        assert 0 <= a < 21 and 0 <= b < 21, (a, b)
        assert a != b, "自环画不出骨头"


def test_connections_are_connected_and_deduped():
    """21 条边、无重边、从手腕走得到每一个关节.

    注意**不是树**: 掌根 0-5-9-13-17-0 是闭环(MediaPipe 官方拓扑就这样),
    21 节点 21 边 = 恰好一个环。按树去数会得到 20, 那是错的。
    """
    assert len(CONNECTIONS) == 21
    assert len({(min(a, b), max(a, b)) for a, b in CONNECTIONS}) == 21
    adj: dict[int, list[int]] = {i: [] for i in range(21)}
    for a, b in CONNECTIONS:
        adj[a].append(b)
        adj[b].append(a)
    seen, stack = {0}, [0]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert seen == set(range(21)), f"从手腕走不到: {set(range(21)) - seen}"


def test_palm_root_ring_is_closed():
    """掌根五点首尾相连成闭环 —— 断一条, 骨架的掌心就露个口子."""
    edges = {(min(a, b), max(a, b)) for a, b in CONNECTIONS}
    ring = (0, 5, 9, 13, 17)
    for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
        assert (min(a, b), max(a, b)) in edges, f"掌根环缺了 {a}-{b}"


def test_palm_ring_is_a_subset_of_landmarks():
    assert len(set(PALM_RING)) == len(PALM_RING) == 6
    assert all(0 <= v < 21 for v in PALM_RING)
