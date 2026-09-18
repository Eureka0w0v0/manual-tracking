"""顶点序的单一权威 —— 以及"两个求解器说的是同一个盒子"这件事.

`effects.BOX_FACES` 里每个面的 `verts` 是四个写死的整数, 它们的含义完全由
`boxgeom.CORNER_SIGNS` 定义。这层对齐一旦错开, **什么都不会抛** —— 只是
glitch 跑到别的面上、HUD 报错误的面名、某个面的像素处理换了个位置, 要靠
肉眼在实拍里发现。test_faces.py 测的是 BOX_FACES 内部自洽(八个角都用到、
每条棱被两个面共享), 这里测的是它与角点符号、以及与两个求解器的对齐。
"""

from __future__ import annotations

import numpy as np
import pytest

from manual_tracking.boxgeom import CORNER_SIGNS, project_box
from manual_tracking.effects import BOX_FACES
from manual_tracking.floatcube import FloatCube
from manual_tracking.glassbox import GlassBox
from synth import hand

SHAPE = (720, 1280)
# 顶点索引 k = x*4 + u*2 + w, 所以沿三个轴的棱就是下标差 4 / 2 / 1 的顶点对
AXES = {4: "长轴", 2: "高轴", 1: "深轴"}


def _edges(pts: np.ndarray, stride: int) -> np.ndarray:
    """沿某一个轴的那 4 条棱(向量). 只取"低位相同"的配对, 免得数重."""
    return np.array([pts[k + stride] - pts[k] for k in range(8) if (k // stride) % 2 == 0])


def _assert_is_a_box(pts: np.ndarray, who: str) -> None:
    """三组棱各自平行等长 = 这 8 个点按 k 的语义构成一个长方体."""
    for stride, name in AXES.items():
        e = _edges(pts, stride)
        assert len(e) == 4, (who, name)
        lens = np.linalg.norm(e, axis=1)
        assert np.allclose(lens, lens[0], rtol=1e-4), f"{who} 的{name}四条棱不等长: {lens}"
        for v in e[1:]:
            cross = np.linalg.norm(np.cross(e[0], v))
            assert cross <= 1e-3 * lens[0] * lens[0], f"{who} 的{name}四条棱不平行"


# ---------- CORNER_SIGNS 本身 ----------


def test_corner_table_shape_and_range():
    assert CORNER_SIGNS.shape == (8, 3)
    assert CORNER_SIGNS.dtype == np.float32
    assert set(np.unique(CORNER_SIGNS).tolist()) == {-0.5, 0.5}


def test_corner_table_is_read_only():
    """三个模块共用同一份, 谁就地改一下另外两个跟着错 —— 而且不会报错."""
    with pytest.raises(ValueError):
        CORNER_SIGNS[0, 0] = 9.0


def test_index_decodes_as_x4_u2_w1():
    """k = x*4 + u*2 + w 是全项目的约定, 这里把它写成可执行的形式.

    w 那一位是**反的**: wi=0 取 +0.5(前, 靠观察者), wi=1 取 -0.5(后)。
    effects.BOX_FACES 的 verts 就是按这个排的, 别顺手"修正"成同向。
    """
    for k in range(8):
        xi, ui, wi = k // 4, (k // 2) % 2, k % 2
        assert CORNER_SIGNS[k, 0] == (0.5 if xi else -0.5)
        assert CORNER_SIGNS[k, 1] == (0.5 if ui else -0.5)
        assert CORNER_SIGNS[k, 2] == (-0.5 if wi else 0.5)


def test_the_eight_corners_are_a_cube():
    assert len({tuple(r) for r in CORNER_SIGNS.tolist()}) == 8
    _assert_is_a_box(CORNER_SIGNS, "CORNER_SIGNS")


# ---------- BOX_FACES 与角点符号的对齐 ----------


@pytest.mark.parametrize("face", BOX_FACES, ids=lambda f: f.tag)
def test_each_face_is_flat_in_corner_space(face):
    """一个面的 4 个角必须恰好共享一个坐标分量(那就是这个面所在的那一侧).

    这条直接绑住 Face.verts 的整数与 CORNER_SIGNS 的符号: 谁动了其中一边,
    面就不再是平的, 当场红 —— 而不是等到实拍时看见 glitch 长在底面上。
    """
    q = CORNER_SIGNS[list(face.verts)]
    const = [c for c in range(3) if len(set(q[:, c].tolist())) == 1]
    assert len(const) == 1, f"面「{face.tag}」在角点空间里不是一个平面: {q.tolist()}"
    for c in range(3):
        if c != const[0]:
            assert sorted(set(q[:, c].tolist())) == [-0.5, 0.5], f"面「{face.tag}」不是完整的一侧"


def test_opposite_faces_sit_on_opposite_sides():
    """六个面 = 三个轴 × 两侧, 不许两个面压在同一侧(那会有一侧永远画不到)."""
    sides = set()
    for face in BOX_FACES:
        q = CORNER_SIGNS[list(face.verts)]
        axis = next(c for c in range(3) if len(set(q[:, c].tolist())) == 1)
        sides.add((axis, float(q[0, axis])))
    assert len(sides) == 6, f"六个面只占了 {len(sides)} 侧: {sorted(sides)}"


# ---------- project_box ----------


def test_camera_frame_flips_y_only():
    """屏幕 y 朝下, 相机系 y 朝上 —— 只有这一位取负, x/z 原样."""
    local = np.array([[3.0, 5.0, 7.0]], np.float32)
    _, cam = project_box(local, (0.0, 0.0), 100.0)
    assert cam[0].tolist() == [3.0, -5.0, 7.0]


def test_nearer_points_project_larger():
    """弱透视 f/(f−z): 靠观察者(z 大)的点离画面中心更远 —— 后棱因此短于前棱."""
    local = np.array([[10.0, 0.0, -40.0], [10.0, 0.0, 0.0], [10.0, 0.0, 40.0]], np.float32)
    scr, _ = project_box(local, (0.0, 0.0), 200.0)
    assert scr[0, 0] < scr[1, 0] < scr[2, 0]


def test_origin_is_the_projection_center():
    """origin 就是不动点: z=0 的点原样落在 origin + 局部 xy 上."""
    scr, _ = project_box(np.array([[7.0, -3.0, 0.0]], np.float32), (640.0, 360.0), 500.0)
    assert scr[0].tolist() == [647.0, 357.0]


def test_output_stays_float32():
    """下游 cv2.fillPoly / _poly_window 要 float32; 悄悄升成 float64 会多一次拷贝."""
    scr, cam = project_box(CORNER_SIGNS * 100.0, (0.0, 0.0), 300.0)
    assert scr.dtype == np.float32 and cam.dtype == np.float32


# ---------- 两个求解器说的是同一个盒子 ----------


def test_floatcube_projects_a_rigid_box():
    """立方体转过一阵之后仍然是长方体(而不是被剪切成平行六面体)."""
    c = FloatCube()
    for i in range(20):
        c.update([hand(640 + i * 25, 456, pinch=True)], SHAPE)
    _, cam, _ = c.project(SHAPE)
    _assert_is_a_box(cam, "FloatCube.project")


def test_glassbox_projects_a_rigid_box():
    gb = GlassBox()
    geo = None
    for _ in range(12):  # 跑几帧让出现动画(ease)走完
        geo = gb.solve(
            hand(400, 400, pinch=False, tid=0, handedness="Left"),
            hand(900, 380, pinch=False, tid=1, handedness="Right"),
        )
    assert geo is not None
    _assert_is_a_box(geo[1], "GlassBox.solve")


def test_both_solvers_agree_on_which_axis_each_stride_means():
    """同一个下标差在两个求解器里必须指同一根轴.

    差 4 = 长轴(盒子最长的那根), 差 1 = 进深轴。要是哪个求解器把三重循环的
    嵌套顺序写反了, 这里的对应就会错位 —— 而 BOX_FACES 的 verts 是按这个
    语义写死的, 错位之后六种像素处理会集体换面, 却一条断言都不响。
    """
    c = FloatCube()
    c.update([], SHAPE)
    _, cube_cam, _ = c.project(SHAPE)

    gb = GlassBox()
    geo = None
    for _ in range(12):
        geo = gb.solve(
            hand(300, 400, pinch=False, tid=0, handedness="Left"),
            hand(1000, 400, pinch=False, tid=1, handedness="Right"),
        )
    assert geo is not None
    box_cam = geo[1]

    # 立方体三轴等长, 所以拿它测不出"哪根是长轴" —— 用 glassbox: 双手拉开
    # 700px, 长轴必然远长于高/深。两者只要在**同一个 stride 上**取到各自的
    # 长轴即可。
    box_len = {s: float(np.linalg.norm(_edges(box_cam, s)[0])) for s in AXES}
    assert max(box_len, key=box_len.get) == 4, f"glassbox 的长轴不在 stride 4 上: {box_len}"
    # 立方体那边则要求三轴等长(它是正方体), 顺带确认没有哪一轴退化成 0
    cube_len = np.array([np.linalg.norm(_edges(cube_cam, s)[0]) for s in AXES])
    assert cube_len.min() > 1.0
    assert np.allclose(cube_len, cube_len[0], rtol=1e-4), f"立方体三轴不等长: {cube_len}"
