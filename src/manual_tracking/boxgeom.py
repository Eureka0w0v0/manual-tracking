"""长方体的顶点序 + 弱透视投影 —— screen 与 cube 共用的那一份. 零依赖叶子.

`effects.BOX_FACES` 里每个面的 `verts` 都是按 **k = x*4 + u*2 + w** 写死的
整数, 而"哪个 k 对应哪个角"过去在三处各推导了一遍:

    glassbox.solve      三重循环 (-0.5, 0.5) × (-h/2, h/2) × (w_front, w_front-depth)
    FloatCube.project   三重循环 (-s, s) × (-s, s) × (s, -s)
    renderer._CORNERS   推导式   (-0.5, 0.5) × (-0.5, 0.5) × (0.5, -0.5)

三份**目前**是一致的, 维持它一致的只有"注释里写了三遍要一致"。而
`effects.Face` 的 docstring 讲的正是同一个病: 平行的表靠同一个下标索引,
错开了**不会抛异常**, 只是 glitch 跑到别的面上、HUD 报错误的面名。所以把
这个序收成一份, 三个调用点都来这里取。
"""

from __future__ import annotations

import numpy as np

# 顶点 k 的三个分量, 与 effects.Face 的 x/u/w 一一对应:
#   [:, 0] = x  沿长轴      -0.5 = 左端  +0.5 = 右端
#   [:, 1] = u  沿截面高轴  -0.5 = ui0   +0.5 = ui1
#   [:, 2] = w  沿进深轴    +0.5 = 前(靠观察者)  -0.5 = 后
#
# **u 只表示"哪一侧", 不表示"上"**: 两个调用方给它配的基向量朝向相反 ——
# glassbox 乘的是 b̂(冷启动挑成朝屏幕上方), floatcube 乘的是立方体自己的局部
# y 轴(而 p3[1] 直接当屏幕 y 用, 屏幕 y 朝下)。所以同一个 k, 在 screen 里落
# 在盒子上沿, 在 cube 的初始姿态里落在下沿。这不影响任何东西: 立方体会自由
# 翻滚, "顶面"在它身上本来就只是第四种像素处理的名字, 六个面的可见性一律由
# 运行时的 3D 外法线判定(renderer._split_faces), 不看这里的符号。
CORNER_SIGNS: np.ndarray = np.array(
    [[x, u, w] for x in (-0.5, 0.5) for u in (-0.5, 0.5) for w in (0.5, -0.5)],
    np.float32,
)
CORNER_SIGNS.setflags(write=False)  # 三个模块共用同一份, 谁就地改一下全都跟着错

_CAM_FLIP = np.array([1.0, -1.0, 1.0], np.float32)


def project_box(
    local: np.ndarray, origin: np.ndarray | tuple[float, float], focal: float
) -> tuple[np.ndarray, np.ndarray]:
    """局部点 → (屏幕坐标, 相机系坐标). 两种风格唯一的投影出口.

    `local` 是**屏幕对齐系**: x 右 / y 下(与画布同向) / z 朝观察者。相机系只
    差一个 y 取负(x 右 / y 上 / z 朝观察者) —— 面法线、Lambert 光照、可见性
    判定全在那个系里做。

    弱透视 f/(f−z): 后棱自动短于前棱, 两点透视是这么构造出来的, 不是画上去的。
    f−z 恒为正由调用方保证(glassbox 取 BOX_FOCAL×max(盒长, 高+深), floatcube
    取 CUBE_FOCAL×边长, 都在外接尺寸之上), 所以这里不再兜一次底 —— 兜了反而
    会把"焦距给错了"这种事悄悄糊过去。

    点数不限 8 个: 炸开视图按面拿 4 个角进来, 走的是同一条路径。
    """
    p = np.asarray(local, np.float32)
    cam = p * _CAM_FLIP
    scr = np.asarray(origin, np.float32) + p[:, :2] * (focal / (focal - p[:, 2]))[:, None]
    return scr, cam
