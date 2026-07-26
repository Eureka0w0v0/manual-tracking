"""玻璃盒(screen 风格)的刚体几何 —— 双手参数 → 长方体 8 顶点.

这一层**不碰画布**: 只把两只手的姿态解成一个 3D 长方体并投影成屏幕坐标,
"每个面涂什么"归 effects.py, "涂在哪块画布上"归 renderer.py。

全部常数的标定过程见 docs/GLASS_BOX_GEOMETRY.md —— 改参数前先读那份,
里面记着每个数字是怎么从原片反解出来的, 以及试过但被否掉的方案。

跨帧状态(盒高/ψ/长轴深度的 EMA、ψ 角速度、截面基向量符号)全部封在 GlassBox
实例里, 手离场时由 reset() 一次清干净。
"""

from __future__ import annotations

import numpy as np

from .handgeom import grip
from .tracker import HandPose



# ---- tunables (哥哥要调效果基本都在这里) ----
MIN_SPAN_PX = 40.0  # 双手跨距小于此值不画特效
PINCH_SHUT_PX = 16.0  # 双手捏距都小于此值 → 侧视细线(实测捏合张开 10-34px@1920)
ROLE_HYST_PX = 25.0  # 左右角色互换需越过的掌心 x 差(防双手并拢时颜色频闪)
ORIENT_GAIN = 2.2  # 手掌朝向→明暗的灵敏度(越大翻手反应越猛)
BASE_B = 0.85  # 默认亮度(纸平摊时接近亮白)
B_SWING = 0.65  # 翻手带来的亮度摆幅
COOL_START = 0.7  # 亮度低于此值开始变冷
COOL_RATE = 1.8  # 变冷速度
MIRROR_SHIFT = 0.35  # 折起的面采样点外移量(跨距比例)
# 长方体参数化: 下面四个常数由原片帧 312 的四个角点 + 后/前棱比联立数值反解,
# 四角残差 RMS 0.3px(盒高 231px), 透视比 0.808 vs 实测 0.81。改动请重跑标定。
BOX_H_GAIN = 1.14  # 盒真高 / 指弧展开量(食指尖→小指尖); 反解 313/274
BOX_DEPTH_RATIO = 1.47  # 进深 / 盒真高; 反解 461/313
BOX_ROLL_BIAS = 0.830  # 静止时的截面转角 rad(=相机俯角, 47.6°); 保证看得见蓝顶面
# 掌面朝向 o → 绕长轴 roll。**不是线性 gain**, 而是两端收敛到 ±180°:
#     ψ = BIAS + (π·sign(o) − BIAS) · |o|^EXPO
# 为什么不用线性 gain: 背红面的投影面积对 ψ 是单峰的(峰在 180°, 那里红面
# 满屏), 线性映射只能在直线上挑落点, 顾此失彼——实测 gain 2.0 正端 ψ=162°
# 红面积 89.6%, 提到 3.0 后正端跑到 219° 越过峰值, 红面积反而掉到 56.4%,
# 同时可见面切换从 3.37 次/秒涨到 5.29 次/秒(颜色频闪)。
# expo 把两端直接钉在峰值 ±180° 上, 且 o=0 时恰好回到 BIAS(静止外观逐像素
# 不变)。dψ/do = (π − sign(o)·BIAS)·EXPO·|o|^(EXPO−1) > 0 恒成立 → 严格单调。
# EXPO 越大中心越钝、静止越稳(1.0 线性 / 1.5 平衡 / 2.0 最稳)。
BOX_ROLL_EXPO = 1.5
BOX_AXIS_Z_GAIN = 1.2  # 掌宽比 → 长轴深度分量; 一只手往前伸盒子就指向镜头(0 = 长轴锁在像平面)
BOX_ANCHOR_LIFT = 0.85  # 锚点在 掌心(0)↔指弧中点(1) 之间的位置; 帧 312 反解最优 0.95
BOX_DEPTH_BIAS = 0.5  # 锚点在进深方向的位置, 同时也是绕长轴旋转的不动点:
#   0   = 前面压在手上(原片位置), 但翻转时是"前棱当轴甩", 盒子整体堆在手后面
#   0.5 = 体心落在手上 → 盒子跟着手整体转(手感对), 深度上以手为中心
#   1   = 后面压在手上
BOX_FOCAL = 1.10  # 弱透视焦距 = 该值 × max(盒长, 高+深)
BOX_SMOOTH = 0.6  # 盒高/长轴深度的轻 EMA(慢变量, 固定系数够用)
# ψ 的速度自适应 EMA(与 tracker.py 的 landmark 级 One Euro 同源思路, 但这里
# 按"帧"而非墙钟计时, 保证离线渲染可复现)。固定 EMA 实测在 0.4s 快速翻手中
# 恒定落后 31°、手停后还要 5 帧追上——那就是"太滑不跟手"的来源。
BOX_ROLL_RESP = 0.40  # 静止时每帧吸收的新值比例(= 旧固定 EMA 的 1−0.6, 抖动不劣化)
BOX_ROLL_RESP_GAIN = 1.35  # 每 (rad/帧) 角速度把上面这个值抬高多少
BOX_ROLL_RESP_MAX = 0.92  # 上限, 留一点滤波防单帧误检直接甩过去
BOX_RATE_SMOOTH = 0.5  # 角速度估计自身的 EMA(不平滑的话增益会跟着抖)
# ψ 的角速度硬限幅(度/检测帧)。_orient 在饱和区会被噪声掀翻符号——实测原片
# 里出现过"一只手 o 从 −1.00 一帧跳到 +0.74 而另一只手没动", 两手平均后 ψ
# 单帧弹 199.5°(p99 117.6°, >90° 的帧占 1.6%)。手物理上不可能 33ms 转 180°,
# 所以超过这个速率的一律是误检。20°/帧 @30fps = 600°/s, 比最快的翻腕还快
# 一倍有余, 不会削掉真实动作。限幅作用在滤波之后, 直接约束"看到的"角速度。
BOX_ROLL_MAX_RATE = 20.0
# 锚点(两手挂盒点)的速度自适应滤波。它原先是**唯一没有盒级平滑的量** —— ψ/盒高/
# 长轴深度都有 EMA, 锚点却裸传, 于是 landmark 噪声直通 8 个顶点: 实测顶点帧间
# 位移中位 62.7px、p90 176.6px, 而同期 ψ 只有 4.46°、盒高 3.88px。
# 用二阶差分把"真实运动"和"抖动"分开后, 锚点的抖动分量中位 13.8px/帧 —— 就是
# 它撑起了顶点的大部分抖。这里用与 ψ 同源的自适应 EMA: 手不动时重滤波, 手快速
# 移动时自动放开, 不引入拖影。
BOX_ANCHOR_RESP = 0.25  # 静止时每帧吸收的新锚点比例(越小越稳、越钝)
# 每 px/帧 速度把上面这个值抬高多少。这个数很敏感: 原片手速中位就有 28px/帧
# (=833px/秒), 用 0.012 时 k 中位被抬到 0.68、23% 的帧顶到上限, 滤波形同虚设。
BOX_ANCHOR_RESP_GAIN = 0.001
BOX_ANCHOR_RESP_MAX = 0.95  # 上限; 留一点滤波, 单帧误检不会整盒瞬移
BOX_ANCHOR_RATE_SMOOTH = 0.5  # 锚点速度估计自身的 EMA
# 双手碰到一起 → 收起盒子; 拉开 → 重新出现。
# 判据是**两手之间的最近距离**(21x21 个点对取最小), 按掌宽归一化。
#
# 为什么不用锚点距离: 锚点在指弧附近, 两手贴合时锚点仍隔着约一个手宽 ——
# 实测原片双手最接近的那一帧, 锚点距离还有 0.72 个掌宽, 而最近点只剩 0.03。
# 拿锚点距离当"碰上了"的判据, 阈值要设到 1.1 以上, 那时手其实还离得挺远,
# 反过来真贴上了却因为锚点没到阈值而不触发。最近距离直接对应"碰到"这件事。
#
# 按掌宽归一 → 与手离镜头远近无关(近处手大, 像素间距也大, 比值不变)。
# 阈值的直观换算(掌宽 200px 时): 0.12≈24px 指头挨上 / 0.35≈70px 差一指多宽 /
# 0.55≈110px 差半个手掌。调大 = 手还没真碰上就收起, 手势更省力。
GAP_SHUT = 0.35  # 最近距离 / 掌宽 低于它 → 收起
GAP_HYST = 0.40  # 出现阈值比收起阈值高这么多; 拉开间距才防得住频闪


class GlassBox:
    """把双手姿态解成刚体长方体. 五个公开属性是 live 的实时旋钮."""

    def __init__(self) -> None:
        self.roll_expo = BOX_ROLL_EXPO  # live [ ] : 翻转曲线陡度
        self.anchor_lift = BOX_ANCHOR_LIFT  # live ; ' : 盒子挂多高
        self.depth_bias = BOX_DEPTH_BIAS  # live , . : 旋转不动点/进深中心
        self.roll_resp = BOX_ROLL_RESP  # live 9 0 : 旋转跟手程度
        self.roll_max_rate = BOX_ROLL_MAX_RATE  # live 7 8 : 角速度上限
        self.anchor_resp = BOX_ANCHOR_RESP  # live - = : 锚点跟手程度
        self.gap_shut = GAP_SHUT  # live ( ) : 双手多近才收起(出现阈值跟着走)
        self.debug = ""  # HUD 用: 当前 ψ / 双手掌朝向 / 长轴深度
        self._ema: tuple[float, float, float] | None = None  # (盒高, ψ, 长轴深度比)
        self._psi_rate = 0.0  # ψ 的角速度估计(rad/帧), 驱动自适应滤波
        self._b0_prev: np.ndarray | None = None  # 上帧的截面"上"轴(符号帧间传播)
        self._anchor_prev: np.ndarray | None = None  # 上帧滤波后的两个锚点 (2,2)
        self._anchor_rate = 0.0  # 锚点速度估计(px/帧)
        self._shut = False  # 双手是否已靠拢(滞回状态; 见 solve 里的 SPAN_*)

    def reset(self) -> None:
        """清跨帧状态(手离场/合拢/切风格). 下一帧当作冷启动.

        **不含左右手角色滞回** —— 那个是三种风格共用的, 归 renderer 管。
        """
        self._ema = None
        self._psi_rate = 0.0
        self._b0_prev = None
        self._anchor_prev = None
        self._anchor_rate = 0.0
        self.debug = ""

    def anchor(self, hand: HandPose) -> np.ndarray:
        """一只手的锚点(掌心↔指弧中点插值). 双手合拢画种子点时也要用."""
        c, f, _s, _p, _o = grip(hand)
        return c + (f - c) * self.anchor_lift

    def solve(
        self, left: HandPose, right: HandPose
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """双手参数 → 刚体长方体, 返回 (屏幕 8 顶点, 相机系 8 顶点, 焦距, 盒长).

        顶点索引 = x*4 + u*2 + w, 与 effects.BOX_FACES 一致。

        长轴是**真 3D 向量**: 屏幕位移来自两掌心连线, 深度分量来自两手掌宽比
        (投影尺寸 ∝ 1/距离, 一只手往前伸盒子就指向镜头)。所以盒子能朝任意
        方向, 端面也能露出来——长轴锁在像平面时端面结构上永不可见。
        截面两轴 (b̂ 屏幕上 / ĉ 朝观察者) 由长轴正交化得到, 再绕长轴转 ψ:
            b̂ = b̂₀cosψ + ĉ₀sinψ      ĉ = −b̂₀sinψ + ĉ₀cosψ
        ψ = 相机俯角 + 用户 roll——两者同轴, 合成一个角, 不需要翻面状态机。
        顶点 = t·长轴 + u·b̂ + w·ĉ, 再按 f/(f−z) 弱透视投影。

        w 的原点(= 锚点 = 手)就是旋转的不动点, 位置由 depth_bias 决定:
        默认 0.5 把体心放在手上, 盒子跟着手整体转; 取 0 会把前面钉在手上,
        翻转就变成"拿前棱当轴甩"、盒子整体堆在手后面(实测手感不对)。

        刚性是构造出来的: 8 个顶点由 4 个标量(长/高/深/ψ)+一个 3D 轴生成,
        恒为长方体。旧版 8 个角各自独立跟指尖, 实测"本应等长"的两条深度棱
        中位差 1.56×(p90 2.94×), 楔形在那个架构里无解。
        """
        cL, fL, sL, pL, oL = grip(left)
        cR, fR, sR, pR, oR = grip(right)
        # 碰到一起就收起。用**未滤波的原始 landmark**算最近距离 —— 锚点带了
        # 自适应 EMA, 快速合拢时它滞后于真手, 会漏判。
        a, b = left.points[:, :2], right.points[:, :2]
        gap = float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())
        gap_r = gap / max((pL + pR) * 0.5, 1e-3)
        # 出现阈值 = 收起阈值 + 固定间距, 这样实时调一个值滞回宽度不变
        self._shut = gap_r < (self.gap_shut + GAP_HYST if self._shut else self.gap_shut)
        if self._shut:
            keep = self._shut  # reset 会清滤波历史, 但滞回状态必须留着
            self.reset()
            self._shut = keep
            return None
        t = self.anchor_lift  # 0=掌心(偏低) 1=指弧中点(原片高度)
        anc = np.stack([cL + (fL - cL) * t, cR + (fR - cR) * t])  # (2,2) 左右锚点
        # 速度自适应滤波: 手不动时重滤(去抖), 手快速移动时自动放开(不拖影)。
        # 两只手共用一个速度估计——它们本来就一起动, 分开估会让快的那只带偏慢的。
        if self._anchor_prev is not None:
            step = anc - self._anchor_prev
            v = float(np.linalg.norm(step, axis=1).max())
            self._anchor_rate = (
                self._anchor_rate * BOX_ANCHOR_RATE_SMOOTH + v * (1.0 - BOX_ANCHOR_RATE_SMOOTH)
            )
            k = min(
                self.anchor_resp + BOX_ANCHOR_RESP_GAIN * self._anchor_rate, BOX_ANCHOR_RESP_MAX
            )
            anc = self._anchor_prev + step * k
        self._anchor_prev = anc
        gL, gR = anc[0], anc[1]
        span_v = gR - gL
        span = float(np.linalg.norm(span_v))
        if span < MIN_SPAN_PX:  # 兜底: 锚点几乎重合, 几何本身没法解
            self.reset()
            return None

        # 长轴深度分量: 掌宽比 → 单目深度线索(哪只手更近就更大)。
        # 存 EMA 的是**比值**不是像素: 比值是无量纲的, 而像素随 span 变——快速
        # 收手时旧的大 dz 会被带进来, 实测 span 900→60 时 dz/span 从 1.08 冲到
        # 4.03(长轴偏出像平面 76°), 突破了 BOX_AXIS_Z_GAIN 这个上界。
        dzr_raw = BOX_AXIS_Z_GAIN * (pR - pL) / max(pR + pL, 1e-3)
        height_raw = BOX_H_GAIN * (sL + sR) * 0.5
        o = (oL + oR) * 0.5
        psi_raw = BOX_ROLL_BIAS + (np.pi * np.sign(o) - BOX_ROLL_BIAS) * abs(o) ** self.roll_expo
        if self._ema is None:
            height, psi, dzr = height_raw, psi_raw, dzr_raw
        else:
            a = BOX_SMOOTH
            height = self._ema[0] * a + height_raw * (1.0 - a)
            dzr = self._ema[2] * a + dzr_raw * (1.0 - a)
            # ψ: 转得越快滤波越松 → 静止不抖, 快速翻手仍然跟手
            prev = self._ema[1]
            # 走最短弧: ψ 是角度, ±π 是同一姿态。线性插值会绕远路——实测原片
            # 186 个帧里有 3 帧 |ψ_raw − prev| > 180°(最大 216.9°, 最短弧只要
            # 143.1°), 叠加限幅后要多花 ~4 帧才转到位。
            err = float(np.arctan2(np.sin(psi_raw - prev), np.cos(psi_raw - prev)))
            self._psi_rate = self._psi_rate * BOX_RATE_SMOOTH + abs(err) * (1.0 - BOX_RATE_SMOOTH)
            k = min(self.roll_resp + BOX_ROLL_RESP_GAIN * self._psi_rate, BOX_ROLL_RESP_MAX)
            cap = np.radians(self.roll_max_rate)  # 硬限幅: 挡掉 _orient 掀翻符号造成的弹飞
            psi = prev + float(np.clip(err * k, -cap, cap))
        self._ema = (height, psi, dzr)
        dz = dzr * span  # 比值 → 当帧像素
        self.debug = f"psi{np.degrees(psi):+5.0f} oL{oL:+.2f} oR{oR:+.2f} dz{dz:+4.0f}"

        depth = BOX_DEPTH_RATIO * height
        axis = np.array([span_v[0], span_v[1], dz], np.float32)  # (屏幕x, 屏幕y, 朝观察者)
        length = float(np.linalg.norm(axis))
        axis_hat = axis / length
        # 截面正交基: ĉ₀ 朝观察者(+z), b̂₀ 屏幕"上"。
        # ĉ₀ 必须**恒定朝向观察者**——早先版本用 −axiŝ×(0,−1,0), 其 z 分量正比于
        # span_v.x, 于是双手 x 差过零时整组基翻号、盒子瞬间里外翻(可见面变成互补
        # 集)。而 ROLE_HYST_PX=25 的角色滞回恰好让 dx∈(−25,0) 成为可达状态, 不是
        # 理论边角。改为把 +ẑ 对长轴做 Gram-Schmidt 正交化: z 分量恒 ≥0, 无符号翻转。
        c0 = np.array([0.0, 0.0, 1.0], np.float32) - axis_hat * float(axis_hat[2])
        n0 = float(np.linalg.norm(c0))
        if n0 < 1e-3:  # 长轴几乎正对镜头, +ẑ 退化 → 换屏幕"上"当参考
            c0 = np.array([0.0, -1.0, 0.0], np.float32)
            c0 = c0 - axis_hat * float(axis_hat @ c0)
            n0 = max(float(np.linalg.norm(c0)), 1e-6)
        c0 = c0 / n0
        # b̂₀ = ±(ĉ₀ × âxis), 符号得挑一个。按"屏幕上"挑(b0[1]<0)会在 b0[1]=0 处
        # 180° 翻转 —— 解析上该条件等价于 span_v.x=0(长轴竖直), 而 ROLE_HYST_PX
        # 守的是**掌心** x 差, 与 lift 后的锚点 x 差实测相隔 559px(p95), 根本护不住:
        # 锚点 dx 过零实测单帧 608px 跳变、3.8% 画面像素变化。
        # 改为帧间传播(与上帧同向者胜): 长轴扫过竖直时连续穿过, 不存在翻转点。
        b0 = np.cross(c0, axis_hat)
        if self._b0_prev is not None:
            if float(b0 @ self._b0_prev) < 0.0:
                b0 = -b0
        elif b0[1] > 0:  # 冷启动才用"屏幕上"定初值
            b0 = -b0
        self._b0_prev = b0.copy()
        cos_p, sin_p = float(np.cos(psi)), float(np.sin(psi))
        b_hat = b0 * cos_p + c0 * sin_p
        c_hat = -b0 * sin_p + c0 * cos_p

        # f 至少 1.1×(高+深) → f−z 恒为正, 双手贴近时不会被透视除爆
        focal = BOX_FOCAL * max(length, height + depth)
        anchor = (gL + gR) * 0.5

        cam = np.empty((8, 3), np.float32)
        scr = np.empty((8, 2), np.float32)
        # w 的原点 = 锚点 = 绕长轴旋转的不动点。depth_bias=0.5 时体心落在手上,
        # 盒子跟着手整体转; =0 时前面压在手上, 翻转会变成"前棱当轴甩"。
        w_front = depth * self.depth_bias
        for xi, t in enumerate((-0.5, 0.5)):
            for ui, u in enumerate((-0.5 * height, 0.5 * height)):
                for wi, w in enumerate((w_front, w_front - depth)):
                    p3 = axis * t + b_hat * u + c_hat * w
                    k = xi * 4 + ui * 2 + wi
                    # 相机系右手基: x 右, y 上(屏幕 y 取负), z 朝观察者
                    cam[k] = (p3[0], -p3[1], p3[2])
                    scr[k] = anchor + p3[:2] * (focal / (focal - p3[2]))
        return scr, cam, focal, length
