# manual-tracking

实时手部特效（Python）：

三种特效的颜色/几何参数全部来自对原视频的逐像素逆向测量（4 个分析代理的报告）：

- **`mirror`**：折纸镜面（v1 前半）——角钉在拇指尖+食指尖；平摊单面、翻手拧麻花 X、捏死压成双瓣细线；面=反相镜面 `clamp(283−0.56×背景)`，折起面变冷灰
- **`screen`**：彩色玻璃长方体（v1 后半）——**参数化刚体**：手不钉顶点，只给低噪参数（每手 掌心↔指弧中点插值 = 锚点 → 长轴与盒长；指弧展开量 → 盒高；掌宽比 → 长轴深度分量；掌面前缩 → 绕长轴 roll），长方体在 3D 里造好再弱透视投影，刚性和两点透视是构造出来的；面可见性用 3D 外法线；五指收拢塌成扁带、双手合拢变白色种子点。**六个面六种像素处理**：

  | 面 | 处理 | 来源 |
  |---|---|---|
  | 顶 | 蓝反相 gradient map + 横条 glitch | 原片逐像素实测 |
  | 前 | riso 套色版画（双色 + 通道错位边条 + 抖动） | douyin 屏录复刻 |
  | 背 | 硬阈值双色（丝网印/宝丽来） | douyin 屏录复刻 |
  | 底 | 点云 / 全息（单目近似，局部对比度当伪深度） | douyin 屏录复刻 |
  | 左端 | 四叉树自适应马赛克 | douyin 屏录复刻 |
  | 右端 | 半调网点（暖白纸+墨点） | 自由创作 |

  常数标定见 [`docs/GLASS_BOX_GEOMETRY.md`](docs/GLASS_BOX_GEOMETRY.md)
- **`banner`**：TouchDesigner 横幅（v2）——黄阈值头带 / X-ray 中窗 / 白分隔线 / 悬出红脚带，全部是摄像头画面的屏幕空间双色调
- **`wire`**：纯骨架调试（`fabric`/`track`/`outline` 为旧名别名）

可调参数集中在 `renderer.py` 顶部 tunables 区。

```bash
./run.sh live                              # 默认 1920x1080，自动挑本机内置摄像头
./run.sh live --style screen               # 直接进玻璃盒
./run.sh live --window-scale 1.4           # 窗口开大点（也可直接拖拽边角）
# 或双击 start-live.command
```

| 键 | 作用 |
|---|---|
| `S` | mirror / screen / banner / wire |
| `D` | 实拍底 / 黑底 |
| `+` `-` | 实拍底亮度 |
| `R` | 录制（按实测帧率、不含 HUD）→ `output/live_*.mp4` |
| `Q` | 退出 |

`screen` 的实时调参（HUD 上同步显示当前值）：

| 键 | 参数 | 作用 |
|---|---|---|
| `[` `]` | `roll_expo` | 翻转曲线陡度（小=灵敏，大=中心钝但静止更稳） |
| `;` `'` | `anchor_lift` | 盒子挂多高（0=掌心，1=指弧中点） |
| `,` `.` | `depth_bias` | 绕长轴旋转的不动点（0=前面，0.5=体心，1=后面） |
| `7` `8` | `roll_max_rate` | 角速度上限 °/帧（挡掉 `_orient` 饱和区翻符号造成的瞬移） |
| `9` `0` | `roll_resp` | 旋转跟手程度（大=跟手，小=顺滑） |

HUD 末尾显示当前朝向镜头的面（如 `顶+前`），调 roll 时用来区分「几何没转到」和「画了但读不出来」。

## TouchDesigner

完整 TD 安装与 torinmb 插件说明见：

[`docs/TOUCHDESIGNER_SETUP.md`](docs/TOUCHDESIGNER_SETUP.md)

插件：https://github.com/torinmb/mediapipe-touchdesigner
