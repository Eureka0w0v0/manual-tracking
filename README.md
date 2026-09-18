# manual-tracking

实时手部特效（Python）：

三种特效的颜色/几何参数全部来自对原视频的逐像素逆向测量（4 个分析代理的报告）：

- **`mirror`**：折纸镜面（v1 前半；`fabric`/`frame`/`planes`/`fluid` 为旧名别名）——角钉在拇指尖+食指尖；平摊单面、翻手拧麻花 X、捏死压成双瓣细线；面=反相镜面 `clamp(283−0.56×背景)`，折起面变冷灰
- **`screen`**：彩色玻璃长方体（v1 后半；`track` 为旧名别名）——**参数化刚体**：手不钉顶点，只给低噪参数（每手 掌心↔指弧中点插值 = 锚点 → 长轴与盒长；指弧展开量 → 盒高；掌宽比 → 长轴深度分量；掌面前缩 → 绕长轴 roll），长方体在 3D 里造好再弱透视投影，刚性和两点透视是构造出来的；面可见性用 3D 外法线 + Lambert 光照（转动时明暗随朝向流动）；**双手靠拢 → 盒子压扁收起，拉开 → 长出重现**（跨距/掌宽比值判据 + 滞回 + 0.17s 过渡，与手离镜头远近无关）。**六个面六种像素处理**：

  | 面 | 处理 | 来源 |
  |---|---|---|
  | 顶 | 蓝反相 gradient map + 横条 glitch | 原片逐像素实测 |
  | 前 | riso 套色版画（双色 + 通道错位边条 + 抖动） | douyin 屏录复刻 |
  | 背 | 硬阈值双色（丝网印/宝丽来） | douyin 屏录复刻 |
  | 底 | 点云 / 全息（单目近似，局部对比度当伪深度） | douyin 屏录复刻 |
  | 左端 | 四叉树自适应马赛克 | douyin 屏录复刻 |
  | 右端 | 半调网点（暖白纸+墨点） | 自由创作 |

  常数标定见 [`docs/GLASS_BOX_GEOMETRY.md`](docs/GLASS_BOX_GEOMETRY.md)
- **`cube`**：**悬浮立方体**（v3）——和 `screen` 的根本区别是**它有自己的位姿**：立方体自己存着位置/朝向/大小，手只是控制器，松手后它留在原地继续慢慢自转（≈10°/秒）。六个面复用 `screen` 那六种像素处理。

  | 手势 | 作用 |
  |---|---|
  | 单手**捏在立方体上**拖动 | 转动：横拖绕竖轴、竖拖绕横轴 —— 用来翻面。抓住的瞬间捏点炸开一圈**涟漪**（回执）；捏在空气里无效——抓取 = 捏合 ∧ 捏点落在盒上（建立后拖到哪都跟手，检测掉帧 ~0.25s 内不脱手，单帧"松开"尖峰不算松，松开捏合才断） |
  | 双手**都捏在立方体上** | 平移（跟两手中点）+ 缩放（跟两手距离；缩放会把手拉出盒外，建立过就不脱手）+ **拧转**（两手像拧方向盘一样转 → 绕屏幕法线滚动，补上拖动够不到的第三轴） |
  | **五指张开** | **炸开视图**：六个面沿各自法线飞离体心悬停——唯一能同时看全六种像素处理的姿态；握拳/放下手收拢 |
  | 松开 | 带走动量：转动继续滑、平移继续漂、**碰到画面边缘会反弹**，摩擦渐停后回到悠闲自转。按 `F` 开重力后松手走**抛物线**，落地弹跳滚停 |

  转动驱动用的是**拖动增量**而不是手掌朝向，所以往一个方向一直拖能无限翻下去，不会像 `screen` 那样翻到某个面自己转回来（`screen` 受制于 `_orient` 的 cos 型饱和，见 `docs/GLASS_BOX_GEOMETRY.md` §6）。手感底线有自动断言：`python tools/cube_check.py`
- **`banner`**：TouchDesigner 横幅（v2）——黄阈值头带 / X-ray 中窗 / 白分隔线 / 悬出红脚带，全部是摄像头画面的屏幕空间双色调
- **`wire`**：霓虹电流骨架——辉光 + 芯线 + 沿骨骼流动的光点（`outline` 为旧名别名）

可调参数就放在各自负责的模块顶部，下面的实时键位改的是同名字段，退出后恢复常量默认值：

| 想调的东西 | 去哪个文件 |
|---|---|
| 每个面的像素处理（riso / 硬阈值 / 点云 / 四叉树 / 网点 / 蓝顶 glitch） | `effects.py` |
| `screen` 的盒子几何与滤波 | `glassbox.py` |
| `cube` 的手感（增益、惯性、重力、炸开、抓取判据） | `floatcube.py` |
| `mirror` / `banner` / `wire` 三种风格 | `sheet.py` / `banner.py` / `neon.py` |
| 长方体的顶点序与弱透视投影（`screen` 与 `cube` 共用的**唯一**一份） | `boxgeom.py` |
| 多边形填充、描边这些绘制原语 | `paint.py` |
| 检测线程与速度外推 | `detect_service.py` |

`renderer.py` 只剩「按 style 分派 + `screen`/`cube` 的绘制 + 底图缓存 + 左右手角色滞回」。

```bash
./run.sh live                              # 默认风格 cube、1920x1080，自动挑本机内置摄像头
./run.sh live --style screen               # 直接进玻璃盒
./run.sh live --style cube                 # 悬浮立方体（捏住拖=翻面，张开手=炸开）
./run.sh live --fps 60                     # 请求 60fps 采集（需摄像头支持，检测链路喂得饱）
./run.sh live --window-scale 1.4           # 窗口开大点（也可直接拖拽边角）
# 或双击 start-live.command
```

IDE 里想直接点运行按钮的话用仓库根的 `main.py`。不带参数 = `live`，默认风格 `cube`
由 `live.DEFAULT_STYLE` 统一决定，和 `./run.sh live`、`start-live.command` 一致（所有入口进的是同一个风格）：

```bash
python main.py                             # = ./run.sh live --style cube
python main.py live --style mirror         # 想要别的风格就照常传参
```

**不要**直接执行 `src/manual_tracking/__main__.py`——它用相对导入，只能走
`python -m manual_tracking`，当脚本跑必报 `attempted relative import with no
known parent package`。`main.py` 存在的唯一理由就是替 IDE 绕开这条规则。

首次运行会自动下载 MediaPipe 手部模型（~7.5MB）到 `models/`。它不进 git——官方
URL 随时能取回字节级一致的同一份，没必要让仓库永远背着它。想用自己的模型就
`--model /path/to/xxx.task`（显式指定时**不会**自动下载，缺了直接报错）。

| 键 | 作用 |
|---|---|
| `S` | mirror / screen / cube / banner / wire |
| `D` | 实拍底 / 黑底 |
| `o` `p` | 实拍底亮度（暗 / 亮） |
| `R` | 录制（按实测帧率、不含 HUD）→ `output/live_*.mp4` |
| `Q` / `Esc` | 退出 |

`screen` 的实时调参（HUD 上同步显示当前值）：

| 键 | 参数 | 作用 |
|---|---|---|
| `[` `]` | `roll_expo` | 翻转曲线陡度（小=灵敏，大=中心钝但静止更稳） |
| `;` `'` | `anchor_lift` | 盒子挂多高（0=掌心，1=指弧中点） |
| `,` `.` | `depth_bias` | 绕长轴旋转的不动点（0=前面，0.5=体心，1=后面） |
| `7` `8` | `roll_max_rate` | 角速度上限 °/帧（挡掉 `_orient` 饱和区翻符号造成的瞬移） |
| `9` `0` | `roll_resp` | 旋转跟手程度（大=跟手，小=顺滑） |
| `g` `h` | `face_alpha` | 玻璃通透度（内壁 alpha 按 0.42 倍跟着走） |
| `-` `=`（或 `_` `+`） | `anchor_resp` | 锚点跟手程度（大=跟手，小=稳但钝） |
| `<` `>` | `gap_shut` | 双手靠多近才收起盒子（按掌宽归一，与手离镜头远近无关） |
| `{` `}` | `box_edge_w` | 盒子棱线粗细 px（0=无缝，默认 0） |

`cube` 的实时调参：

| 键 | 参数 | 作用 |
|---|---|---|
| `9` `0` | `orbit_gain` | 拖 1px 转多少（大=一点点手就翻一圈） |
| `g` `h` | `face_alpha` | 玻璃通透度 |
| `F` | `gravity` | 重力开关（扔出去走抛物线、落底弹跳、地面摩擦滚停） |
| `X` | — | 位姿归位（转乱了/推到边上时按） |

HUD 末尾显示当前朝向镜头的面（如 `顶+前`），调 roll 时用来区分「几何没转到」和「画了但读不出来」。

上面三张表**按风格分流**：在用不上某个键的风格里按它，终端会给一句
`[ 只对 screen 有用 (当前 cube)`，而不是默默改一个不影响画面的字段。开机横幅
也按同样的分组打印，两边都由 `live._KNOBS` 生成，不手抄。

## 开发

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check main.py src tools tests                # 静态检查（配置在 pyproject.toml）
.venv/bin/pytest                                           # 纯逻辑契约 + 渲染冒烟，153 条，<1 秒
PYTHONPATH=src .venv/bin/python tools/cube_check.py        # cube 手感底线，27 条断言
PYTHONPATH=src .venv/bin/python tools/e2e_check.py         # screen 手感底线，3 条断言
PYTHONPATH=src .venv/bin/python tools/e2e_check.py --sweep # 扫 expo/cap 找参数
```

验证分三层，各挡各的：

| 层 | 位置 | 挡什么 | 依赖 |
|---|---|---|---|
| 契约 | `tests/` | 改错了会崩：面拓扑与顶点序对齐、effect 的尺寸/不可变契约、盒子刚性、别名表、handedness 滞回、外推与录制帧率、键位分流 | 无（合成手，不读素材） |
| 手感 | `tools/cube_check.py` | 拖不动、转回头、松手乱飘、撑出画面 | 无（合成手） |
| 回归 | `tools/e2e_check.py` | 改差了不好用：颜色频闪、背面读不出 | 样片 + 模型 |

前两层不碰摄像头、不读素材、不下模型，所以每次 push 都在 CI 上跑一遍
（`.github/workflows/ci.yml`）。**第三层留在本地** —— CI 绿了不代表没有频闪。

改完 `screen` 的几何/映射/滤波必须跑 `e2e_check`：它是唯一能发现「可见面每秒切换
5 次」这类体感灾难的手段——`tests/` 那 153 条全绿也照样看不见频闪。基线与结论见
[`docs/GLASS_BOX_GEOMETRY.md`](docs/GLASS_BOX_GEOMETRY.md) §3.6–3.8。

`tools/synth.py` 是 `tests/` 和 `cube_check` 共用的合成手（21 个 landmark 全铺满：
少设一个，那个 `(0,0)` 会把掌心往画面左上角拽，测出来的「跟手程度」是假的）。

## TouchDesigner

完整 TD 安装与 torinmb 插件说明见：

[`docs/TOUCHDESIGNER_SETUP.md`](docs/TOUCHDESIGNER_SETUP.md)

插件：https://github.com/torinmb/mediapipe-touchdesigner
