**中文** · [English](TUTORIAL.en.md) · [日本語](TUTORIAL.ja.md)

# 使用教程

从零到把特效贴在自己手上，再到改参数、加风格。按顺序读，或者直接跳到你卡住的那一节。

- [0. 你需要什么](#0-你需要什么)
- [1. 安装与第一次运行](#1-安装与第一次运行)
- [2. 屏幕上那些东西是什么](#2-屏幕上那些东西是什么)
- [3. 五种风格怎么玩](#3-五种风格怎么玩)
- [4. 实时调参](#4-实时调参)
- [5. 录制](#5-录制)
- [6. 命令行完整参考](#6-命令行完整参考)
- [7. 改参数：去哪个文件、改哪一行](#7-改参数去哪个文件改哪一行)
- [8. 它是怎么工作的（架构导览）](#8-它是怎么工作的架构导览)
- [9. 开发：验证四层与 CI](#9-开发验证四层与-ci)
- [10. 加一种新风格](#10-加一种新风格)
- [11. 常见问题](#11-常见问题)

---

## 0. 你需要什么

| 项 | 要求 | 说明 |
|---|---|---|
| 电脑 | macOS（实测），Apple Silicon 最舒服 | M4 Pro 上 1080p：检测 6.9 ms/帧、绘制最重的风格 p95 6.5 ms，全程 30 fps 有余。Intel Mac 能跑但没量过。Linux / Windows **没验证过**，见 [11](#11-常见问题) 末尾 |
| 摄像头 | 内置的就行 | 1080p 最好；720p 也能用，特效尺寸按画面自适应 |
| Python | 3.12 – 3.14 | `mediapipe` 的 wheel 对新版本一向滞后，装不上就退一个版本（CI 钉 3.12，本机开发用 3.14） |
| 网络 | 首次运行需要 | 装依赖 + 下载 7.5 MB 的 MediaPipe 手部模型，之后离线可用 |
| ffmpeg | 可选 | 只有录制时的帧率修正和离线渲染的 h264 转封装用到；没有也能跑 |

依赖只有三个：`mediapipe`、`opencv-contrib-python`、`numpy`（版本范围见 `requirements.txt`，上界钉在下一个 major，原因写在文件里）。

---

## 1. 安装与第一次运行

### 1.1 一键（推荐）

```bash
git clone https://github.com/Eureka0w0v0/manual-tracking.git
cd manual-tracking
./run.sh
```

`run.sh` 第一次会：建 `.venv/` → 装 `requirements.txt` → 设 `PYTHONPATH=src` → 启动 `live --style cube`。
之后每次只做最后一步。

首次启动还会自动下载模型到 `models/hand_landmarker.task`（Google 官方 URL，带版本号，内容不可变；
下完校验 sha256，对不上就删掉报错——防代理劫持和 CDN 错误页）。

### 1.2 摄像头权限（macOS）

第一次会弹「"终端"想访问摄像头」，点**好**。没弹或点错了：
系统设置 → 隐私与安全性 → 摄像头 → 打开你用的终端（Terminal / iTerm / VS Code）。

也可以**双击 `start-live.command`**：它先用自动挑出的内置摄像头探一帧触发权限弹窗，再启动。

> 接了 iPhone 的「连续互通相机」时，OpenCV 的设备 0 可能是手机——被选中时手机会亮屏接管。
> 默认 `--camera -1` 会按 `system_profiler` 的顺序挑第一个不是 iPhone/iPad 的设备，通常就是内置的。

### 1.3 手动安装（想自己管环境）

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m manual_tracking live --style cube
```

**注意**：包故意不做 `pip install -e .`（见 `pyproject.toml` 首行），所以 `PYTHONPATH=src` 必须有。
另外 `src/manual_tracking/__main__.py` 用的是相对导入，只能走 `python -m manual_tracking`，
直接 `python __main__.py` 必报 `attempted relative import with no known parent package`。

### 1.4 IDE

- **VS Code**：仓库自带 `.vscode/`。`Cmd+Shift+B` = 一键跑默认风格；`F5` = 带调试启动（实时循环建议 `Ctrl+F5` 不带调试，debugpy 会拖慢帧率）；测试面板直接能跑 `tests/`。
- **其它 IDE**：打开仓库根的 `main.py` 点运行按钮。它只做一件事——把 `src/` 塞进 `sys.path` 再以包方式 import，替 IDE 绕开相对导入那条规则。参数照常传：`python main.py live --style mirror`。

### 1.5 退出

窗口聚焦时按 `Q` 或 `Esc`。终端里 `Ctrl+C` 也行——收尾顺序是 录像文件 → 摄像头 → 窗口 → 检测线程 → 模型，
任何一步失败都不拖累其余（所以录到一半强退也不会得到一个打不开的 mp4）。

---

## 2. 屏幕上那些东西是什么

### 2.1 开机横幅（终端）

```
========================================================
  MANUAL TRACKING LIVE — 折纸镜面 / 彩色玻璃盒 / 悬浮立方体 / TD横幅
  拇指+食指捏纸；翻转一只手拧麻花；捏死压成细线
  采集 1920x1080 (req 1920x1080)  帧率 30 (req 30)  推理边 全帧
  窗口 1920x1080 (可拖拽边角缩放)
  Q退出  S风格  D暗底  R录制  o p 底亮度
  screen 调参: [ ] 翻转曲线  ; ' 挂多高  , . 旋转轴  - = 锚点跟手  < > 收起距离  7 8 角速度上限
  screen/cube: 9 0 旋转跟手  g h 通透  { } 棱线
  cube  键位: F 重力开关  X 归位
  cube  手势: 捏在盒上拖=转 | 双手捏住=移动+缩放+拧 | 张开手=炸开
========================================================
```

- `采集 … (req …)`：摄像头**实际**给的分辨率/帧率 vs 你请求的。设备按能力和光线自己决定，低光下会自动降到 15 fps。
- 键位那几行不是手抄的，由 `live.py` 里的键位表生成——和运行时「按错风格会提示」用的是同一张表。

### 2.2 HUD（窗口顶部的黑条）

```
FPS  30.1  det   6.9ms  draw  4.2ms  hands:2  cube  idle  size 180 grip1 open0.62  顶+前+右
```

| 字段 | 含义 |
|---|---|
| `FPS` | 显示帧率（EMA，前 5 帧不计——冷启动的首帧慢得离谱，不该拿来播种） |
| `det` | 最近一次 MediaPipe 检测耗时（跑在 worker 线程，不阻塞画面） |
| `draw` | 渲染耗时（EMA） |
| `hands:N` | 这一帧用到的手数 |
| 风格名 | 当前风格 |
| `busy` / `idle` | 检测线程是否正在推理 |
| `ERRn` | 检测抛异常的累计次数。**出现就不正常**——第一条异常已经打在终端里了；没有这个计数的话，检测挂了只会表现成 `hands:0`，和「真没手」分不清 |
| `REC` + 右上红点 | 正在录制 |

后面跟着风格专属的一段：

- **cube**：`size` 边长 px、`grip` 抓住盒子的手数、`open` 未捏合手的张开度（炸开的驱动量，阈值 0.70→1.05）、然后是**当前朝向镜头的面**（如 `顶+前+右`）或 `explode`。
- **screen**：`expo lift bias resp cap` 五个旋钮的当前值、`psi` 盒子绕长轴的角、`oL oR` 双手掌面朝向（驱动 roll 的原始信号，-1..1）、`dz` 长轴的深度分量、然后是可见面。
  调 roll 时靠这个区分「几何没转到」和「画了但读不出来」。

HUD 用的是 OpenCV 的 Hershey 字体，**只认 ASCII**，所以这里都是英文缩写。

### 2.3 画面

- 实拍底默认压到 55% 亮度（`D` 切黑底，`o`/`p` 调亮度），特效是主角。
- 骨架：金色是第一只手、橙色是第二只，指尖的点比别的关节大一圈。`wire` 风格的骨架换成霓虹。
- 画面默认**镜像**（像照镜子，左右手方向和直觉一致）。`--no-mirror` 关掉。

---

## 3. 五种风格怎么玩

`S` 键按 `mirror → screen → cube → banner → wire` 循环。`--style` 直接指定。

先说通用的手势质量：

- 手占画面高度 **1/4 到 1/2** 最稳。太远 landmark 抖，太近手指出画。
- 正对镜头、光从前面来。逆光和顶光下 MediaPipe 容易丢手。
- 背景越乱检测越慢；`det` 超过 15 ms 就该换个地方坐。
- 两只手别叠在一起太久：互相遮挡时 MediaPipe 会偶尔只认出一只，程序有 8 帧的宽限（双手模式）和 250 ms 的轨迹保留，但超过就得重新建立。

### 3.1 cube — 悬浮立方体（默认）

它和别的风格根本不同：**立方体自己有位置、朝向、大小**，手只是去推它转它，松手后它留在原地慢慢自转（≈10°/秒），像一个真的悬在你面前的东西。

| 想做什么 | 怎么做 | 背后的判据 |
|---|---|---|
| 抓住它 | 拇指尖和食指尖**捏在一起**，捏点落在立方体上 | 捏合 = 两指尖距离 / 掌宽 < 0.42（松开要 > 0.62，中间是滞回区不抖）；捏点 = 两指尖中点，落在体心 ±0.75 边长内才算抓到——**捏在空气里什么都不控制**。抓住的瞬间捏点炸开一圈涟漪当回执 |
| 翻面 | 单手抓住后**拖** | 横拖绕竖轴、竖拖绕横轴，1 px ≈ 0.009 rad（拖 349 px 翻 180°）。用的是拖动增量而不是手掌朝向，所以往一个方向一直拖能无限翻，不会自己转回来 |
| 移动 / 缩放 / 拧 | **两只手都抓住**它 | 立方体跟两手中点走（1:1，不打折）；两手拉开多少倍它就大多少倍（边长限制在画面短边的 6%–60%）；两手像拧方向盘一样转 → 绕屏幕法线滚（这是拖动够不到的第三轴） |
| 炸开看六个面 | 松开捏合，**五指张开** | 张开度 = 食指尖↔小指尖距离 / 掌宽，0.70 以下收拢、1.05 以上全炸开；六个面沿各自法线飞离体心悬停。握拳或放下手就收回 |
| 扔它 | 拖着的时候松手 | 带走动量：转动继续滑（摩擦 0.85/帧，约 1 秒停）、平移继续漂、**碰到画面边缘反弹**。猛甩正好翻过一个面周期 |
| 重力 | 按 `F` | 松手走抛物线，落底弹跳，地面摩擦滚停。再按关掉 |
| 转乱了 / 推到边上 | 按 `X` | 位姿归位到画面中央 |

手感细节（都能在 `floatcube.py` 顶部改）：

- 抓住后**拖到哪都跟手**，哪怕捏点已经跑出盒子（双手缩放必然把手拉出盒外）。松开捏合才断。
- 检测掉帧 ~0.25 s 内不脱手；单帧的「松开」尖峰（翻腕时指尖被自己手背挡住，landmark 乱跳一下）不算松，要连续 3 帧才认。
- 双手模式掉了一只手，宽限 8 帧内**停住不动**——而不是掉回单手模式突然转起来。

### 3.2 screen — 彩色玻璃盒

**双手张开，像捧着一个看不见的长方体**。盒子由两只手的几个低噪参数「构造」出来：

- 两手的锚点（掌心↔指弧中点之间 85% 处）连线 = 长轴，长度 = 盒长；
- 指弧展开量（食指尖↔小指尖）× 1.14 = 盒高，进深 = 1.47 × 盒高；
- 掌面朝向（翻手）→ 盒子绕长轴 **roll**；
- 两手**掌宽比** → 长轴的深度分量：一只手往镜头前伸，盒子就指向镜头外，端面露出来。

手不钉顶点，所以盒子永远是刚体、透视永远正确（旧版八个角各自跟指尖，楔形无解——来龙去脉见 [`GLASS_BOX_GEOMETRY.md`](GLASS_BOX_GEOMETRY.md)）。

| 想做什么 | 怎么做 |
|---|---|
| 让盒子出现 | 双手张开、相距至少 40 px，指尖大致朝上 |
| 翻转看背面 | **翻手腕**（掌心↔掌背对镜头）。两手的朝向取平均，所以两只手一起翻最干脆 |
| 看端面 | 一只手往前伸、另一只往后收 |
| 收起 / 重现 | 双手靠拢到最近点 < 0.35 掌宽 → 盒子 0.17 s 压扁消失；拉开到 > 0.75 掌宽 → 长出重现（有滞回，不会在临界处闪） |

六个面六种像素处理，一眼可分：

| 面 | 处理 |
|---|---|
| 顶 | 蓝反相 gradient map + 横条 glitch |
| 前 | riso 套色版画（双色 + 通道错位 + 有序抖动） |
| 背 | 硬阈值双色（丝网印） |
| 底 | 点云 / 全息（局部对比度当伪深度） |
| 左端 | 四叉树自适应马赛克 |
| 右端 | 半调网点 |

有一条**已知边界**：掌面朝向是 cos 型信号，一整圈里走 `0→+1→0→−1`，所以一直往一个方向翻，盒子转到某个角度会自己往回转。这是信号本性，想无限翻用 `cube`。

### 3.3 mirror — 折纸镜面

**每只手拇指+食指捏着纸的一角**：左手食指尖=左上、左拇指尖=左下，右手对称。

- 平摊：一张反相镜面的纸（`clamp(283 − 0.56 × 背景)`，背景以负片鬼影透出）。
- **翻一只手**：那半张纸变暗、变冷灰、采样点外移——像纸折起来了。
- 两手错位到顶边和底边交叉：纸拧成麻花，两个三角翼，右翼压在前面。
- **两手都捏死**（捏距 < 16 px）：纸转到侧面，只剩两条白线夹一道缝。

### 3.4 banner — TouchDesigner 横幅

和 `mirror` 同一套四角（食指尖上边、拇指尖下边），但内容是四层条带：黄阈值头带 / 苍白 X-ray 中窗（左右内缩 7%，带黄侧线） / 白软阈值分隔线 / 悬出画外的红脚带。全部是摄像头画面的屏幕空间双色调变换，没有整板描边。

### 3.5 wire — 霓虹电流骨架

不需要手势，手放进画面就行：辉光 + 芯线 + 沿骨骼流动的光点。最便宜的风格，不碰盒子几何、不采样背景，卡的时候用它看检测本身稳不稳。

---

## 4. 实时调参

每个键改的是**内存里的字段**，退出后恢复模块顶部的常量默认值。想永久改去 [7](#7-改参数去哪个文件改哪一行)。

按了当前风格用不上的键，终端会说一句 `[ 只对 screen 有用 (当前 cube)`，而不是默默改一个不影响画面的字段。

### 所有风格

| 键 | 作用 |
|---|---|
| `S` | 循环切风格 |
| `D` | 实拍底 ↔ 黑底 |
| `o` `p` | 实拍底亮度（每步 0.05） |
| `R` | 录制开/关 |
| `Q` / `Esc` | 退出 |

### screen 与 cube 共用

| 键 | 参数 | 白话 |
|---|---|---|
| `g` `h` | `face_alpha` | 玻璃多透。正向面的不透明度 0.25–1.0，内壁按 0.42 倍跟着走 |
| `{` `}` | `box_edge_w` | 棱线粗细 px，0 = 无缝（默认）。原片是 4 px 白线，但六个面各有质感之后白线反而压过面 |
| `9` `0` | screen: `roll_resp` / cube: `orbit_gain` | 旋转跟手程度。screen 是加性步进 0.05；cube 跨一个数量级，乘性 ×1.15 |

### 只有 screen

| 键 | 参数 | 白话 |
|---|---|---|
| `[` `]` | `roll_expo` | 翻转曲线陡度：小=灵敏，大=中心钝但静止更稳。**不是**降噪旋钮（见下一条） |
| `7` `8` | `roll_max_rate` | 角速度上限 °/帧。这才是压「颜色频闪」的主控：挡掉掌面朝向在饱和区翻符号造成的瞬移。默认 10 = 300°/s，仍在真手速之上；8 就会削真动作 |
| `;` `'` | `anchor_lift` | 盒子挂多高：0 = 掌心（偏低）、1 = 指弧中点（原片位置）、默认 0.85 |
| `,` `.` | `depth_bias` | 绕长轴旋转的不动点：0 = 前面压在手上、0.5 = 体心（默认，盒子跟着手整体转）、1 = 后面 |
| `-` `=` | `anchor_resp` | 锚点跟手程度：大=跟手，小=稳但钝。手快速移动时会自动放开（速度自适应） |
| `<` `>` | `gap_shut` | 双手靠多近收起（按掌宽归一，与手离镜头远近无关）。出现阈值 = 它 + 0.40 |

### 只有 cube

| 键 | 作用 |
|---|---|
| `F` | 重力开关 |
| `X` | 位姿归位 |

---

## 5. 录制

- 按 `R` 开始，再按停止。文件在 `output/live_YYYYmmdd_HHMMSS.mp4`，**不含 HUD**。
- `--record path.mp4` 让第一次录制写到指定路径；之后再按 `R` 自动命名，不会覆盖刚录完的那条。
- 容器帧率取开录时的实测 FPS（钳在 10–60；启动头 15 帧还没热身时退回 30——否则首帧那个 ~6 fps 会被冻进容器，成片慢放 67%）。
- 停录时如果真实平均帧率和容器差 >5%（低光下摄像头自动降到 15 fps 很常见），且录了至少 1 秒，就用 ffmpeg **重封装**（`-c copy`，不重编码）到真实帧率。没装 ffmpeg 会提示已跳过，文件仍然能放，只是速度不对。

---

## 6. 命令行完整参考

三个子命令。空参 = `live`（风格由 `live.DEFAULT_STYLE` 决定，目前 `cube`）。

### `live` — 实时

```bash
./run.sh live [--style S] [--camera N] [--width W --height H] [--fps F]
              [--window-scale K] [--infer-size N] [--source-dim D]
              [--no-source] [--no-mirror] [--no-filter] [--model PATH] [--record PATH]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--style` | `cube` | `mirror` / `screen` / `cube` / `banner` / `wire`，旧名 `fabric`→mirror、`track`→screen、`outline`→wire 仍可用 |
| `--camera` | `-1` | 摄像头索引；-1 = 自动挑本机内置（绕开 iPhone 连续互通） |
| `--width` `--height` | 1920 × 1080 | 请求的采集分辨率。设备给了别的尺寸会被缩放到这个尺寸，特效坐标系保持一致 |
| `--fps` | 0（=30） | **请求**帧率。60 需要摄像头支持；检测中位 6.9 ms，60 Hz 喂得饱 |
| `--window-scale` | 1.0 | 窗口初始尺寸倍率。窗口本身可拖拽边角缩放，只影响显示 |
| `--infer-size` | 0 | MediaPipe 推理最长边；0 = 全帧。**Apple Silicon 上全帧最快且误差最小**，只在推理确实过慢的机器上设 640 之类 |
| `--source-dim` | 0.55 | 实拍底亮度 0–1 |
| `--no-source` | | 黑底 |
| `--no-mirror` | | 不做左右镜像 |
| `--no-filter` | | 关掉 One Euro 时域滤波，看裸 landmark（会抖，调试用；此时没有跨帧轨迹 id） |
| `--model` | 自动 | 指定 `.task` 模型；显式指定时**不会**自动下载，缺了直接报错 |
| `--record` | | 启动后第一次录制的输出路径 |

### `run` — 离线渲染视频

```bash
./run.sh run -i input.mp4 -o output.mp4 [--style S] [--source-dim 0.35] [--no-source] [--no-filter] [--max-frames N] [--model PATH]
```

逐帧跑检测 + 渲染，写 `mp4v`；装了 ffmpeg 会再转封装成 h264 + `faststart`（播放器兼容性）。
离线默认风格是 `mirror`、底亮度 0.35（成片里特效是主角，底图只当环境）。
`cube` 在离线视频里意义不大——没人捏它，只会自转。

### `dump` — 导出 21 点 JSON

```bash
./run.sh dump -i input.mp4 -o landmarks.json [--model PATH]
```

给 After Effects / TouchDesigner / 你自己的脚本用：

```json
{
  "width": 1920, "height": 1080, "fps": 30.0, "video": "input.mp4",
  "frame_count": 358,
  "frames": [
    {
      "frame": 0,
      "hands": [
        {"handedness": "Right", "score": 0.98,
         "points": [[x_px, y_px, z_norm], "... 共 21 个, 顺序同 MediaPipe（0 腕 … 4 拇指尖 … 8 食指尖 … 20 小指尖）"]}
      ]
    }
  ]
}
```

`x_px`/`y_px` 是**全帧像素坐标**（已经过 One Euro 滤波），`z_norm` 是 MediaPipe 的原始相对深度（只滤 xy，z 直通，免得跨参考系混合）。手按 `Right` 在前、`Left` 在后排序。

---

## 7. 改参数：去哪个文件、改哪一行

可调常数全部放在**各自负责的模块顶部**，每个都带一句「为什么是这个数」。

| 想改的东西 | 文件 | 例子 |
|---|---|---|
| 立方体手感 | `src/manual_tracking/floatcube.py` | 转太快 → `ORBIT_GAIN` 调小；松手滑太久 → `SPIN_DAMP` 调小；炸开太敏感 → `EXPLODE_LO/HI` 抬高；初始大小 → `CUBE_SIZE0` |
| 玻璃盒几何与滤波 | `src/manual_tracking/glassbox.py` | 盒子太扁 → `BOX_H_GAIN`；进深 → `BOX_DEPTH_RATIO`；颜色频闪 → `BOX_ROLL_MAX_RATE`（先看 `docs/GLASS_BOX_GEOMETRY.md` §3.7，别瞎调） |
| 六个面各自的像素处理 | `src/manual_tracking/effects.py` | 换颜色 → `RISO_DARK` / `DUOTONE_*` / `PC_TINT`…；换某个面用哪种处理 → `BOX_FACES` 那六行 |
| 玻璃通透度 / 棱线 / 光照 | `src/manual_tracking/renderer.py` | `BOX_FACE_ALPHA`、`BOX_BACK_ALPHA`、`BOX_EDGE_W`、`_LIGHT`、`SHADE_MIN` |
| 折纸镜面 / 横幅 / 霓虹 | `sheet.py` / `banner.py` / `neon.py` | 纸的亮度摆幅 `B_SWING`；横幅条带比例 `band(...)` 那几个数；辉光半径 `GLOW_BLUR` |
| 手部滤波 | `src/manual_tracking/tracker.py` | 嫌抖 → `OE_MIN_CUTOFF` 降到 0.6；嫌拖影 → 升到 1.5 |
| 检测延迟补偿 | `src/manual_tracking/detect_service.py` | `EXTRAP_CAP_MS` / `EXTRAP_DAMP` / `EXTRAP_MAX_PX` |
| 实时键位 | `src/manual_tracking/live.py` | `_KNOBS`（±步进的旋钮）和 `_ACTIONS`（其它键）两张表；撞键在启动时当场炸 |
| 底亮度默认值 | `live.SOURCE_DIM`（实时 0.55）/ `pipeline.SOURCE_DIM`（离线 0.35） | 两个场景两个值，各只定义一次 |
| 默认风格 | `live.DEFAULT_STYLE` / `pipeline.DEFAULT_STYLE` | 所有入口从这两处取，别在别处手抄 |

改完请跑一遍 [9](#9-开发验证四层与-ci) 的验证；改了 `screen` 的几何/映射/滤波要跑第四层。

---

## 8. 它是怎么工作的（架构导览）

### 8.1 数据流

```
摄像头 ──cap.read()──▶ 主线程                                      窗口
                        │ 镜像/缩放                                  ▲
                        ├─submit(frame)──▶ 检测线程 (detect_service)  │ imshow
                        │                    │ MediaPipe HandLandmarker
                        │                    │ tracker: 轨迹配对 + One Euro + handedness 锁存
                        │                    └─▶ 最近两次结果 (带时间戳)
                        ├─latest_pair() ◀────┘
                        ├─extrapolate(): 按当前时刻把 landmark 速度外推
                        ├─renderer.render(frame, hands)
                        │    ├─ cube:   floatcube.update → project → 六个面 fill
                        │    ├─ screen: glassbox.solve → 六个面 fill
                        │    ├─ mirror / banner / wire: sheet / banner / neon
                        │    └─ 骨架
                        ├─recorder.write(out)      ← 录制在画 HUD 之前
                        └─HUD → imshow → waitKey → 键位表分派
```

两条设计决定了「跟手」：

1. **检测不阻塞画面**。MediaPipe 跑在 worker 线程，主线程每帧零拷贝提交最新一帧（latest-wins 邮箱：worker 醒来只处理最新那帧，旧的直接丢）。检测率 = 1/max(检测耗时, 帧间隔)，不会被排队拖慢。
2. **按显示时刻外推**。检测结果总比画面晚 1–2 帧；主线程拿最近两次结果，按 track_id 配对算速度，把 landmark 推到「现在」（最多补 80 ms，阻尼 0.7，单点上限 40 px 防甩飞）。这消掉了检测率低于显示率时的阶跃顿挫。

### 8.2 追踪层（`tracker.py`）

- 每只手一条**持久轨迹**（`track_id`）：帧间按手腕距离做 2×2 最优指派，门限随掌宽自适应；短暂丢检测保留 250 ms（按墙钟，不按帧数——机器一卡帧数就不准）。
- **One Euro** 滤波替代固定 EMA：静止时重滤，快速运动时自动放开。只滤 xy，z 直通。
- **handedness 锁在轨迹上**：MediaPipe 每帧独立判左右手，翻腕时会单帧翻转；同一条轨迹就是同一只手，连续 5 帧改判才切。首帧标签不可信（实测刚认出的手最容易判错），起手就差一票。

### 8.3 渲染层的分工

```
paths ──▶ landmarks ──▶ tracker ──▶ handgeom ──▶ boxgeom ──▶ glassbox / floatcube
                                                     │                │
                                     effects ──▶ paint ──▶ sheet / banner / neon ──▶ renderer ──▶ detect_service ──▶ live ──▶ __main__
```

| 模块 | 只做一件事 |
|---|---|
| `landmarks.py` | MediaPipe 21 点的下标和骨骼连接表 |
| `handgeom.py` | 手 → 几个低噪标量/锚点（掌心、捏合、掌面朝向、指弧展开量、掌宽）。不碰画布，不知道有盒子 |
| `boxgeom.py` | 长方体顶点序 + 弱透视投影。`screen` 与 `cube` **共用的唯一一份** |
| `glassbox.py` | 双手参数 → 刚体长方体 8 顶点（含跨帧滤波状态）。不碰画布 |
| `floatcube.py` | 有位姿的立方体：抓取判定、拖动/双手/惯性/重力/炸开的状态机。不碰画布 |
| `effects.py` | 六种像素处理（`src → out` 的纯函数，参数在构造期烘进查表）。完全不知道「手」 |
| `paint.py` | 把多边形填成「被某种处理过的背景」（bbox 局部窗口 + 掩码 + alpha + 光照）。不知道手和盒子 |
| `sheet / banner / neon.py` | 三种不带盒子的风格 |
| `renderer.py` | 按 style 分派；`screen`/`cube` 的面排序（3D 外法线可见性 + 由远及近 + Lambert 光照）；底图缓存；左右手角色滞回 |
| `detect_service.py` | 检测线程 + 外推 |
| `live.py` | 摄像头、录制、HUD、键位、主循环 |
| `pipeline.py` | 离线视频 → 视频 / JSON |

依赖方向只能往右，`ruff` 的 isort 规则顺带把 import 顺序钉成这条线。

### 8.4 性能（1920×1080，真实 landmark，M4 Pro）

| 风格 | 绘制 mean | p95 |
|---|---|---|
| mirror | 0.66 ms | 0.88 |
| banner | 0.85 | 1.20 |
| wire | 2.40 | 2.77 |
| cube | 3.55 | 5.21 |
| screen | 4.05 | 6.49 |

检测 6.9 ms（中位）在另一个线程。33 ms 的帧预算用了不到 1/5，热路径已经没有值得动的地方——每个 effect 的 docstring 里写着它从多少 ms 优化到多少 ms 以及试过什么没用。

---

## 9. 开发：验证四层与 CI

```bash
.venv/bin/pip install -r requirements-dev.txt                # ruff + pytest
.venv/bin/ruff check main.py src tools tests                 # 1. 静态检查（含 isort）
.venv/bin/pytest                                             # 2. 契约：163 条，<1 秒，不碰摄像头/素材/模型
PYTHONPATH=src .venv/bin/python tools/cube_check.py          # 3. cube 手感底线：27 条断言，合成手
PYTHONPATH=src .venv/bin/python tools/e2e_check.py           # 4. screen 回归：需要样片 + 模型，约 1 分钟
```

| 层 | 挡什么 |
|---|---|
| 契约（`tests/`） | 改错了会崩：面拓扑与顶点序对齐、effect 的尺寸/不可变契约、盒子刚性、别名表、handedness 滞回、外推与录制帧率、键位分流、入口默认值、模型下载校验、收尾释放 |
| 手感（`cube_check`） | 拖不动、转回头、松手乱飘、撑出画面 |
| 回归（`e2e_check`） | 改差了不好用：**可见面每秒切换 5 次这类颜色频闪**、背面读不出。`tests/` 全绿也照样看不见这个 |

前三层每次 push 在 GitHub Actions 上跑（`.github/workflows/ci.yml`，ubuntu + Python 3.12）。第四层留在本地。

**关于样片**：`assets/sample.mp4` 不在仓库里（它是原作者的抖音视频）。想跑第四层，放一段自己的 1080p/30fps
双手翻转录像到这个路径。文档里所有实测数字都是对那段原片测的，换素材后：

```bash
PYTHONPATH=src .venv/bin/python tools/e2e_check.py --sweep   # 扫 expo × cap 网格
```

看完网格再决定 `tools/e2e_check.py` 顶部三个阈值要不要动——它们卡的是「越过体感红线」，不是「跟上次一模一样」。

`tools/synth.py` 是 `tests/` 和 `cube_check` 共用的合成手：21 个 landmark 按真实手比例全铺满（少设一个，那个 `(0,0)` 会把掌心往画面左上角拽，测出来的「跟手程度」是假的，而且不报错）。

提交习惯：一批改动**按主题拆 commit**，message 用中文 conventional commits（`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `build:` / `ci:` / `chore:`），正文写**为什么**——原先什么坏、实测数字多少。翻 `git log` 就是这个项目的设计文档。

---

## 10. 加一种新风格

假设叫 `glow`：

1. `src/manual_tracking/glow.py`：写一个 `draw(canvas, frame_bgr, ...)`。只吃已经分好左右的手（角色滞回归 renderer 管），可调常数放模块顶部并注明来源。填充用 `paint.fill`，像素处理写成 `effects.FaceEffect`（`src → out` 纯函数）。
2. `renderer.py`：`STYLES` 加名字（顺序 = `S` 键循环顺序），`render()` 里加一个分支。旧名想保留就进 `STYLE_ALIASES`。
3. 需要专属键？`live.py` 的 `_KNOBS`（±步进）或 `_ACTIONS`（其它形状），带上 `styles=("glow",)`，横幅和「按错风格提示」自动跟上。撞键会在启动时炸。
4. 测试：`tests/test_registry.py` 会自动覆盖注册表；给 `tests/test_render_smoke.py` 加一条「合成手喂进去不崩、画面确实变了」。
5. README 的风格列表和键位表加一行；跑四层。

---

## 11. 常见问题

**打不开摄像头 / `打不开摄像头 #0`**
权限（[1.2](#12-摄像头权限macos)）；被别的程序占着（Zoom、OBS、浏览器标签）；`--camera 1` 试别的索引。
接了 iPhone 时索引会变，默认的自动挑选已经绕开它，手动指定时注意。

**`hands:0`，手明明在画面里**
光线（前光）、距离（手占画面高 1/4–1/2）、手别贴边。HUD 有 `ERRn` 就不是检测不到，是检测**挂了**——终端里有第一条异常，通常是模型文件坏了（删掉 `models/` 让它重下）。

**画面卡 / `FPS` 掉到 15**
先看 HUD：`det` 高 → 机器慢，试 `--infer-size 640` 或 `--width 1280 --height 720`；`draw` 高 → 几乎不可能（见 [8.4](#84-性能19201080真实-landmarkm4-pro)），检查是不是开了调试器；两个都不高但 FPS 低 → 摄像头在低光下自动降帧，开灯。

**`attempted relative import with no known parent package`**
你直接跑了 `src/manual_tracking/__main__.py`。走 `./run.sh`、`python main.py` 或 `python -m manual_tracking`。

**`ModuleNotFoundError: No module named 'manual_tracking'`**
少了 `PYTHONPATH=src`（`run.sh` / `main.py` / `.vscode` 都替你设了，手动跑要自己加）。

**`模型校验失败: sha256 … ≠ 预期 …`**
下载到的不是官方模型（公司代理 / 校园网劫持 / CDN 错误页），已自动删除。换个网络重跑，或手动下载注释里那个 URL 放到 `models/hand_landmarker.task`，或 `--model` 指定你自己的文件。

**`import cv2` 报错 / 行为诡异**
同一个环境里同时装了 `opencv-python` 和 `opencv-contrib-python` 会互相覆盖：`pip uninstall opencv-python`，只留 contrib（mediapipe 依赖的就是它，且 `live` 要它的 GUI）。Linux 上 `import cv2` 报 `libGL.so.1` 缺失：`apt install libgl1 libglib2.0-0`。

**录出来的视频快放 / 慢放**
装 ffmpeg（`brew install ffmpeg`），停录时会按真实帧率重封装。见 [5](#5-录制)。

**盒子（screen）颜色一直闪**
先按 `7` 把 `roll_max_rate` 降到 8 看是否消失；是的话你的手速/光线让掌面朝向信号在饱和区翻符号，`docs/GLASS_BOX_GEOMETRY.md` §3.7 有完整分析。不要去调 `roll_expo`，它不是降噪旋钮。

**Linux / Windows 能跑吗？**
没验证过。代码里 macOS 专属的两处都有退化分支：AVFoundation 后端用 `hasattr` 守卫，缺了走 `CAP_ANY`；自动挑摄像头依赖 `system_profiler`，缺了退回索引 0。其余是纯 OpenCV / MediaPipe。`start-live.command` 是 macOS 的双击入口，别的平台用 `run.sh` 或 `python main.py`。跑通了欢迎提 issue 说一声。
