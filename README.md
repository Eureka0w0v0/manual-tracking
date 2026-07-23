# manual-tracking

**你本人对着摄像头 → 手上实时出现和抖音 `#manualtracking` 同类的矢量特效。**

Google MediaPipe 跟手 + 矢量贴合层。不是拿别人成片离线滤镜，是 **你自己出镜**。

```
你的摄像头 → MediaPipe 21 点跟手 → 矢量 fluid/wire/outline → 实时窗口
```

## 一键实时（你要的这个）

**macOS 必须在有摄像头权限的终端里跑**（Cursor/后台代理通常没权限）。

### 方式 A：双击

双击项目里的：

`start-live.command`

第一次会弹「是否允许使用摄像头」→ 点 **好**。

### 方式 B：终端

```bash
cd ~/Claudecode/manual-tracking
./run.sh live
```

把手伸到镜头前，特效会贴在你手上。

| 键 | 作用 |
|---|---|
| `Q` / `ESC` | 退出 |
| `S` | 切换风格 fluid → wire → outline |
| `D` | 实拍底 / 纯黑底 |
| `R` | 开始/停止录制到 `output/live_*.mp4` |
| `+` / `-` | 调画面亮度 |

摄像头被拒时：

**系统设置 → 隐私与安全性 → 摄像头 → 打开「终端」（或 iTerm）**

## 离线处理视频（可选）

```bash
./run.sh run -i assets/sample.mp4 -o output/sample_fluid.mp4 --style fluid
./run.sh dump -i assets/sample.mp4 -o output/landmarks.json
```

## 目录

```
manual-tracking/
  start-live.command         # 双击启动实时
  run.sh                     # ./run.sh live
  src/manual_tracking/
    live.py                  # 实时摄像头主程序
    tracker.py               # MediaPipe
    renderer.py              # 矢量特效
  models/hand_landmarker.task
  output/
```

## 和原片的关系

| | 抖音原片 | 本项目 live |
|---|---|---|
| 主体 | 别人的手 | **你的手** |
| 跟手 | AM 手搓关键帧 | MediaPipe 实时 21 点 |
| 图形 | 手绘矢量 | 程序化矢量绑点 |

程序复刻的是「跟手矢量特效」体验；100% 手绘质感仍需回 AM 精修。

## 依赖

- Python 3.10+
- ffmpeg（可选，用于 H.264 重封装）
- 见 `requirements.txt`
