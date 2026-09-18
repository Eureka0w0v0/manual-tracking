# TouchDesigner + MediaPipe 安装指南

> **English** — How to set up TouchDesigner with the torinmb/mediapipe-touchdesigner plugin (v0.5.2, TD 2023/2025, Mac + PC) to build the same kind of hand-driven effect natively in TD: download the release zip, enable External .tox, pick the webcam, turn on Hand tracking, read the 21 landmarks from the CHOP/DAT, and feed TD output back in via Spout (Windows) or Syphon → OBS virtual camera (macOS). Also notes on the Douyin clip this project imitates and its creator's public pages. Body in Chinese; the links and menu paths are the useful part.
>
> **日本語** — torinmb/mediapipe-touchdesigner プラグイン（v0.5.2、TD 2023/2025、Mac + PC）で、同じ種類の手のエフェクトを TouchDesigner 側で組むための導入手順です。release zip の取得、External .tox の有効化、Webcam の選択、Hand トラッキングをオンにして CHOP/DAT から 21 点を読む、TD の出力を Spout（Windows）や Syphon → OBS 仮想カメラ（macOS）で MediaPipe に戻す方法。このプロジェクトが真似た Douyin の動画と作者の公開ページについてのメモも。本文は中国語ですが、リンクとメニューのパスがそのまま使えます。

对应抖音「Github上的TouchDesigner项目」那类**实时跟手特效**的常用底座。

> 官方插件（标配）：**torinmb/mediapipe-touchdesigner**  
> https://github.com/torinmb/mediapipe-touchdesigner  
> 当前 Release：**v0.5.2**（支持 TD 2023.11880 / 2025.31500，Mac + PC）

---

## 1. 装 TouchDesigner

1. 打开 https://derivative.ca/download  
2. 下载 **TouchDesigner**（有非商用免费版）  
3. macOS：拖进「应用程序」  
4. 首次打开允许摄像头 / 网络权限（插件用内嵌浏览器）

检查版本：Help → About TouchDesigner。推荐 **2023.10k+** 或 **2025.x**。

---

## 2. 下 MediaPipe 插件

### 方式 A（推荐）：Release 包

```text
https://github.com/torinmb/mediapipe-touchdesigner/releases/latest/download/release.zip
```

或打开：

https://github.com/torinmb/mediapipe-touchdesigner/releases/tag/v0.5.2

下载 **`release.zip`**（约 180MB）→ 解压。

解压后大致有：

```text
release/
  MediaPipe TouchDesigner.toe   # 示例工程
  toxes/
    MediaPipe.tox               # 主组件
    HandTracking.tox            # 手部结果处理示例
    FaceTracking.tox
    PoseTracking.tox
    ...
```

### 方式 B：git clone 源码（开发用）

```bash
git clone https://github.com/torinmb/mediapipe-touchdesigner.git
cd mediapipe-touchdesigner
# 需要 node + yarn 才能从源码 build；日常使用请用 release.zip
```

---

## 3. 第一次跑通

1. 双击打开 **`MediaPipe TouchDesigner.toe`**  
2. 等 `MediaPipe` 组件加载完（内嵌 Chromium）  
3. 在 MediaPipe 参数里：
   - **Webcam** 下拉选你的摄像头  
   - 打开 **Hand**（手部）  
   - 需要的话再开 Face / Pose  
4. 应看到预览 TOP 上的手部骨架叠加  
5. 看输出 DAT/CHOP 是否在跳数据

### 重要：External .tox

把 `MediaPipe.tox` 拖进自己工程时，勾选：

> **Enable External .tox**

否则 `.toe` 会被撑得极大。

参考：https://derivative.ca/community-post/word-about-external-files-202310k-builds/68166

---

## 4. 分辨率

- 旧版常见上限 720p  
- **v0.5.2** 起可在 MediaPipe 组件里改更高分辨率（摄像头要支持）

---

## 5. 把 TD 画面送进 MediaPipe（可选）

### Windows：SpoutCam

1. 装 [SpoutCam](https://github.com/leadedge/SpoutCam/releases)  
2. TD 里 `Syphon Spout Out TOP`，Sender 名默认 `TDSyphonSpoutOut`  
3. MediaPipe 摄像头选 **SpoutCam**

### macOS：Syphon → OBS 虚拟摄像头

Mac 没有 SpoutCam 等价物，常用：

TD Syphon Out → OBS 接收 → OBS 虚拟摄像头 → MediaPipe 选该虚拟摄像头  

会有额外延迟。

---

## 6. 手部数据怎么用（做特效）

MediaPipe 手部任务输出 **每手 21 点**（与 Google MediaPipe Hands 一致）。

在 TD 里典型链路：

```text
Webcam
  → MediaPipe.tox
      → TOP: 预览/叠加
      → DAT/CHOP: 关键点坐标、handedness、gesture
  → 你的网络：
      SOP / GLSL TOP / Line MAT / Particle
      用指尖 CHOP 驱动曲线、布料、实例化
```

做「双手之间能量膜 / 橙金丝线」时：

1. 取左手/右手对应 `TIP` 点（拇指 4、食指 8、中指 12…）  
2. 用 **SOP 曲线** 或 **GLSL** 在两点间画 bezier  
3. 掌心中点做一张 **半透明 mesh**（膜）  
4. 颜色用橙金：`rgb(1.0, 0.55, 0.1)` 一类  

本仓库早期的 `fabric` 风格（橙金空间布）已退役，`--style fabric` 现在只是 `mirror`（折纸镜面）的别名。
不装 TD 想预览「双手之间撑一张面」这个思路，用 `./run.sh live --style mirror`；只要骨架和指尖用 `--style wire`。

---

## 7. 性能建议

- 关掉不用的模型（Face/Pose/Object）  
- 看 MediaPipe CHOP 里的 `detectTime` / `isRealTime`  
- Windows 笔记本可尝试在 BIOS 关超线程（文档称有时大幅提升）  
- 多 GPU 时注意 Spout 发送/接收在同一显卡管线

---

## 8. 相关链接

| 资源 | URL |
|---|---|
| 插件仓库 | https://github.com/torinmb/mediapipe-touchdesigner |
| 最新 Release | https://github.com/torinmb/mediapipe-touchdesigner/releases/latest |
| 介绍视频 | https://www.youtube.com/watch?v=Cx4Ellaj6kk |
| TouchDesigner 下载 | https://derivative.ca/download |
| MediaPipe Hands 文档 | https://developers.google.com/mediapipe/solutions/vision/hand_landmarker |
| 双手透视框例子 | https://github.com/AKCodez/xray-vision-touchdesigner |
| TD tox 合集 | https://github.com/DBraun/TouchDesigner_Shared |

---

## 9. 和抖音成片的关系

程序员阿飞那条「Github上的TouchDesigner项目」：

- 水印 **© 2026 Nunu Koe**
- 视频文案 **没有 GitHub 链接**
- 作者公开主页（**未发现公开同款 tox / 仓库**）：

| 平台 | URL |
|---|---|
| Instagram | https://www.instagram.com/nunu.koe/ （Miftahul Nur） |
| YouTube | https://www.youtube.com/c/nunukoe · https://www.youtube.com/@nunukoe |
| Threads | https://www.threads.net/@nunu.koe |

能确定的公开底座是 **torinmb/mediapipe-touchdesigner**；橙金空间布特效需在 TD 里用关键点自建，或用本仓库 `mirror` 风格（双手之间的折纸镜面）做实时近似——原先的 `fabric` 布料风格已退役。
