# manual-tracking

实时手部特效（Python）：

三种特效的颜色/几何参数全部来自对原视频的逐像素逆向测量（4 个分析代理的报告）：

- **`mirror`**：折纸镜面（v1 前半）——角钉在拇指尖+食指尖；平摊单面、翻手拧麻花 X、捏死压成双瓣细线；面=反相镜面 `clamp(283−0.56×背景)`，折起面变冷灰
- **`screen`**：彩色玻璃盒（v1 后半）——盒长=食指尖距、端棱=食指→小指；蓝顶面(反相 LUT)+绿前面(正相 LUT)+翻折露红背面；顶面折射采样+横条 glitch，全棱白描边
- **`banner`**：TouchDesigner 横幅（v2）——黄阈值头带 / X-ray 中窗 / 白分隔线 / 悬出红脚带，全部是摄像头画面的屏幕空间双色调
- **`wire`**：纯骨架调试（`fabric`/`track`/`outline` 为旧名别名）

可调参数集中在 `renderer.py` 顶部 tunables 区。

```bash
./run.sh live
# 或 start-live.command
```

| 键 | 作用 |
|---|---|
| `S` | mirror / screen / banner / wire |
| `D` | 实拍底 / 黑底 |
| `R` | 录制（按实测帧率、不含 HUD） |
| `Q` | 退出 |

## TouchDesigner

完整 TD 安装与 torinmb 插件说明见：

[`docs/TOUCHDESIGNER_SETUP.md`](docs/TOUCHDESIGNER_SETUP.md)

插件：https://github.com/torinmb/mediapipe-touchdesigner
