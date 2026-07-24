# manual-tracking

实时手部特效（Python）：

三种特效的颜色/几何参数全部来自对原视频的逐像素逆向测量（4 个分析代理的报告）：

- **`mirror`**：折纸镜面（v1 前半）——角钉在拇指尖+食指尖；平摊单面、翻手拧麻花 X、捏死压成双瓣细线；面=反相镜面 `clamp(283−0.56×背景)`，折起面变冷灰
- **`screen`**：彩色玻璃长方体（v1 后半）——**参数化刚体**：手不钉顶点，只给 6 个低噪参数（每手 (食指尖+小指尖)/2 = 前面锚点 → 长轴与盒长；指弧展开量 → 盒高；掌面前缩 → 绕长轴 roll），长方体在 3D 里造好再弱透视投影，刚性和两点透视是构造出来的；面可见性用 3D 外法线；五指收拢塌成扁带、双手合拢变白色种子点；蓝顶面(反相 LUT+glitch)/绿前面/底面 X-ray/背面红。常数标定见 [`docs/GLASS_BOX_GEOMETRY.md`](docs/GLASS_BOX_GEOMETRY.md)
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
