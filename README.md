# manual-tracking

实时手部特效（Python）：

- **`fabric`**：贴手金橙骨架（双手之间不再绘制能量丝/薄膜）  
- **`track`**：骨架 + 双手指尖间少量半透明面  
- **`wire` / `outline`**：纯骨架调试

```bash
./run.sh live
# 或 start-live.command
```

| 键 | 作用 |
|---|---|
| `S` | fabric / track / wire |
| `D` | 实拍底 / 黑底 |
| `R` | 录制（按实测帧率、不含 HUD） |
| `Q` | 退出 |

## TouchDesigner

完整 TD 安装与 torinmb 插件说明见：

[`docs/TOUCHDESIGNER_SETUP.md`](docs/TOUCHDESIGNER_SETUP.md)

插件：https://github.com/torinmb/mediapipe-touchdesigner
