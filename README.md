# manual-tracking

实时手部特效（Python）：

- **`mirror`**：双手拇指+食指之间捏出一块泛白镜面板（抖音 manualtracking / AM 风格）  
- **`screen`**：同一块板换成黄红横幅夹负片实时画面（TouchDesigner 风格）  
- **`wire`**：纯骨架调试（`fabric`/`track`/`outline` 为旧名别名）

手指捏合板子会压扁消失，张开自然展开。

```bash
./run.sh live
# 或 start-live.command
```

| 键 | 作用 |
|---|---|
| `S` | mirror / screen / wire |
| `E` | 板面亮度 energy / calm / hot |
| `D` | 实拍底 / 黑底 |
| `R` | 录制（按实测帧率、不含 HUD） |
| `Q` | 退出 |

## TouchDesigner

完整 TD 安装与 torinmb 插件说明见：

[`docs/TOUCHDESIGNER_SETUP.md`](docs/TOUCHDESIGNER_SETUP.md)

插件：https://github.com/torinmb/mediapipe-touchdesigner
