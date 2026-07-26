"""仓库布局相关的路径. 零依赖叶子模块.

单独拆出来是因为 live/pipeline/__main__ 都要 default_model_path(), 而它跟
"视频→视频管线"没有任何关系——放在 pipeline 里会让 live 为了一个路径函数
把整个 pipeline(连同 process_video/export_landmarks_json)拉进依赖。
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def repo_root() -> Path:
    # src/manual_tracking/paths.py -> repo root
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    return repo_root() / "models" / MODEL_NAME


def default_output_dir() -> Path:
    return repo_root() / "output"


def ensure_model(model_path: str | Path | None = None) -> Path:
    """返回就位可用的模型路径; 默认位置没有就从 MediaPipe 官方下载(~7.5MB).

    模型**不进 git**: 它是官方产物, 按 URL 随时能取回字节级一致的同一份, 而
    每换一个版本仓库就要永远背一份 7.5MB 的二进制(实测占 .git 的 78%)。

    显式传了 model_path 就**不下载** —— 那是用户指定的文件, 缺了该报错, 而不是
    偷偷换成官方版跑出用户没要的结果。

    先下到 .part 再原子改名: 下到一半被 Ctrl-C 打断时, 留下的是一个显然没写完
    的临时文件, 而不是一个大小不对却"存在"的 .task —— 后者会让下一次运行跳过
    下载, 然后在 MediaPipe 里炸出一个和网络毫无关系的报错。
    """
    if model_path is not None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"指定的模型不存在: {p}")
        return p

    p = default_model_path()
    if p.exists():
        return p

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    print(f"模型不在本地, 从 MediaPipe 官方下载 (~7.5MB)\n  {MODEL_URL}\n  → {p}")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"下载完成 ({p.stat().st_size / 1e6:.1f} MB)")
    return p
