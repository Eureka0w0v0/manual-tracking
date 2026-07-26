"""仓库布局相关的路径. 零依赖叶子模块.

单独拆出来是因为 live/pipeline/__main__ 都要 default_model_path(), 而它跟
"视频→视频管线"没有任何关系——放在 pipeline 里会让 live 为了一个路径函数
把整个 pipeline(连同 process_video/export_landmarks_json)拉进依赖。
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
# URL 带版本号(/1/), 内容不可变 —— pin 住指纹。没有它, 代理劫持/CDN 错误页
# 会被当成"下载完成", 然后下次运行在 MediaPipe 深处炸出和网络无关的报错。
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


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

    先下到 .part、校验 sha256、再原子改名: 中断留下的是显然没写完的临时文件;
    内容不对(劫持/错误页)当场报错删掉 —— 两种坏文件都不可能顶着 .task 的名字
    活到下一次运行, 否则 MediaPipe 会炸出一个和网络毫无关系的报错。
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
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise RuntimeError(
                f"模型校验失败: sha256 {digest[:12]}… ≠ 预期 {MODEL_SHA256[:12]}…\n"
                f"  下载的内容不是官方模型(代理劫持/错误页?), 已删除。\n"
                f"  手动下载: {MODEL_URL} → {p}"
            )
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"下载完成 ({p.stat().st_size / 1e6:.1f} MB)")
    return p
