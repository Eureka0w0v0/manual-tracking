"""仓库布局相关的路径. 零依赖叶子模块.

单独拆出来是因为 live/pipeline/__main__ 都要 default_model_path(), 而它跟
"视频→视频管线"没有任何关系——放在 pipeline 里会让 live 为了一个路径函数
把整个 pipeline(连同 process_video/export_landmarks_json)拉进依赖。
"""

from __future__ import annotations

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
