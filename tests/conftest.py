"""tests 的共享装置.

这些测试**不碰摄像头、不跑 MediaPipe、不读素材**——只测纯逻辑契约, 全套跑完
不到一秒。需要真实信号驱动的手感回归在 `tools/` 里(cube_check / e2e_check),
两者互补: 这边挡"改错了会崩", 那边挡"改差了不好用"。

import 路径用**绝对路径**算, 不用 "src": pytest 可以从任意目录调起(IDE 常这么
干), 相对路径那一刻就静默失效, 然后报一个和路径八竿子打不着的 ImportError。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "src", ROOT / "tools"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from synth import hand  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """固定种子 —— 测试挂了必须能原样复现, 不能这次红下次绿."""
    return np.random.default_rng(20260726)


@pytest.fixture
def two_hands():
    """一对张开的手, 跨距 500px 足够 glassbox 解出盒子."""
    return hand(400, 400, pinch=False, tid=0), hand(900, 400, pinch=False, tid=1)
