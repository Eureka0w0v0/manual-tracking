#!/bin/bash
# 双击这个文件，或在「终端」里运行，才能拿到 macOS 摄像头权限
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
fi

echo "=========================================="
echo "  把手伸到镜头前 — 特效会贴在你自己手上"
echo "  悬浮立方体: 捏在它上面拖=翻面 | 双手捏住它=移动+缩放 | 张开手=炸开 | X 归位"
echo "  Q 退出 | S 换风格 | D 暗底 | R 录制"
echo "=========================================="

# 触发一次摄像头权限请求（若尚未授权）。用自动挑出的内置摄像头索引——
# 写死 0 的话，接了 iPhone 连续互通时可能探到手机上去、把手机叫醒接管。
.venv/bin/python - <<'PY' || true
import sys, time
sys.path.insert(0, "src")
import cv2
from manual_tracking.live import _builtin_camera_index

cap = cv2.VideoCapture(_builtin_camera_index(), cv2.CAP_AVFOUNDATION)
for _ in range(30):
    if cap.isOpened():
        ok, f = cap.read()
        if ok:
            print("摄像头 OK")
            break
    time.sleep(0.1)
else:
    print("还没拿到摄像头权限：请在弹窗点「好」，或去 系统设置→隐私→摄像头 打开终端")
cap.release()
PY

exec .venv/bin/python -m manual_tracking live --style cube
