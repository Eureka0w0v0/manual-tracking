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
echo "  Q 退出 | S 换风格 | D 暗底 | R 录制"
echo "=========================================="

# 触发一次摄像头权限请求（若尚未授权）
.venv/bin/python - <<'PY' || true
import cv2, time
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
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

exec .venv/bin/python -m manual_tracking live --camera 0 --style mirror
