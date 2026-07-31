#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

VENV="${ROOT}/.venv"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install -r "${ROOT}/requirements.txt"
fi

# 空参不在这里补默认值 —— 交给 __main__.DEFAULT_ARGV, 那是全部一键入口的唯一
# 权威。这里原先写的是 `set -- live`, 漏了 `--style cube`, 于是 ./run.sh 进
# 折纸镜面而双击入口进立方体, 同一个"一键"两种结果。
exec "$VENV/bin/python" -m manual_tracking "$@"
