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

# 空参不在这里补默认值 —— 交给 __main__.DEFAULT_ARGV(= ["live"]), 风格由
# live.DEFAULT_STYLE 统一决定, `./run.sh` 与 `./run.sh live` 进的是同一个。
# 历史: 这里曾写 `set -- live`, 而当时 argparse 里 live 的默认还是 mirror, 于是
# ./run.sh 进折纸镜面而双击入口进立方体, 同一个"一键"两种结果。
exec "$VENV/bin/python" -m manual_tracking "$@"
