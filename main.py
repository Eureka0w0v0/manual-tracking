#!/usr/bin/env python3
"""IDE 友好的入口 —— 在 VS Code 里打开这个文件点绿色 ▶ 就能跑起来。

为什么要多这么一个文件:
  真正的入口是 `src/manual_tracking/__main__.py`, 但它用的是相对导入
  (`from .pipeline import ...`), 只能走 `python -m manual_tracking`。
  而 IDE 的运行按钮干的是"把当前文件当**脚本**执行" —— 那样 Python 不知道
  它属于哪个包, 必然报 `attempted relative import with no known parent
  package`。这里先把 `src/` 塞进 sys.path, 再以**包**的方式 import 进来,
  绕开那条规则。

  另外本项目故意不做 pip install(见 pyproject.toml 首行), 所以也不能指望
  解释器自己找得到 manual_tracking。

不带参数直接跑 = `live --style cube`, 但那个默认值**不在这里**定义 ——
它是 `__main__.DEFAULT_ARGV`, 全部一键入口(双击 / Cmd+Shift+B / F5 / ▶ /
裸 run.sh)共用同一份。这里只负责把空参原样交下去。

想换风格/分辨率, 照常传参:

    python main.py live --style mirror
    python main.py run -i in.mp4 -o out.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from manual_tracking.__main__ import main  # noqa: E402

if __name__ == "__main__":
    # 空参交给 __main__.DEFAULT_ARGV, 不在这里手抄一份默认风格
    sys.exit(main(sys.argv[1:]))
