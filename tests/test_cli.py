"""__main__ 的参数默认值 —— "默认风格"只许有一个事实源.

`./run.sh`(空参 → DEFAULT_ARGV)和 `./run.sh live`(argparse 里 live 的默认)原先
各有一份: 前者 cube, 后者 mirror, 差一个词两种结果, 而 README 第一条示例就是
后者。现在 live 的默认只在 live.DEFAULT_STYLE 一处, 这里钉住三条路都从它取。
"""

from __future__ import annotations

import inspect

from manual_tracking import live, pipeline
from manual_tracking.__main__ import DEFAULT_ARGV, build_parser
from manual_tracking.renderer import STYLES


def test_bare_launch_and_explicit_live_agree_on_style():
    """`./run.sh` 与 `./run.sh live` 必须进同一个风格."""
    p = build_parser()
    assert p.parse_args(DEFAULT_ARGV).style == p.parse_args(["live"]).style == live.DEFAULT_STYLE


def test_cli_defaults_match_the_function_signatures():
    """argparse 的默认和 run_live()/process_video() 的签名默认必须是同一个值 ——
    命令行和直接 import 调用不该跑出两种画面(source_dim 早先就分岔过: 0.55 vs 0.65)。"""
    live_args = build_parser().parse_args(["live"])
    sig = inspect.signature(live.run_live).parameters
    assert live_args.style == sig["style"].default
    assert live_args.source_dim == sig["source_dim"].default

    run_args = build_parser().parse_args(["run", "-i", "in.mp4", "-o", "out.mp4"])
    sig = inspect.signature(pipeline.process_video).parameters
    assert run_args.style == sig["style"].default
    assert run_args.source_dim == sig["source_dim"].default


def test_default_styles_are_real_registry_names():
    """canon_style 对认不出的名字**静默**退回 mirror —— 默认值打错字不会报错,
    只会让每个入口都悄悄进折纸镜面。所以默认值必须是注册表里的正名。"""
    assert live.DEFAULT_STYLE in STYLES
    assert pipeline.DEFAULT_STYLE in STYLES
