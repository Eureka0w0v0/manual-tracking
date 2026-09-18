"""paths.ensure_model 的下载契约 —— 坏文件不可能顶着 .task 的名字活到下一次运行.

下到 .part → 校验 sha256 → 原子改名, 是模型加载链路上唯一会和网络打交道的
地方, 之前零测试。它坏掉的症状不在这里, 而是下一次启动时 MediaPipe 深处炸出
一个和网络毫无关系的报错。全部用假的 urlopen, 不联网。
"""

from __future__ import annotations

import hashlib
import io

import pytest

from manual_tracking import paths

PAYLOAD = b"not really a tflite model, but a few dozen bytes of one"


class _Broken(io.BytesIO):
    """读到一半断线的响应."""

    def read(self, size=-1):
        raise OSError("connection reset")


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """默认模型位置指到 tmp; urlopen 换成假的, 记下被调了几次."""
    target = tmp_path / "models" / "hand_landmarker.task"
    monkeypatch.setattr(paths, "default_model_path", lambda: target)
    calls: list[str] = []

    def serve(body):
        def fake_urlopen(url, timeout=0):
            calls.append(url)
            return body

        monkeypatch.setattr(paths.urllib.request, "urlopen", fake_urlopen)

    return target, calls, serve


def _part(target):
    return target.with_name(target.name + ".part")


def test_a_good_download_lands_atomically(sandbox, monkeypatch):
    target, calls, serve = sandbox
    monkeypatch.setattr(paths, "MODEL_SHA256", hashlib.sha256(PAYLOAD).hexdigest())
    serve(io.BytesIO(PAYLOAD))
    assert paths.ensure_model() == target
    assert target.read_bytes() == PAYLOAD
    assert not _part(target).exists()
    assert calls == [paths.MODEL_URL]


def test_a_wrong_digest_is_rejected_and_nothing_is_left_behind(sandbox):
    """代理劫持 / CDN 错误页会被当成"下载完成" —— 除非校验. 校验失败要删干净."""
    target, _, serve = sandbox
    serve(io.BytesIO(PAYLOAD))  # 真 MODEL_SHA256 是官方模型的指纹, 对不上
    with pytest.raises(RuntimeError, match="校验失败"):
        paths.ensure_model()
    assert not target.exists()
    assert not _part(target).exists()


def test_an_interrupted_download_leaves_no_partial_file(sandbox):
    """中断留下的 .part 不许活到下一次: 它显然没写完, 但名字一改就成了"模型"."""
    target, _, serve = sandbox
    serve(_Broken(PAYLOAD))
    with pytest.raises(OSError, match="connection reset"):
        paths.ensure_model()
    assert not target.exists()
    assert not _part(target).exists()


def test_an_existing_default_model_is_never_redownloaded(sandbox):
    target, calls, serve = sandbox
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    serve(io.BytesIO(b"should not be read"))
    assert paths.ensure_model() == target
    assert target.read_bytes() == PAYLOAD
    assert calls == []


def test_an_explicit_path_is_never_substituted(sandbox, tmp_path):
    """显式指定的模型缺了要报错, 不许偷偷换成官方版跑出用户没要的结果."""
    _, calls, serve = sandbox
    serve(io.BytesIO(PAYLOAD))
    mine = tmp_path / "mine.task"
    with pytest.raises(FileNotFoundError):
        paths.ensure_model(mine)
    mine.write_bytes(PAYLOAD)
    assert paths.ensure_model(mine) == mine
    assert calls == []
