"""`scripts/relay_l2l_net_probe.py` 的轻量判据测试（不需要 GPU / 模型）。

只覆盖不依赖推理的部分：endpoint 解析与 prompt token 构造；真正的数值判据由探针在实际设备上
运行时给出（记录见 `build/relay-records/l2l-net-*.json`）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relay_l2l_net_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("l2l_probe_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_endpoint_accepts_host_port() -> None:
    module = _load()
    assert module._split("127.0.0.1:50184") == ("127.0.0.1", 50184)


def test_split_endpoint_rejects_bad_shape() -> None:
    module = _load()
    for bad in ("127.0.0.1", "127.0.0.1:abc", ":50184"):
        with pytest.raises(SystemExit):
            module._split(bad)


class _FakeTokenizer:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, _text: str) -> list[int]:
        return list(self._ids)


def test_build_prompt_tokens_pads_to_prefill(tmp_path: Path) -> None:
    """短文本要重复到 prefill 长度（否则不同切点的输入长度不一致，比较就无效）。"""
    module = _load()
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello world", encoding="utf-8")
    tokens = module._build_prompt_tokens(_FakeTokenizer([1, 2, 3]), prompt, 8)
    assert len(tokens) == 8
    assert tokens[:3] == [1, 2, 3]


def test_build_prompt_tokens_truncates_long_text(tmp_path: Path) -> None:
    module = _load()
    prompt = tmp_path / "p.txt"
    prompt.write_text("x", encoding="utf-8")
    tokens = module._build_prompt_tokens(_FakeTokenizer(list(range(50))), prompt, 8)
    assert tokens == list(range(8))
