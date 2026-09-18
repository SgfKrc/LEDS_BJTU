"""tests/test_layer_prefix_detection.py — A2：层前缀自动探测的测试

背景：主仓 `_load_qwen2_layer_range` 此前硬编码 `model.layers.` 等 key 前缀，
而不同 Qwen 系包装器的前缀不同（实测 Qwen3.5 为 `model.language_model.layers.`）。
本测试覆盖新增的 `_detect_qwen_root_prefix`：
  * 标准 Qwen2 前缀 → `model.`
  * Qwen3.5 风格（language_model + 视觉塔）→ `model.language_model.`
  * 无 index.json 的单文件 → 仍能探测
  * 无 `layers.` key / 路径不存在 → 回退 `model.`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import model_module  # noqa: E402


def _write_shard(path: Path, keys: list[str]) -> None:
    from safetensors.torch import save_file

    tensors = {key: torch.zeros(2, 2) for key in keys}
    save_file(tensors, str(path))


def _write_index(model_dir: Path, mapping: dict[str, str]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping}), encoding="utf-8"
    )


class TestDetectRootPrefix:
    def test_standard_qwen2(self, tmp_path: Path) -> None:
        keys = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(3)]
        keys += ["model.embed_tokens.weight", "model.norm.weight"]
        _write_index(tmp_path, {k: "model.safetensors" for k in keys})
        assert model_module._detect_qwen_root_prefix(str(tmp_path)) == "model."

    def test_qwen35_language_model(self, tmp_path: Path) -> None:
        keys = [f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in range(3)]
        keys += ["model.language_model.embed_tokens.weight",
                 "model.visual.merger.linear_fc1.weight"]
        _write_index(tmp_path, {k: "model.safetensors" for k in keys})
        assert model_module._detect_qwen_root_prefix(str(tmp_path)) == "model.language_model."

    def test_single_file_without_index(self, tmp_path: Path) -> None:
        keys = [f"model.language_model.layers.{i}.mlp.up_proj.weight" for i in range(2)]
        keys.append("model.language_model.norm.weight")
        _write_shard(tmp_path / "model.safetensors", keys)
        assert model_module._detect_qwen_root_prefix(str(tmp_path)) == "model.language_model."

    def test_no_layer_keys_falls_back(self, tmp_path: Path) -> None:
        _write_index(tmp_path, {"lm_head.weight": "model.safetensors"})
        assert model_module._detect_qwen_root_prefix(str(tmp_path)) == "model."

    def test_missing_path_falls_back(self, tmp_path: Path) -> None:
        assert model_module._detect_qwen_root_prefix(str(tmp_path / "nope")) == "model."

    def test_custom_fallback(self, tmp_path: Path) -> None:
        _write_index(tmp_path, {"lm_head.weight": "model.safetensors"})
        assert model_module._detect_qwen_root_prefix(
            str(tmp_path), fallback="foo.") == "foo."

    def test_prefers_most_common_root(self, tmp_path: Path) -> None:
        """出现次数最多的 root 胜出（视觉塔只有 1 个 key，不应被选中）。"""
        keys = [f"model.language_model.layers.{i}.x" for i in range(5)]
        keys.append("model.visual.layers.7.proj.weight")
        _write_index(tmp_path, {k: "model.safetensors" for k in keys})
        assert model_module._detect_qwen_root_prefix(str(tmp_path)) == "model.language_model."
