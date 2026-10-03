"""PC 层段 worker 的能力声明（`Scheduler._task_worker_capabilities`）。

补 #33 的接线：载了层段工件就必须**同时**声明 `layer_forward` 与 `layer_ranges` ——
否则调度侧不会把层段 Stage 派给这台 PC worker（执行路径早就实现了，只是永远轮不到）。

测试只验证「声明 ⇔ 就绪」这一条契约，不加载任何真实模型。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scheduler import Scheduler  # noqa: E402
import scheduler_task_worker  # noqa: E402


class _FakeHost:
    """最小 host 打桩：capabilities 只读这几个属性。"""

    def __init__(self, *, layer_range=None, full_model_loaded=False):
        self.layer_range = layer_range
        self.model_loaded = full_model_loaded
        self.is_loaded = full_model_loaded
        self.active_model_id = "fake-model"

    def has_loaded_model(self):
        return self.model_loaded


def _capabilities(*, active_layer_config=None, layer_range=None, full_model_loaded=False):
    """绕过 Scheduler.__init__ 直接调能力快照。"""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._host = _FakeHost(
        layer_range=layer_range, full_model_loaded=full_model_loaded,
    )
    scheduler._active_layer_config = active_layer_config
    return scheduler._task_worker_capabilities()


class TestLayerCapabilities:
    def test_full_worker_keeps_legacy_stage_types(self):
        caps = _capabilities()
        assert caps["stage_types"] == ["full_inference", "aggregate"]
        assert caps["layer_worker"] is False
        assert caps["layer_ranges"] == []

    def test_layer_config_declares_layer_forward_and_range(self):
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
        )
        assert "layer_forward" in caps["stage_types"]
        assert caps["layer_worker"] is True
        assert caps["layer_ranges"] == [[16, 20]]

    def test_malformed_layer_range_does_not_break_hello(self):
        # layer_ranges 必须是 [start, end) 整数对；畸形值宁可不上报，也不能让 hello 挂掉。
        caps = _capabilities(
            active_layer_config={"layer_range": "16-20", "engine": "llama_cpp"},
        )
        assert "layer_forward" in caps["stage_types"]
        assert caps["layer_ranges"] == []

    def test_artifact_derived_model_id_is_protocol_safe(self, tmp_path, monkeypatch):
        """从工件 manifest 推导的 `model_id` 必须能过协议正则。

        manifest 的 `artifact` 字段是**相对路径**（如 `models\\x.gguf`），而
        `task_worker_protocol._SAFE_ID = ^[A-Za-z0-9_.:-]{1,128}$` 不允许反斜杠 ⇒
        直接拿它当 `model_id` 会让整个 hello 被判
        `payload.capabilities.models[0].model_id is invalid`，worker 永远进不了
        `admitted`（实测踩到，且被 legacy 通道的噪声盖了很久）。
        """
        import json

        from task_worker_protocol import _SAFE_ID, _validate_capabilities

        gguf = tmp_path / "qwen25-05b-f16-cut-16-20.gguf"
        gguf.write_bytes(b"")
        (tmp_path / "qwen25-05b-f16-cut-16-20.manifest.json").write_text(
            json.dumps({
                "source_layer_range": [16, 20],
                "artifact": "models\\qwen25-05b-f16-cut-16-20.gguf",
                "artifact_sha256": "a" * 64,
                "generator_version": 2,
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("QLH_LAYER_GGUF", str(gguf))

        caps = _capabilities()
        model_id = caps["models"][0]["model_id"]
        assert _SAFE_ID.match(model_id), f"model_id 不合法: {model_id!r}"
        assert "\\" not in model_id
        assert caps["layer_ranges"] == [[16, 20]]
        _validate_capabilities(caps, version=3)


class TestLayerCapabilitiesSurviveProtocolValidation:
    """声明的形状必须能被协议侧接受（否则 hello 直接被拒）。"""

    def test_declared_capabilities_pass_validation(self):
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
        )
        from task_worker_protocol import PROTOCOL_VERSION, _validate_capabilities

        _validate_capabilities(caps, version=PROTOCOL_VERSION)

    def test_full_worker_capabilities_pass_validation(self):
        caps = _capabilities()
        from task_worker_protocol import PROTOCOL_VERSION, _validate_capabilities

        _validate_capabilities(caps, version=PROTOCOL_VERSION)
