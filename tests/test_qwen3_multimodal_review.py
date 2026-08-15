"""Regression tests for the MM1 review fixes."""

from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_multimodal_preflight import (
    Qwen3MultimodalPreflightError,
    build_mm1_media_tensor_reference,
    build_mm1_visual_input_contract,
    mm1_ledger_commit,
    mm1_vision_tower_placement,
    validate_mm1_resource_ledger,
)


def _reference():
    return build_mm1_media_tensor_reference(
        {
            "image": {
                "pixel_values_shape": [1, 3, 32, 32],
                "dtype": "float16",
                "token_count_estimate": 4,
            },
            "video": {
                "pixel_values_shape": [],
                "dtype": "",
                "token_count_estimate": 0,
            },
            "output_bytes_estimate": 128,
        },
        model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower"],
    )


def test_visual_input_contract_malformed_reference_is_contract_error():
    with pytest.raises(Qwen3MultimodalPreflightError):
        build_mm1_visual_input_contract({}, node_capacity_bytes=1024)


def test_ledger_rejects_conflicting_duplicate_entry():
    with pytest.raises(Qwen3MultimodalPreflightError):
        mm1_ledger_commit(
            [
                {"entry_id": "vision", "kind": "vision_tower_weights", "bytes": 10},
                {"entry_id": "vision", "kind": "vision_tower_weights", "bytes": 11},
            ],
            ledger_id="node-a",
            node_capacity_bytes=1024,
        )


def test_ledger_validator_rejects_negative_entry_without_raw_exception():
    ledger = mm1_ledger_commit(
        [{"entry_id": "vision", "kind": "vision_tower_weights", "bytes": 10}],
        ledger_id="node-a",
        node_capacity_bytes=1024,
    )
    tampered = dict(ledger)
    tampered["entries"] = [{"entry_id": "vision", "kind": "vision_tower_weights", "bytes": -1}]
    with pytest.raises(Qwen3MultimodalPreflightError):
        validate_mm1_resource_ledger(tampered)


def test_placement_requires_boolean_media_flag_and_positive_media_bytes():
    ledger = mm1_ledger_commit([], ledger_id="node-a", node_capacity_bytes=1024)
    with pytest.raises(Qwen3MultimodalPreflightError):
        mm1_vision_tower_placement(ledger, request_has_media="false", vision_tower_bytes=1)
    with pytest.raises(Qwen3MultimodalPreflightError):
        mm1_vision_tower_placement(ledger, request_has_media=True, vision_tower_bytes=0)


def test_mm118_defaults_to_no_full_model_materialization():
    from scripts.model_tools.qwen3_multimodal_vision_text_smoke_worker import execute_request

    report = execute_request({
        "schema_version": 1,
        "tool": "qwen3_multimodal_vision_text_smoke",
        "operation": "qwen3_vision_text_real_semantics",
        "read_only": True,
        "network_access": "disabled",
        "model_path": "unused",
        "image_path": "unused",
        "text_chain_id": "a" * 64,
        "generation": 1,
        "allow_full_model_materialization": False,
    })
    assert report["status"] == "resource_rejected"
    assert report["errors"][0]["code"] == "full_model_materialization_disabled"


def test_mm119_worker_processes_multiple_images_with_one_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from PIL import Image
    import scripts.model_tools.qwen3_multimodal_vision_text_smoke_worker as worker

    model_path = tmp_path / "model"
    model_path.mkdir()
    image_paths = [tmp_path / "sd-001.png", tmp_path / "90s-style.png"]
    for index, path in enumerate(image_paths):
        Image.new("RGB", (8, 8), (200 - index * 20, 10, 10)).save(path)

    class FakeTensor:
        shape = (1, 4)

        def __getitem__(self, key):
            return self

    class FakeModel:
        load_count = 0

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.load_count += 1
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            return FakeTensor()

    class FakeProcessor:
        decode_count = 0

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def apply_chat_template(self, conversation, **kwargs):
            return "describe"

        def __call__(self, **kwargs):
            return {"input_ids": FakeTensor()}

        def batch_decode(self, generated_ids, **kwargs):
            self.decode_count += 1
            return ["red apple on wood" if self.decode_count == 1 else "a colorful 1990s scene"]

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return SimpleNamespace(model_type="qwen3_vl")

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=512 * 2**20)

    fake_psutil = ModuleType("psutil")
    fake_psutil.Process = FakeProcess
    fake_psutil.virtual_memory = lambda: SimpleNamespace(available=20 * 2**30)
    fake_torch = ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.no_grad = FakeNoGrad
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    fake_transformers.Qwen3VLForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    report = worker.execute_request({
        "schema_version": 1,
        "tool": "qwen3_multimodal_vision_text_smoke",
        "operation": "qwen3_vision_text_real_semantics",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model_path),
        "image_paths": [str(path) for path in image_paths],
        "expected_keywords": [["apple", "red", "wood"], ["1990s", "colorful"]],
        "text_chain_id": "a" * 64,
        "generation": 1,
        "allow_full_model_materialization": True,
    })

    assert report["status"] == "vision_semantics_loaded"
    assert report["response"]["image_count"] == 2
    assert len(report["response"]["images"]) == 2
    assert report["response"]["semantic_pass_count"] == 2
    assert report["response"]["images"][1]["keyword_hits"] == {"1990s": True, "colorful": True}
    assert FakeModel.load_count == 1
    assert report["response"]["resource_observation"]["rss_peak_bytes"] == 512 * 2**20
    assert report["response"]["resource_observation"]["model_load_latency_ms"] >= 0
    assert report["response"]["resource_observation"]["total_latency_ms"] >= 0
    assert report["response"]["resource_observation"]["cuda_available"] is False
    assert str(tmp_path) not in json.dumps(report)


def test_mm119_controller_passes_bounded_images_and_adds_route_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import scripts.model_tools.qwen3_multimodal_vision_text_smoke as controller

    fake_root = tmp_path / "controller-root"
    sidecar = fake_root / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    sidecar.parent.mkdir(parents=True)
    sidecar.touch()
    model_path = tmp_path / "model"
    model_path.mkdir()
    image_paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in image_paths:
        path.write_bytes(b"fixture")
    captured: dict = {}

    worker_report = {
        "schema_version": 1,
        "tool": "qwen3_multimodal_vision_text_smoke",
        "operation": "qwen3_vision_text_real_semantics",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": True,
        "status": "vision_semantics_loaded",
        "errors": [],
        "response": {
            "image_count": 2,
            "images": [
                {"image_index": 0, "semantic_gate_passed": True},
                {"image_index": 1, "semantic_gate_passed": True},
            ],
            "resource_observation": {
                "rss_peak_bytes": 10,
                "rss_peak_delta_bytes": 5,
                "available_ram_before_bytes": 100,
            },
            "full_model_materialized": True,
            "explicit_full_model_opt_in": True,
        },
    }

    def fake_run(command, **kwargs):
        captured.update(json.loads(kwargs["input"]))
        return SimpleNamespace(stdout=json.dumps(worker_report), returncode=0)

    monkeypatch.setattr(controller, "ROOT", fake_root)
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    report = controller.run_qwen3_multimodal_vision_text_smoke(
        model=model_path,
        images=image_paths,
        expected_keywords=[["apple", "red", "wood"], ["1990s", "colorful"]],
        allow_full_model_materialization=True,
    )

    assert captured["image_paths"] == [str(path.resolve()) for path in image_paths]
    assert captured["expected_keywords"][1] == ["1990s", "colorful"]
    assert report["production_route_evaluation"]["recommended_route"] == "external_api"
    assert report["production_route_evaluation"]["native_sidecar"]["admitted"] is False
    assert "full_model_materialization_true" in report["production_route_evaluation"]["native_sidecar"]["reasons"]


def test_mm119_controller_rejects_more_than_four_images_before_sidecar_start():
    from scripts.model_tools.qwen3_multimodal_vision_text_smoke import (
        run_qwen3_multimodal_vision_text_smoke,
    )

    report = run_qwen3_multimodal_vision_text_smoke(
        model=Path("model"),
        images=[Path(f"image-{index}.png") for index in range(5)],
    )
    assert report["status"] == "invalid_request"
    assert report["errors"][0]["code"] == "request_incomplete"
