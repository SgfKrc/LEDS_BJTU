from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_multimodal_contract import build_mm1_model_manifest, build_mm1_model_profile  # noqa: E402
from qwen3_multimodal_preflight import (  # noqa: E402
    Qwen3MultimodalPreflightError,
    build_mm1_visual_worker_request,
    inspect_mm1_processor_assets,
    validate_mm1_processor_smoke_response,
)
from scripts.model_tools.qwen3_multimodal_processor_probe import (  # noqa: E402
    run_qwen3_multimodal_processor_probe,
)
from scripts.model_tools.qwen3_multimodal_processor_probe_worker import (  # noqa: E402
    execute_request,
)


def _component(component_id: str, kind: str, digest: str) -> dict:
    return {
        "component_id": component_id,
        "artifact_id": f"{component_id}-artifact",
        "component_kind": kind,
        "format": "tokenizer" if kind == "processor" else "safetensors",
        "revision": "fixture-revision",
        "size_bytes": 128 if kind == "processor" else 1024,
        "sha256": digest * 64,
    }


def _manifest(model_name: str = "qwen3-vl-4b-instruct") -> dict:
    config = json.loads(
        (ROOT / "models" / model_name / "config.json").read_text(encoding="utf-8"),
    )
    profile = build_mm1_model_profile(config)
    return build_mm1_model_manifest(
        model_id=f"fixture-{profile['model_family']}-processor-smoke",
        model_family=profile["model_family"],
        runtime="transformers_sidecar",
        revision="fixture-revision",
        components=[
            _component("processor", "processor", "a"),
            _component("text", "text_weights", "b"),
            _component("vision", "vision_weights", "c"),
        ],
        text=profile["text"],
        vision=profile["vision"],
        processor=profile["processor"],
    )


def _prepared(model_name: str = "qwen3-vl-4b-instruct") -> tuple[dict, dict, dict]:
    manifest = _manifest(model_name)
    inspection = inspect_mm1_processor_assets(ROOT / "models" / model_name, manifest)
    request = build_mm1_visual_worker_request(
        request_id=f"processor-smoke-{manifest['model_family']}",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["processor", "vision"],
        modality="image",
        item_count=1,
        frame_count=0,
        width=256,
        height=256,
    )
    return manifest, inspection, request


def _fake_transformers(*, version: str = "4.57.6", kwargs_sink: list[dict] | None = None):
    class Qwen2VLImageProcessorFast:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

    class Qwen3VLVideoProcessor:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

    class Qwen2TokenizerFast:
        image_token_id = 151655
        video_token_id = 151656

    class Qwen3VLProcessor:
        image_token_id = 151655
        video_token_id = 151656

        def __init__(self):
            self.image_processor = Qwen2VLImageProcessorFast()
            self.video_processor = Qwen3VLVideoProcessor()
            self.tokenizer = Qwen2TokenizerFast()

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            if kwargs_sink is not None:
                kwargs_sink.append(kwargs)
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return Qwen3VLProcessor()

    return SimpleNamespace(__version__=version, AutoProcessor=AutoProcessor)


def _worker_request(model_name: str = "qwen3-vl-4b-instruct") -> tuple[dict, dict, dict]:
    manifest, inspection, visual_request = _prepared(model_name)
    return ({
        "schema_version": 1,
        "tool": "qwen3_multimodal_processor_probe",
        "operation": "qwen3_visual_worker_processor_smoke",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(ROOT / "models" / model_name),
        "manifest": manifest,
        "visual_request": visual_request,
        "controller_python": str(Path(sys.executable).with_name("controller.exe")),
    }, manifest, visual_request)


def test_processor_worker_constructs_real_contract_with_fake_isolated_transformers():
    request, _manifest_value, visual_request = _worker_request()
    kwargs_seen: list[dict] = []
    result = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(kwargs_sink=kwargs_seen),
    )
    assert result["status"] == "ready_for_offline_start"
    assert result["gate_passed"] is True
    assert result["response"]["runtime"]["processor_class"] == "Qwen3VLProcessor"
    assert result["response"]["runtime"]["tokenizer_class"] == "Qwen2TokenizerFast"
    assert result["response"]["runtime"]["image_token_id"] == visual_request["processor"]["image_token_id"]
    assert result["response"]["cleanup"]["completed"] is True
    assert result["response"]["cleanup"]["weight_materialized"] is False
    assert kwargs_seen == [{"local_files_only": True, "trust_remote_code": False}]
    assert str(ROOT) not in json.dumps(result)
    assert "model_path" not in json.dumps(result).lower()


def test_processor_worker_rejects_old_runtime_and_contract_drift():
    request, manifest, visual_request = _worker_request("qwen3-5-2b")
    old = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(version="4.47.1"),
    )
    assert old["status"] == "runtime_rejected"
    assert old["errors"][0]["code"] == "transformers_too_old"

    drifted = dict(visual_request)
    drifted["processor"] = dict(visual_request["processor"])
    drifted["processor"]["patch_size"] = 32
    request["visual_request"] = drifted
    rejected = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(),
    )
    assert rejected["status"] == "artifact_rejected"
    assert rejected["errors"][0]["code"] == "mm1_preflight_rejected"


def test_processor_worker_rejects_missing_processor_metadata(tmp_path: Path):
    source = ROOT / "models" / "qwen3-vl-4b-instruct"
    target = tmp_path / "qwen3-vl-metadata"
    target.mkdir()
    for name in (
        "config.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer_config.json",
    ):
        shutil.copy2(source / name, target / name)
    (target / "video_preprocessor_config.json").unlink()
    request, _manifest_value, _visual_request = _worker_request()
    request["model_path"] = str(target)
    result = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(),
    )
    assert result["status"] == "artifact_rejected"
    assert result["errors"][0]["code"] == "mm1_preflight_rejected"
    assert str(target) not in json.dumps(result)


def test_processor_worker_rejects_class_drift_and_network_protocol():
    request, _manifest_value, _visual_request = _worker_request()

    class WrongImageProcessor:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

    transformers = _fake_transformers()
    original = transformers.AutoProcessor.from_pretrained

    class DriftedAutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            processor = original(path, **kwargs)
            processor.image_processor = WrongImageProcessor()
            return processor

    transformers.AutoProcessor = DriftedAutoProcessor
    drifted = execute_request(request, module_loader=lambda name: transformers)
    assert drifted["status"] == "processor_contract_rejected"
    assert drifted["errors"][0]["code"] == "processor_contract_rejected"

    request["network_access"] = "enabled"
    unsafe = execute_request(request, module_loader=lambda name: _fake_transformers())
    assert unsafe["valid"] is False
    assert unsafe["status"] == "invalid_request"


def test_processor_smoke_response_contract_rejects_incomplete_cleanup():
    manifest, inspection, request = _prepared("qwen3-5-2b")
    runtime = {
        "transformers_version": "4.57.6",
        "isolated": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "processor_class": "Qwen3VLProcessor",
        "image_processor_class": "Qwen2VLImageProcessorFast",
        "video_processor_class": "Qwen3VLVideoProcessor",
        "tokenizer_class": "Qwen2TokenizerFast",
        "declared_tokenizer_class": "Qwen2Tokenizer",
        "image_token_id": manifest["processor"]["image_token_id"],
        "video_token_id": manifest["processor"]["video_token_id"],
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
    }
    from qwen3_multimodal_preflight import build_mm1_processor_smoke_response

    response = build_mm1_processor_smoke_response(
        request, manifest=manifest, inspection=inspection, runtime=runtime,
    )
    response["cleanup"]["completed"] = False
    with pytest.raises(Qwen3MultimodalPreflightError, match="cleanup is incomplete"):
        validate_mm1_processor_smoke_response(response, request=request)


def test_processor_controller_maps_worker_failure_and_repeat_cleanup():
    manifest, _inspection, request = _prepared()
    failed = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        worker_runner=lambda payload, timeout: (_ for _ in ()).throw(RuntimeError("worker exited")),
    )
    assert failed["status"] == "worker_failed"
    assert failed["errors"][0]["code"] == "worker_runner_failed"

    def runner(payload, timeout):
        isolated_payload = dict(payload)
        isolated_payload["controller_python"] = str(Path(sys.executable).with_name("controller.exe"))
        return execute_request(isolated_payload, module_loader=lambda name: _fake_transformers())

    first = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        worker_runner=runner,
    )
    second = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        worker_runner=runner,
    )
    assert first["response"] == second["response"]
    assert first["response"]["cleanup"]["completed"] is True


@pytest.mark.real_model
@pytest.mark.parametrize("model_name", ["qwen3-vl-4b-instruct", "qwen3-5-2b"])
def test_real_isolated_autoprocessor_smoke(model_name: str):
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.6 requires the isolated Qwen3 pipeline sidecar")
    manifest, _inspection, request = _prepared(model_name)
    report = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / model_name,
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
    )
    assert report["status"] == "ready_for_offline_start"
    assert report["gate_passed"] is True
    response = report["response"]
    assert response["processor_constructed"] is True
    assert response["runtime"]["transformers_version"] == "4.57.6"
    assert response["runtime"]["processor_class"] == "Qwen3VLProcessor"
    assert response["runtime"]["image_processor_class"] == "Qwen2VLImageProcessorFast"
    assert response["runtime"]["video_processor_class"] == "Qwen3VLVideoProcessor"
    assert response["runtime"]["tokenizer_class"] == "Qwen2TokenizerFast"
    assert response["cleanup"]["completed"] is True
    assert response["cleanup"]["weight_materialized"] is False
    encoded = json.dumps(report, ensure_ascii=True).lower()
    assert str(ROOT).lower() not in encoded
    assert "model_path" not in encoded
