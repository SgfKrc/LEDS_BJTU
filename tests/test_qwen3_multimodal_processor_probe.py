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

        def __call__(self, images, **kwargs):
            # MM1.7：合成媒体预处理——返回模拟 pixel_values（不加载权重）
            import numpy as np
            arr = np.asarray(images)
            h, w = arr.shape[:2]
            return {"pixel_values": np.zeros((1, 3, h, w), dtype=np.float16)}

    class Qwen3VLVideoProcessor:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

        def __call__(self, frames, **kwargs):
            import numpy as np
            arr = np.asarray(frames)
            n, h, w = arr.shape[:3]
            return {"pixel_values": np.zeros((1, n, 3, h, w), dtype=np.float16)}

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
        media_smoke={"image_size": [32, 32], "video_size": [32, 32], "video_frames": 2},
    )
    assert report["status"] == "ready_for_offline_start"
    assert report["gate_passed"] is True
    response = report["response"]
    assert response["processor_constructed"] is True
    # MM1.7：真实 processor 媒体预处理摘要（shape/dtype/token 数，无权重）
    summary = response["media_summary"]
    assert summary is not None
    assert summary["weight_materialized"] is False
    # 真实 processor 会 resize/网格化（shape 维度数因实现而异）——
    # 契约只投影存在性、dtype 与正数 token 数
    assert len(summary["image"]["pixel_values_shape"]) >= 2
    assert "float" in summary["image"]["dtype"]
    assert summary["image"]["token_count_estimate"] and summary["image"]["token_count_estimate"] > 0
    if summary["video"]["pixel_values_shape"]:
        # 真实 video_processor 对合成帧可能不产张量——非空时校验
        assert len(summary["video"]["pixel_values_shape"]) >= 2
        assert "float" in summary["video"]["dtype"]
        assert summary["video"]["token_count_estimate"] and summary["video"]["token_count_estimate"] > 0
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


# ================================================================
# MM1.7：CPU 合成媒体预处理与张量摘要合同
# ================================================================

def test_media_preprocess_projects_shape_dtype_and_tokens():
    """合成图像/帧经 processor 预处理后投影摘要（shape/dtype/token 数）。"""
    request, _manifest, _visual = _worker_request()
    request["media_smoke"] = {"image_size": [64, 64], "video_size": [32, 32], "video_frames": 3}
    result = execute_request(
        request, module_loader=lambda name: _fake_transformers(),
    )
    assert result["gate_passed"] is True
    summary = result["response"]["media_summary"]
    assert summary is not None
    assert summary["weight_materialized"] is False
    assert summary["full_model_materialized"] is False
    # 图像：shape [1,3,H,W]、fp16、token 网格 = H*W
    assert summary["image"]["pixel_values_shape"][-2:] == [64, 64]
    assert summary["image"]["dtype"] == "float16"
    assert summary["image"]["token_count_estimate"] == (64 // 16) * (64 // 16)  # patch=16
    # 视频：shape [1,F,3,H,W]
    assert summary["video"]["pixel_values_shape"][1] == 3
    assert summary["video"]["pixel_values_shape"][-2:] == [32, 32]
    assert summary["video"]["token_count_estimate"] == (32 // 16) * (32 // 16)
    assert summary["output_bytes_estimate"] > 0


def test_media_preprocess_rejects_out_of_bounds_media():
    """超限尺寸/帧数 fail-closed（MM1.7 契约限制）。"""
    for bad_media in (
        {"image_size": [2048, 64]},            # 图像边长超 1024
        {"video_size": [64, 4096]},            # 视频边长超 1024
        {"video_frames": 64},                  # 帧数超 32
        {"video_frames": 0},                   # 空媒体
        {"image_size": [2, 2]},                # 过小
    ):
        request, _manifest, _visual = _worker_request()
        request["media_smoke"] = bad_media
        result = execute_request(
            request, module_loader=lambda name: _fake_transformers(),
        )
        assert result["gate_passed"] is False, bad_media
        assert result["errors"][0]["code"] in (
            "processor_contract_rejected", "processor_smoke_failed",
        )


def test_media_summary_contains_no_sensitive_data():
    """摘要不含像素值/原始媒体/路径（path-free 合同）。"""
    request, _manifest, _visual = _worker_request()
    request["media_smoke"] = {"image_size": [32, 32], "video_frames": 2}
    result = execute_request(
        request, module_loader=lambda name: _fake_transformers(),
    )
    payload = json.dumps(result, ensure_ascii=True)
    # 字段名 pixel_values_shape 允许；实际像素数组/原始媒体/路径禁止
    assert '"pixel_values":' not in payload
    assert "models/" not in payload
    assert "rng" not in payload


# ================================================================
# MM1.8：媒体张量跨边界投影与视觉组件占位
# ================================================================

def test_media_tensor_reference_projects_capacity_and_stays_path_free():
    """媒体摘要投影为 path-free 张量参考（shape/dtype/token/容量预算）。"""
    from qwen3_multimodal_preflight import (
        build_mm1_media_tensor_reference,
        validate_mm1_media_tensor_reference,
    )

    summary = {
        "image": {"pixel_values_shape": [1, 3, 64, 64], "dtype": "float16",
                  "token_count_estimate": 16},
        "video": {"pixel_values_shape": [1, 2, 3, 32, 32], "dtype": "float16",
                  "token_count_estimate": 8},
        "output_bytes_estimate": 4096,
        "weight_materialized": False,
        "full_model_materialized": False,
    }
    reference = build_mm1_media_tensor_reference(
        summary, model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )
    validated = validate_mm1_media_tensor_reference(
        reference, model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )
    assert validated["reference_kind"] == "qwen3_visual_media_tensor_placeholder"
    assert validated["media"]["image"]["token_count_estimate"] == 16
    assert validated["media"]["video"]["token_count_estimate"] == 8
    assert validated["capacity"]["total_media_tokens"] == 24
    assert validated["capacity"]["output_bytes_estimate"] == 4096
    assert validated["weight_materialized"] is False
    payload = json.dumps(validated, ensure_ascii=True)
    assert "pixel_values\"" not in payload
    assert "models/" not in payload


def test_media_tensor_reference_rejects_bad_input(monkeypatch):
    """空/畸形媒体摘要或身份不符 fail-closed。"""
    from qwen3_multimodal_preflight import (
        Qwen3MultimodalPreflightError,
        build_mm1_media_tensor_reference,
        validate_mm1_media_tensor_reference,
    )

    for bad in (None, {}, {"image": {}, "video": {}}):
        try:
            build_mm1_media_tensor_reference(
                bad, model_id="qwen3-vl-4b-instruct", component_ids=["vision_tower"],
            )
        except Qwen3MultimodalPreflightError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")
    summary = {
        "image": {"pixel_values_shape": [1, 3, 64, 64], "token_count_estimate": 16},
        "video": {"pixel_values_shape": [], "token_count_estimate": 0},
    }
    reference = build_mm1_media_tensor_reference(
        summary, model_id="qwen3-vl-4b-instruct", component_ids=["vision_tower"],
    )
    # 身份不符（model_id 不同）拒绝
    try:
        validate_mm1_media_tensor_reference(
            reference, model_id="other-model", component_ids=["vision_tower"],
        )
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("身份不符应拒绝")
    # 篡改 digest 拒绝
    tampered = dict(reference)
    tampered["capacity"] = dict(reference["capacity"])
    tampered["capacity"]["total_media_tokens"] = 999
    try:
        validate_mm1_media_tensor_reference(
            tampered, model_id="qwen3-vl-4b-instruct", component_ids=["vision_tower"],
        )
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("篡改应拒绝")


def test_real_media_tensor_reference_in_response():
    """真实双模型：响应含 media_tensor_reference（与摘要解耦、无权重）。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.8 requires the isolated Qwen3 pipeline sidecar")
    manifest, _inspection, request = _prepared("qwen3-vl-4b-instruct")
    report = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
    )
    assert report["gate_passed"] is True
    reference = report["response"]["media_tensor_reference"]
    assert reference is not None
    assert reference["reference_kind"] == "qwen3_visual_media_tensor_placeholder"
    assert reference["weight_materialized"] is False
    assert reference["capacity"]["total_media_tokens"] > 0
    encoded = json.dumps(report, ensure_ascii=True).lower()
    assert "pixel_values\"" not in encoded
    assert str(ROOT).lower() not in encoded


# ================================================================
# MM1.9：视觉塔执行器输入占位与容量预算接线
# ================================================================

def _sample_reference():
    from qwen3_multimodal_preflight import build_mm1_media_tensor_reference
    return build_mm1_media_tensor_reference(
        {
            "image": {"pixel_values_shape": [1, 3, 64, 64], "dtype": "float16",
                      "token_count_estimate": 16},
            "video": {"pixel_values_shape": [1, 2, 3, 32, 32], "dtype": "float16",
                      "token_count_estimate": 8},
            "output_bytes_estimate": 4096,
            "weight_materialized": False,
            "full_model_materialized": False,
        },
        model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )


def test_visual_input_contract_admitted_within_budget():
    """容量充足 → admitted=True，契约含 grid/token/字节预算且 path-free。"""
    from qwen3_multimodal_preflight import (
        build_mm1_visual_input_contract,
        validate_mm1_visual_input_contract,
    )
    reference = _sample_reference()
    contract = build_mm1_visual_input_contract(
        reference, node_capacity_bytes=8192, safety_margin=1.2,
    )
    validated = validate_mm1_visual_input_contract(contract, media_tensor_reference=reference)
    assert validated["contract_kind"] == "qwen3_visual_input_placeholder"
    assert validated["media_reference_sha256"] == reference["reference_sha256"]
    assert validated["input"]["total_media_tokens"] == 24
    assert validated["capacity"]["admitted"] is True
    assert validated["capacity"]["required_bytes"] == int(4096 * 1.2)
    payload = json.dumps(validated, ensure_ascii=True)
    assert "pixel_values\"" not in payload
    assert "models/" not in payload


def test_visual_input_contract_fails_closed_on_budget_shortage():
    """预算不足 → admitted=False（fail-closed，视觉塔不执行）。"""
    from qwen3_multimodal_preflight import (
        Qwen3MultimodalPreflightError,
        build_mm1_visual_input_contract,
        validate_mm1_visual_input_contract,
    )
    reference = _sample_reference()
    contract = build_mm1_visual_input_contract(
        reference, node_capacity_bytes=1024, safety_margin=1.2,
    )
    assert contract["capacity"]["admitted"] is False
    validated = validate_mm1_visual_input_contract(contract, media_tensor_reference=reference)
    assert validated["capacity"]["admitted"] is False
    # 篡改 admitted（不足却标 True）→ 拒绝
    tampered = dict(contract)
    tampered["capacity"] = dict(contract["capacity"])
    tampered["capacity"]["admitted"] = True
    try:
        validate_mm1_visual_input_contract(tampered, media_tensor_reference=reference)
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("admitted 不一致应拒绝")


def test_real_visual_input_contract_in_response():
    """真实模型：响应含 visual_input_contract（容量比对，无权重）。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.9 requires the isolated Qwen3 pipeline sidecar")
    manifest, _inspection, request = _prepared("qwen3-vl-4b-instruct")
    report = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
    )
    assert report["gate_passed"] is True
    # 未给 node_capacity → 契约为 None（接线点可选）
    assert report["response"]["visual_input_contract"] is None
    # 带容量再跑：契约非 None（接线点生效）
    report2 = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
        node_capacity_bytes=1024 * 1024,
    )
    contract = report2["response"]["visual_input_contract"]
    assert contract is not None
    assert contract["contract_kind"] == "qwen3_visual_input_placeholder"
    assert contract["capacity"]["node_capacity_bytes"] == 1024 * 1024
    encoded = json.dumps(report2, ensure_ascii=True).lower()
    assert "pixel_values\"" not in encoded
    assert str(ROOT).lower() not in encoded
