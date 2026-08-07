import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_worker_protocol import (  # noqa: E402
    WorkerProtocolError,
    build_message,
    canonical_sha256,
    decode_message,
    negotiate_protocol_version,
    stage_input_sha256,
    stage_output_sha256,
    worker_protocol_status,
)


_UNSET = object()


def _component(artifact_id="base_unet", artifact_kind="unet", sha="b" * 64):
    return {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "sha256": sha,
    }


def _manifest(*, pipeline_kind="sd15_pipeline", sha=None):
    manifest = {
        "artifact_id": f"artifact_{pipeline_kind}",
        "pipeline_kind": pipeline_kind,
        "revision": "revision_20260807",
        "components": [_component()],
    }
    manifest["sha256"] = sha or canonical_sha256(manifest)
    return manifest


def _descriptor(
    *,
    purpose="input_image",
    sha="c" * 64,
    blob_id="img_1234567890abcdef",
):
    return {
        "blob_id": blob_id,
        "sha256": sha,
        "size_bytes": 128,
        "content_type": "image/png",
        "width": 512,
        "height": 512,
        "purpose": purpose,
    }


def _transfer_plan(*descriptors):
    ordered = sorted(descriptors, key=lambda item: item["blob_id"])
    return {
        "base_url": "http://100.64.0.10:8000" if ordered else None,
        "downloads": [
            {
                "blob_id": descriptor["blob_id"],
                "lease_id": f"bls_{index:016d}",
                "grant": "a" * 32 + "." + "b" * 43,
            }
            for index, descriptor in enumerate(ordered, start=1)
        ],
    }


def _input_descriptors(root_input):
    descriptors = []
    if "source_blob" in root_input:
        descriptors.append(root_input["source_blob"])
    if root_input.get("mask_blob") is not None:
        descriptors.append(root_input["mask_blob"])
    descriptors.extend(root_input.get("images", []))
    return descriptors


def _hello_payload():
    manifest = _manifest()
    return {
        "node_id": "worker_gpu_1",
        "worker_kind": "pc_diffusion_worker",
        "min_version": 3,
        "max_version": 3,
        "capabilities": {
            "stage_types": ["image_generate", "image_edit", "image_grid"],
            "engines": ["diffusers_sd15"],
            "models": [],
            "max_concurrency": 1,
            "image": {
                "pipeline_kinds": ["sd15_pipeline"],
                "dtypes": ["float16"],
                "max_width": 768,
                "max_height": 768,
                "max_pixels": 768 * 768,
                "max_batch": 1,
                "supports_controlnet": False,
                "supports_step_cancel": True,
                "artifact_manifests": [manifest],
            },
        },
    }


def _generation_input(manifest):
    return {
        "prompt": "a mountain cabin",
        "negative_prompt": "",
        "seed": 19950101,
        "width": 512,
        "height": 512,
        "steps": 20,
        "guidance_scale": 7.5,
        "scheduler": "PNDMScheduler",
        "artifact_manifest_sha256": manifest["sha256"],
    }


def _offer(*, stage_type="image_generate", root_input=None, manifest=_UNSET):
    manifest = _manifest() if manifest is _UNSET else manifest
    root_input = _generation_input(manifest) if root_input is None else root_input
    dependencies = {}
    transfer_plan = _transfer_plan(*_input_descriptors(root_input))
    return {
        "workflow_id": "wf_12345678",
        "stage_id": "image_stage_1",
        "attempt_id": "att_12345678",
        "lease_id": "lease_12345678",
        "lease_epoch": 1,
        "request_id": "request_1",
        "stage_type": stage_type,
        "provider_id": "remote_diffusion_1",
        "lease_expires_at_ms": 2000,
        "root_input": root_input,
        "dependencies": dependencies,
        "input_sha256": stage_input_sha256(
            root_input,
            dependencies,
            transfer_plan,
        ),
        "artifact_manifest": manifest,
        "transfer_plan": transfer_plan,
    }


def test_v3_is_explicit_while_v2_remains_the_preferred_text_protocol():
    hello = build_message(
        "hello",
        _hello_payload(),
        message_id="msg_hello_v3_12345678",
        sent_at_ms=1000,
        version=3,
    )

    assert decode_message(hello.snapshot()) == hello
    assert negotiate_protocol_version(2, 3) == 3
    status = worker_protocol_status()
    assert status["preferred_version"] == 2
    assert status["max_version"] == 3

    v2_payload = _hello_payload()
    v2_payload.update({
        "worker_kind": "pc_full_worker",
        "min_version": 2,
        "max_version": 2,
    })
    with pytest.raises(WorkerProtocolError) as v2_rejects_image_fields:
        build_message(
            "hello",
            v2_payload,
            message_id="msg_hello_v2_12345678",
            sent_at_ms=1000,
            version=2,
        )
    assert v2_rejects_image_fields.value.code == "invalid_fields"


def test_v3_worker_cannot_claim_legacy_models_engines_or_v2_downgrade():
    downgrade = _hello_payload()
    downgrade["min_version"] = 2
    with pytest.raises(WorkerProtocolError) as downgrade_error:
        build_message(
            "hello",
            downgrade,
            message_id="msg_hello_downgrade_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert downgrade_error.value.code == "invalid_version_range"

    mixed_engine = _hello_payload()
    mixed_engine["capabilities"]["engines"].append("pytorch")
    with pytest.raises(WorkerProtocolError) as engine_error:
        build_message(
            "hello",
            mixed_engine,
            message_id="msg_hello_engine_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert engine_error.value.code == "invalid_capabilities"

    legacy_model = _hello_payload()
    legacy_model["capabilities"]["models"] = [{
        "model_id": "legacy_model",
        "engine": "pytorch",
        "format": "safetensors",
        "revision": "revision_1",
        "sha256": "a" * 64,
    }]
    with pytest.raises(WorkerProtocolError) as model_error:
        build_message(
            "hello",
            legacy_model,
            message_id="msg_hello_model_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert model_error.value.code == "invalid_capabilities"


def test_v3_status_runtime_cannot_override_protocol_facts():
    status = worker_protocol_status({
        "preferred_version": 3,
        "max_version": 2,
        "image_v3_schema_ready": False,
        "image_v3_adapter_connected": True,
        "image_v3_data_plane": "http_experimental",
    })

    assert status["preferred_version"] == 2
    assert status["max_version"] == 3
    assert status["image_v3_schema_ready"] is True
    assert status["image_v3_adapter_connected"] is True
    assert status["image_v3_data_plane"] == "http_experimental"


def test_v3_generate_offer_binds_manifest_and_strict_json_input():
    offer = build_message(
        "stage_offer",
        _offer(),
        message_id="msg_offer_v3_12345678",
        sent_at_ms=1000,
        version=3,
    )
    assert offer.payload["stage_type"] == "image_generate"

    mismatched = offer.snapshot()
    mismatched["payload"]["root_input"]["artifact_manifest_sha256"] = "d" * 64
    mismatched["payload"]["input_sha256"] = stage_input_sha256(
        mismatched["payload"]["root_input"],
        {},
        mismatched["payload"]["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as manifest_error:
        decode_message(mismatched)
    assert manifest_error.value.code == "artifact_manifest_mismatch"

    inline_bytes = offer.snapshot()
    inline_bytes["payload"]["root_input"]["image_base64"] = "not-allowed"
    inline_bytes["payload"]["input_sha256"] = stage_input_sha256(
        inline_bytes["payload"]["root_input"],
        {},
        inline_bytes["payload"]["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as inline_error:
        decode_message(inline_bytes)
    assert inline_error.value.code == "invalid_fields"


@pytest.mark.parametrize(
    ("mode", "pipeline_kind", "mode_fields"),
    [
        ("img2img", "sd15_pipeline", {"strength": 0.55}),
        (
            "reference",
            "sd15_ip_adapter",
            {"strength": None, "ip_adapter_scale": 0.6},
        ),
        (
            "inpaint",
            "sd15_inpaint_pipeline",
            {
                "strength": 0.6,
                "mask_blob": _descriptor(
                    purpose="mask",
                    sha="d" * 64,
                    blob_id="img_abcdef1234567890",
                ),
            },
        ),
        (
            "instruction",
            "sd15_instruction_pipeline",
            {
                "strength": None,
                "instruction": "make it winter",
                "image_guidance_scale": 1.0,
            },
        ),
    ],
)
def test_v3_edit_offer_has_mode_specific_blob_and_guidance_contracts(
    mode,
    pipeline_kind,
    mode_fields,
):
    manifest = _manifest(pipeline_kind=pipeline_kind)
    root_input = {
        **_generation_input(manifest),
        "mode": mode,
        "source_blob": _descriptor(),
        "mask_blob": None,
        "strength": None,
        "instruction": None,
        "image_guidance_scale": None,
        "ip_adapter_scale": None,
        **mode_fields,
    }
    if mode == "instruction":
        root_input["prompt"] = root_input["instruction"]
    payload = _offer(
        stage_type="image_edit",
        root_input=root_input,
        manifest=manifest,
    )

    message = build_message(
        "stage_offer",
        payload,
        message_id=f"msg_edit_{mode}_12345678",
        sent_at_ms=1000,
        version=3,
    )
    assert message.payload["root_input"]["mode"] == mode


def test_v3_edit_rejects_mode_pipeline_mismatch_and_mask_leakage():
    manifest = _manifest(pipeline_kind="sd15_instruction_pipeline")
    root_input = {
        **_generation_input(manifest),
        "prompt": "make it winter",
        "mode": "instruction",
        "source_blob": _descriptor(),
        "mask_blob": _descriptor(
            purpose="mask",
            blob_id="img_abcdef1234567890",
        ),
        "strength": None,
        "instruction": "make it winter",
        "image_guidance_scale": 1.0,
        "ip_adapter_scale": None,
    }
    payload = _offer(stage_type="image_edit", root_input=root_input, manifest=manifest)
    with pytest.raises(WorkerProtocolError) as mask_error:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_edit_bad_mask_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert mask_error.value.code == "invalid_image_stage"

    root_input["mask_blob"] = None
    payload = _offer(
        stage_type="image_edit",
        root_input=root_input,
        manifest=_manifest(pipeline_kind="sd15_pipeline"),
    )
    with pytest.raises(WorkerProtocolError) as pipeline_error:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_edit_bad_pipeline_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert pipeline_error.value.code == "artifact_manifest_mismatch"


def test_v3_result_contains_descriptors_not_paths_or_image_bytes():
    output = {
        "image": _descriptor(purpose="output"),
        "metrics": {
            "elapsed_seconds": 5.5,
            "peak_vram_bytes": 3_000_000_000,
            "seed": 19950101,
            "steps_per_second": 3.6,
        },
    }
    payload = {
        "workflow_id": "wf_12345678",
        "stage_id": "image_stage_1",
        "attempt_id": "att_12345678",
        "lease_id": "lease_12345678",
        "lease_epoch": 1,
        "provider_id": "remote_diffusion_1",
        "output": output,
        "output_sha256": stage_output_sha256(
            output,
            _transfer_plan(output["image"]),
        ),
        "metadata": {
            "node_id": "worker_gpu_1",
            "provider_kind": "remote_diffusion_worker",
            "elapsed_seconds": 5.5,
            "peak_vram_bytes": 3_000_000_000,
            "seed": 19950101,
            "artifact_manifest_sha256": "a" * 64,
            "distributed": True,
        },
        "transfer_plan": _transfer_plan(output["image"]),
    }
    result = build_message(
        "stage_result",
        payload,
        message_id="msg_result_v3_12345678",
        sent_at_ms=1000,
        version=3,
    )
    assert result.payload["output"]["image"]["blob_id"].startswith("img_")

    leaked_path = copy.deepcopy(payload)
    leaked_path["output"]["image"]["path"] = "C:/private/image.png"
    leaked_path["output_sha256"] = stage_output_sha256(
        leaked_path["output"],
        leaked_path["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as path_error:
        build_message(
            "stage_result",
            leaked_path,
            message_id="msg_result_path_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert path_error.value.code == "invalid_fields"

    wrong_purpose = copy.deepcopy(payload)
    wrong_purpose["output"]["image"]["purpose"] = "input_image"
    wrong_purpose["output_sha256"] = stage_output_sha256(
        wrong_purpose["output"],
        wrong_purpose["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as purpose_error:
        build_message(
            "stage_result",
            wrong_purpose,
            message_id="msg_result_purpose_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert purpose_error.value.code == "invalid_image_result"

    invalid_blob_id = copy.deepcopy(payload)
    invalid_blob_id["output"]["image"]["blob_id"] = "object.without.image.prefix"
    invalid_blob_id["output_sha256"] = stage_output_sha256(
        invalid_blob_id["output"],
        invalid_blob_id["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as blob_id_error:
        build_message(
            "stage_result",
            invalid_blob_id,
            message_id="msg_result_blob_id_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert blob_id_error.value.code == "invalid_identifier"


def test_v3_image_grid_has_ordered_descriptor_only_contract():
    images = [
        _descriptor(purpose="output", sha="1" * 64),
        {
            **_descriptor(purpose="output", sha="2" * 64),
            "blob_id": "img_abcdef1234567890",
        },
    ]
    root_input = {
        "images": images,
        "minimum_successful": 2,
        "layout": "horizontal",
    }
    payload = _offer(
        stage_type="image_grid",
        root_input=root_input,
        manifest=None,
    )
    message = build_message(
        "stage_offer",
        payload,
        message_id="msg_grid_v3_12345678",
        sent_at_ms=1000,
        version=3,
    )

    assert [item["sha256"] for item in message.payload["root_input"]["images"]] == [
        "1" * 64,
        "2" * 64,
    ]

    payload["root_input"]["minimum_successful"] = 3
    payload["input_sha256"] = stage_input_sha256(
        payload["root_input"],
        {},
        payload["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as threshold:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_grid_bad_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert threshold.value.code == "invalid_image_stage"


def test_v3_transfer_plan_must_exactly_cover_descriptors_and_safe_origin():
    manifest = _manifest(pipeline_kind="sd15_pipeline")
    root_input = {
        **_generation_input(manifest),
        "mode": "img2img",
        "source_blob": _descriptor(),
        "mask_blob": None,
        "strength": 0.55,
        "instruction": None,
        "image_guidance_scale": None,
        "ip_adapter_scale": None,
    }
    payload = _offer(
        stage_type="image_edit",
        root_input=root_input,
        manifest=manifest,
    )
    payload["transfer_plan"]["downloads"] = []
    payload["input_sha256"] = stage_input_sha256(
        payload["root_input"],
        payload["dependencies"],
        payload["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as missing:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_transfer_missing_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert missing.value.code == "transfer_plan_mismatch"

    payload = _offer(
        stage_type="image_edit",
        root_input=root_input,
        manifest=manifest,
    )
    payload["transfer_plan"]["base_url"] = "http://user:password@host:8000"
    payload["input_sha256"] = stage_input_sha256(
        payload["root_input"],
        payload["dependencies"],
        payload["transfer_plan"],
    )
    with pytest.raises(WorkerProtocolError) as credentials:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_transfer_credentials_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert credentials.value.code == "invalid_transfer_plan"

    payload = _offer(
        stage_type="image_edit",
        root_input=root_input,
        manifest=manifest,
    )
    payload["transfer_plan"]["base_url"] = "http://100.64.0.11:8000"
    with pytest.raises(WorkerProtocolError) as digest:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_transfer_digest_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert digest.value.code == "input_digest_mismatch"


def test_v3_requires_a_dedicated_diffusion_worker_identity():
    payload = _hello_payload()
    payload["worker_kind"] = "pc_full_worker"

    with pytest.raises(WorkerProtocolError) as worker_kind:
        build_message(
            "hello",
            payload,
            message_id="msg_hello_kind_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert worker_kind.value.code == "unsupported_worker_kind"


def test_v3_capabilities_reject_unbounded_or_unregistered_artifacts():
    payload = _hello_payload()
    payload["capabilities"]["image"]["max_width"] = 4096
    with pytest.raises(WorkerProtocolError) as unbounded:
        build_message(
            "hello",
            payload,
            message_id="msg_hello_large_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert unbounded.value.code == "invalid_capabilities"

    payload = _hello_payload()
    payload["capabilities"]["image"]["artifact_manifests"][0][
        "pipeline_kind"
    ] = "sd15_inpaint_pipeline"
    changed_manifest = payload["capabilities"]["image"]["artifact_manifests"][0]
    changed_manifest["sha256"] = canonical_sha256({
        key: value for key, value in changed_manifest.items() if key != "sha256"
    })
    with pytest.raises(WorkerProtocolError) as undeclared:
        build_message(
            "hello",
            payload,
            message_id="msg_hello_manifest_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert undeclared.value.code == "invalid_capabilities"


def test_v3_manifest_digest_rejects_component_tampering():
    payload = _hello_payload()
    payload["capabilities"]["image"]["artifact_manifests"][0]["components"][0][
        "sha256"
    ] = "f" * 64

    with pytest.raises(WorkerProtocolError) as tampered:
        build_message(
            "hello",
            payload,
            message_id="msg_hello_tampered_12345678",
            sent_at_ms=1000,
            version=3,
        )
    assert tampered.value.code == "artifact_manifest_digest_mismatch"
