"""Hardware-free MM1 target executor for the Qwen3 network consume boundary.

This executor deliberately does not load a vision model.  It exercises the
same target-local callback used by an isolated sidecar, constructs and checks
the MM1 visual handoff after the receiver has committed the input, and emits a
small deterministic artifact for downstream lifecycle tests.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen3_multimodal_preflight import (  # noqa: E402
    build_mm1_media_tensor_reference,
    mm1_ledger_commit,
    mm1_vision_tower_placement,
    validate_mm1_media_tensor_reference,
)
from qwen3_multimodal_contract import (
    MM1_MAX_VISUAL_TOKENS,
    Qwen3MultimodalContractError,
    build_mm1_handoff_contract,
    build_mm1_transfer_binding,
    validate_mm1_handoff_contract,
    validate_mm1_model_manifest,
)
from qwen3_pipeline_contract import (
    Qwen3PipelineContractError,
    validate_segment_plan,
)


_TRANSFER_ID = re.compile(r"^qtx_[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGED_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STAGED_DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4}
_STAGED_MAX_BYTES = 1 << 40


class Qwen3MultimodalRuntimeError(RuntimeError):
    """Stable error raised by the synthetic MM1 target executor."""

    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)[:128]
        self.reason = str(reason)[:1024]
        super().__init__(self.reason)


class Qwen3MultimodalSyntheticExecutor:
    """Run a deterministic, metadata-only visual boundary smoke.

    The callback receives a target-local input path from the network
    coordinator.  The path is used only inside this process to create a small
    output artifact; no path is returned in the control result.
    """

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        artifact_root: str | Path,
        visual_tokens: int = 16,
        modality: str = "image",
        frame_count: int = 0,
        fail_phase: str | None = None,
    ) -> None:
        self.manifest = validate_mm1_model_manifest(manifest)
        if isinstance(visual_tokens, bool) or not isinstance(visual_tokens, int):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_dimensions_invalid", "visual token count is invalid",
            )
        if not 1 <= visual_tokens <= MM1_MAX_VISUAL_TOKENS:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_dimensions_invalid", "visual token count is outside limits",
            )
        if modality not in self.manifest["processor"]["modalities"]:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_modality_invalid", "synthetic modality is not supported",
            )
        if fail_phase is not None and fail_phase not in {"prefill", "decode"}:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_phase_invalid", "synthetic failure phase is invalid",
            )
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_artifact_root_missing", "synthetic artifact root is unavailable",
            )
        self.artifact_root = root
        self.visual_tokens = int(visual_tokens)
        self.modality = str(modality)
        self.frame_count = int(frame_count)
        self.fail_phase = fail_phase
        self._outputs: dict[str, Path] = {}

    @staticmethod
    def _reference(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_reference_invalid", "target reference is missing",
            )
        required = {
            "artifact_id", "mode", "source_node_id", "target_node_id", "chain_id",
            "generation", "phase", "from_segment", "to_segment", "size_bytes", "sha256",
        }
        if not required.issubset(value):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_reference_invalid", "target reference is incomplete",
            )
        artifact_id = str(value["artifact_id"])
        chain_id = str(value["chain_id"]).lower()
        digest = str(value["sha256"]).lower()
        if _TRANSFER_ID.fullmatch(artifact_id) is None or _SHA256.fullmatch(chain_id) is None or _SHA256.fullmatch(digest) is None:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_reference_invalid", "target reference identity is invalid",
            )
        if value.get("status") != "committed" or value.get("full_model_materialized") is not False:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_reference_uncommitted", "MM1 input reference is not committed",
            )
        return {
            "artifact_id": artifact_id,
            "mode": str(value["mode"]),
            "source_node_id": str(value["source_node_id"]),
            "target_node_id": str(value["target_node_id"]),
            "chain_id": chain_id,
            "generation": value["generation"],
            "phase": str(value["phase"]),
            "from_segment": value["from_segment"],
            "to_segment": value["to_segment"],
            "size_bytes": value["size_bytes"],
            "sha256": digest,
        }

    def __call__(self, input_path: Path, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            reference = self._reference(request.get("reference"))
            phase = str(request["phase"])
            generation = int(request["generation"])
            batch_size = int(request["batch_size"])
            dtype = str(request["dtype"]).lower()
            device = str(request["device"]).lower()
            handoff = build_mm1_handoff_contract(
                manifest=self.manifest,
                text_chain_id=str(request["chain_id"]),
                generation=generation,
                phase=phase,
                source_node_id=reference["source_node_id"],
                target_node_id=reference["target_node_id"],
                artifact={
                    "artifact_id": reference["artifact_id"],
                    "mode": reference["mode"],
                    "size_bytes": reference["size_bytes"],
                    "sha256": reference["sha256"],
                    "status": "committed",
                },
                shape=[batch_size, self.visual_tokens, self.manifest["vision"]["output_hidden_size"]],
                dtype=dtype,
                device=device,
                modality=self.modality,
                frame_count=self.frame_count,
            )
            binding = build_mm1_transfer_binding(
                handoff=handoff,
                manifest=self.manifest,
                transfer_reference=reference,
            )
            if not input_path.is_file():
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_input_missing", "target input artifact is unavailable",
                )
            digest = hashlib.sha256()
            with input_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            segment_index = int(request.get("segment_index", 0))
            output = self.artifact_root / (
                f"qwen3-consume-mm1-{reference['artifact_id']}-{phase}-"
                f"{generation}-{segment_index}.pt"
            )
            output.write_bytes(b"qwen3-mm1-synthetic-v1:" + digest.digest())
            self._outputs[reference["artifact_id"]] = output
            if self.fail_phase == phase:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_synthetic_failure", "synthetic MM1 execution failed",
                )
            sequence_length = int(request["sequence_length"])
            return {
                "status": "synthetic_mm1_executed",
                "execution": {
                    "segment_materialized": False,
                    "artifact_bytes": reference["size_bytes"],
                    "artifact_sha256": reference["sha256"],
                    "full_model_materialized": False,
                },
                "hidden_handoff": {
                    "shape": [batch_size, self.visual_tokens, self.manifest["vision"]["output_hidden_size"]],
                    "dtype": dtype,
                    "device": device,
                    "has_next_segment": bool(request["has_next_segment"]),
                },
                "kv_contract": {
                    "present": phase == "decode",
                    "shape": [batch_size, sequence_length],
                },
                "mm1_binding_sha256": binding["binding_sha256"],
                "output_path": str(output),
            }
        except Qwen3MultimodalContractError as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_contract_invalid", str(exc),
            ) from exc
        except Qwen3MultimodalRuntimeError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_execution_invalid", "synthetic MM1 request is invalid",
            ) from exc

    def cleanup(self, request: Mapping[str, Any], reason_code: str = "cleanup") -> None:
        transfer_id = str(request.get("transfer_id", ""))
        if transfer_id:
            path = self._outputs.pop(transfer_id, None)
            if path is not None:
                path.unlink(missing_ok=True)
            return
        for path in self._outputs.values():
            path.unlink(missing_ok=True)
        self._outputs.clear()


class Qwen3MultimodalSidecarAdapter:
    """Attach MM1 metadata validation to an existing isolated sidecar callback.

    The wrapped executor still owns model loading and local output cleanup. The
    adapter projects the visual boundary from the already validated manifest;
    it does not claim that a text-only sidecar performed visual understanding.
    """

    def __init__(
        self,
        executor: Any,
        *,
        manifest: Mapping[str, Any],
        visual_tokens: int | None = None,
        modality: str = "image",
        frame_count: int = 0,
    ) -> None:
        if not callable(executor):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_executor_invalid", "sidecar executor is not callable",
            )
        self.executor = executor
        self.manifest = validate_mm1_model_manifest(manifest)
        if modality not in self.manifest["processor"]["modalities"]:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_modality_invalid", "sidecar modality is not supported",
            )
        if visual_tokens is not None and (
            isinstance(visual_tokens, bool)
            or not isinstance(visual_tokens, int)
            or not 1 <= visual_tokens <= MM1_MAX_VISUAL_TOKENS
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_dimensions_invalid", "sidecar visual token count is invalid",
            )
        self.visual_tokens = visual_tokens
        self.modality = str(modality)
        self.frame_count = int(frame_count)

    def __call__(self, input_path: Path, request: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            reference = Qwen3MultimodalSyntheticExecutor._reference(request.get("reference"))
            batch_size = int(request["batch_size"])
            sequence_length = int(request["sequence_length"])
            visual_tokens = int(self.visual_tokens or sequence_length)
            handoff = build_mm1_handoff_contract(
                manifest=self.manifest,
                text_chain_id=str(request["chain_id"]),
                generation=int(request["generation"]),
                phase=str(request["phase"]),
                source_node_id=reference["source_node_id"],
                target_node_id=reference["target_node_id"],
                artifact={
                    "artifact_id": reference["artifact_id"],
                    "mode": reference["mode"],
                    "size_bytes": reference["size_bytes"],
                    "sha256": reference["sha256"],
                    "status": "committed",
                },
                shape=[batch_size, visual_tokens, self.manifest["vision"]["output_hidden_size"]],
                dtype=str(request["dtype"]),
                device=str(request["device"]),
                modality=self.modality,
                frame_count=self.frame_count,
            )
            binding = build_mm1_transfer_binding(
                handoff=handoff,
                manifest=self.manifest,
                transfer_reference=reference,
            )
            report = self.executor(input_path, request)
            if not isinstance(report, Mapping):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_executor_invalid", "sidecar metadata is not an object",
                )
            result = dict(report)
            hidden = result.get("hidden_handoff")
            if hidden is not None and not isinstance(hidden, Mapping):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_hidden_invalid", "sidecar hidden metadata is invalid",
                )
            if hidden:
                shape = hidden.get("shape")
                if shape is not None and (
                    not isinstance(shape, (list, tuple))
                    or len(shape) != 3
                    or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
                    or int(shape[0]) != batch_size
                    or int(shape[2]) != int(self.manifest["text"]["hidden_size"])
                ):
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_hidden_invalid", "sidecar hidden shape does not match manifest",
                    )
                if hidden.get("dtype") is not None and str(hidden["dtype"]).lower() != str(request["dtype"]).lower():
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_hidden_invalid", "sidecar hidden dtype does not match request",
                    )
                if hidden.get("device") is not None and str(hidden["device"]).lower() != str(request["device"]).lower():
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_hidden_invalid", "sidecar hidden device does not match request",
                    )
            result["mm1_binding_sha256"] = binding["binding_sha256"]
            result["mm1_metadata"] = {
                "binding_sha256": binding["binding_sha256"],
                "visual_shape": list(handoff["tensor"]["shape"]),
                "dtype": handoff["tensor"]["dtype"],
                "device": handoff["tensor"]["device"],
                "modality": self.modality,
            }
            return result
        except Qwen3MultimodalContractError as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_contract_invalid", str(exc),
            ) from exc
        except Qwen3MultimodalRuntimeError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_execution_invalid", "sidecar MM1 request is invalid",
            ) from exc

    def cleanup(self, request: Mapping[str, Any], reason_code: str = "cleanup") -> None:
        cleanup = getattr(self.executor, "cleanup", None)
        if callable(cleanup):
            cleanup(request, reason_code)


def run_mm1_visual_tower_skeleton(
    placement: Mapping[str, Any],
    media_tensor_reference: Mapping[str, Any] | None,
    *,
    text_only: bool,
) -> dict[str, Any]:
    """MM1.12：视觉塔执行器骨架——按放置决策选执行路径（不加载权重）。

    - text_only=True：visual_path="skipped"（纯文本守卫：全程不触碰视觉塔，
      media 参考必须为 None）；
    - media + vision_tower_active：visual_path="placeholder_ready"（占位执行）；
    - media + inactive：fail-closed（Qwen3MultimodalRuntimeError）；
    - 一致性：placement.request_has_media 必须与 text_only 参数一致。
    """
    if not isinstance(text_only, bool) or not isinstance(placement, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_placement_invalid", "visual tower placement flags are invalid",
        )
    request_has_media = placement.get("request_has_media")
    if not isinstance(request_has_media, bool):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_placement_invalid", "request_has_media must be boolean",
        )
    if request_has_media == text_only:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_placement_inconsistent",
            "visual tower placement contradicts the text-only request flag",
        )
    if text_only:
        if media_tensor_reference is not None:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_text_only_media_reference",
                "text-only requests must not carry a media tensor reference",
            )
        return {
            "schema_version": 1,
            "skeleton_kind": "qwen3_visual_tower_skeleton",
            "visual_path": "skipped",
            "vision_tower_active": False,
            "reason": "text_only_request_guard",
            "weight_materialized": False,
            "full_model_materialized": False,
        }
    active = placement.get("vision_tower_active")
    if not isinstance(active, bool):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_placement_invalid", "vision_tower_active must be boolean",
        )
    if not active:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_vision_tower_inactive",
            "vision tower is inactive for a media request",
        )
    if media_tensor_reference is None:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_reference_missing",
            "media requests require a media tensor reference",
        )
    if not isinstance(media_tensor_reference, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_reference_invalid", "media tensor reference is invalid",
        )
    try:
        reference_model_id = media_tensor_reference["model_id"]
        reference_components = media_tensor_reference["component_ids"]
    except KeyError as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_reference_invalid", "media tensor reference identity is missing",
        ) from exc
    reference = validate_mm1_media_tensor_reference(
        media_tensor_reference,
        model_id=reference_model_id,
        component_ids=reference_components,
    )
    return {
        "schema_version": 1,
        "skeleton_kind": "qwen3_visual_tower_skeleton",
        "visual_path": "placeholder_ready",
        "vision_tower_active": True,
        "media_reference_sha256": reference["reference_sha256"],
        "total_media_tokens": int(reference["capacity"]["total_media_tokens"]),
        "weight_materialized": False,
        "full_model_materialized": False,
    }


def run_mm1_visual_placeholder_execution(
    skeleton: Mapping[str, Any],
    media_tensor_reference: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """MM1.13：视觉塔占位执行——以媒体参考消费产出 path-free 视觉特征摘要。

    不加载视觉塔：特征 shape = [1, media_tokens, vision.output_hidden_size]
    （合成特征，synthetic=true），只投影 shape/dtype/token 摘要。
    """
    if skeleton.get("visual_path") != "placeholder_ready":
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_skeleton_not_ready",
            "visual placeholder execution requires a ready skeleton path",
        )
    safe_manifest = validate_mm1_model_manifest(manifest)
    if not isinstance(media_tensor_reference, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_reference_invalid", "media tensor reference is invalid",
        )
    try:
        reference_model_id = media_tensor_reference["model_id"]
        reference_components = media_tensor_reference["component_ids"]
    except KeyError as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_reference_invalid", "media tensor reference identity is missing",
        ) from exc
    reference = validate_mm1_media_tensor_reference(
        media_tensor_reference,
        model_id=reference_model_id,
        component_ids=reference_components,
    )
    vision = safe_manifest["vision"]
    tokens = int(reference["capacity"]["total_media_tokens"])
    hidden = int(vision["output_hidden_size"])
    feature = {
        "schema_version": 1,
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": reference["model_id"],
        "media_reference_sha256": reference["reference_sha256"],
        "tensor": {
            "shape": [1, tokens, hidden],
            "dtype": str(vision.get("dtype") or "float16"),
            "device": "cpu",
        },
        "synthetic": True,
        "weight_materialized": False,
        "full_model_materialized": False,
    }
    return feature


def bind_mm1_visual_feature_handoff(
    feature: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    text_chain_id: str,
    generation: int,
    phase: str,
    source_node_id: str,
    target_node_id: str,
    modality: str = "image",
) -> dict[str, Any]:
    """MM1.13：视觉特征绑定回文本段 hidden handoff（visual_to_text 边界）。

    消费一致性：handoff tensor shape 必须与特征 shape 一致。
    """
    if feature.get("feature_kind") != "qwen3_visual_feature_placeholder":
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_feature_invalid",
            "visual handoff requires a placeholder feature summary",
        )
    tensor = feature.get("tensor") or {}
    shape = tensor.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_feature_shape_invalid",
            "visual feature shape must be [batch, tokens, hidden]",
        )
    from qwen3_multimodal_contract import build_mm1_handoff_contract
    contract = build_mm1_handoff_contract(
        manifest=manifest,
        text_chain_id=text_chain_id,
        generation=generation,
        phase=phase,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        artifact={
            "artifact_id": "a" * 64,
            "mode": "local",
            "size_bytes": max(1, int(shape[1]) * int(shape[2]) * 2),
            "sha256": str(feature.get("media_reference_sha256") or "a" * 64),
            "status": "committed",
        },
        shape=[int(v) for v in shape],
        dtype=str(tensor.get("dtype") or "float16"),
        device=str(tensor.get("device") or "cpu"),
        modality=modality,
        item_count=1,
        frame_count=0,
    )
    return contract


def _synthetic_media_summary(media_smoke: Mapping[str, Any]) -> dict[str, Any]:
    """MM1.14：从受限 media_smoke 参数构造合成媒体摘要（模拟 MM1.7 输出）。"""
    image_size = media_smoke.get("image_size") or (32, 32)
    video_size = media_smoke.get("video_size") or (32, 32)
    video_frames = int(media_smoke.get("video_frames", 2))
    image_h, image_w = (int(v) for v in image_size)
    video_h, video_w = (int(v) for v in video_size)
    if (
        not 8 <= image_h <= 1024 or not 8 <= image_w <= 1024
        or not 8 <= video_h <= 1024 or not 8 <= video_w <= 1024
        or not 1 <= video_frames <= 32
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_bounds",
            "synthetic media dimensions are outside the MM1 contract limits",
        )
    patch = 16
    image_tokens = max(1, (image_h // patch) * (image_w // patch))
    video_tokens = max(1, (video_h // patch) * (video_w // patch))
    return {
        "image": {
            "pixel_values_shape": [1, 3, image_h, image_w],
            "dtype": "float16",
            "token_count_estimate": image_tokens,
        },
        "video": {
            "pixel_values_shape": [1, video_frames, 3, video_h, video_w],
            "dtype": "float16",
            "token_count_estimate": video_tokens,
        },
        "output_bytes_estimate": (image_tokens + video_tokens) * 2048,
        "weight_materialized": False,
        "full_model_materialized": False,
    }


def run_mm1_synthetic_visual_chain(
    *,
    manifest: Mapping[str, Any],
    media_smoke: Mapping[str, Any] | None,
    node_capacity_bytes: int,
    text_only: bool = False,
    text_chain_id: str = "a" * 64,
    generation: int = 1,
    phase: str = "prefill",
    source_node_id: str = "node-a",
    target_node_id: str = "node-b",
) -> dict[str, Any]:
    """MM1.14：MM1.7→MM1.13 全链 CPU 合成端到端回归。

    合成媒体 → 摘要 → 张量参考 → 容量账本 → 放置决策 → 骨架 →
    占位执行 → visual_to_text handoff；全程不加载视觉塔/文本权重。
    text_only 路径跳过媒体链（骨架 skipped，文本段独立）。
    """
    safe_manifest = validate_mm1_model_manifest(manifest)
    components = [
        str(item) for item in safe_manifest.get("component_ids", [])
    ] or ["vision_tower", "text_segment_0"]

    if text_only:
        placement = mm1_vision_tower_placement(
            _synthetic_ledger(safe_manifest, node_capacity_bytes, media_bytes=0),
            request_has_media=False,
            vision_tower_bytes=0,
        )
        skeleton = run_mm1_visual_tower_skeleton(placement, None, text_only=True)
        return {
            "chain_kind": "qwen3_synthetic_visual_chain",
            "text_only": True,
            "visual_path": skeleton["visual_path"],
            "vision_tower_active": False,
            "weight_materialized": False,
            "full_model_materialized": False,
        }

    if not isinstance(media_smoke, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_media_required",
            "media requests require media_smoke parameters",
        )
    summary = _synthetic_media_summary(media_smoke)
    reference = build_mm1_media_tensor_reference(
        summary,
        model_id=safe_manifest["model_id"],
        component_ids=components,
    )
    vision = safe_manifest["vision"]
    tower_bytes = int(
        vision.get("hidden_size", 0) * 4096 * vision.get("depth", 1) * 2
    ) or 400_000_000
    ledger = mm1_ledger_commit(
        [
            {"entry_id": "vision_tower", "kind": "vision_tower_weights", "bytes": tower_bytes},
            {"entry_id": "media_1", "kind": "media_input",
             "bytes": int(summary["output_bytes_estimate"])},
            {"entry_id": "text_0", "kind": "text_segment",
             "bytes": int(safe_manifest["text"]["hidden_size"]) * 2048 * 2},
        ],
        ledger_id="node-e2e",
        node_capacity_bytes=int(node_capacity_bytes),
    )
    placement = mm1_vision_tower_placement(
        ledger, request_has_media=True, vision_tower_bytes=tower_bytes,
    )
    skeleton = run_mm1_visual_tower_skeleton(placement, reference, text_only=False)
    feature = run_mm1_visual_placeholder_execution(
        skeleton, reference, manifest=safe_manifest,
    )
    handoff = bind_mm1_visual_feature_handoff(
        feature,
        manifest=safe_manifest,
        text_chain_id=text_chain_id,
        generation=generation,
        phase=phase,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        modality="image",
    )
    return {
        "chain_kind": "qwen3_synthetic_visual_chain",
        "text_only": False,
        "media_tokens": int(reference["capacity"]["total_media_tokens"]),
        "ledger_admitted": bool(ledger["capacity"]["admitted"]),
        "vision_tower_active": bool(placement["vision_tower_active"]),
        "visual_path": skeleton["visual_path"],
        "feature_shape": list(feature["tensor"]["shape"]),
        "handoff_boundary": handoff["boundary"],
        "handoff_tokens": int(handoff["tensor"]["shape"][1]),
        "weight_materialized": False,
        "full_model_materialized": False,
    }


def _synthetic_ledger(
    manifest: Mapping[str, Any], node_capacity_bytes: int, *, media_bytes: int,
) -> dict[str, Any]:
    vision = manifest["vision"]
    tower_bytes = int(vision.get("hidden_size", 0) * 4096 * vision.get("depth", 1) * 2) or 400_000_000
    text_bytes = int(manifest["text"]["hidden_size"]) * 2048 * 2
    return mm1_ledger_commit(
        [
            {"entry_id": "vision_tower", "kind": "vision_tower_weights", "bytes": tower_bytes},
            {"entry_id": "media_1", "kind": "media_input", "bytes": media_bytes},
            {"entry_id": "text_0", "kind": "text_segment", "bytes": text_bytes},
        ],
        ledger_id="node-e2e",
        node_capacity_bytes=int(node_capacity_bytes),
    )


def run_mm1_synthetic_text_decode(
    *,
    vision_feature: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prompt_tokens: int = 4,
    batch_size: int = 1,
    sequence_length: int = 64,
    text_chain_id: str = "a" * 64,
    generation: int = 2,
) -> dict[str, Any]:
    """MM1.16：真实视觉特征接入文本段合成解码（CPU 混合链）。

    文本段零权重（合成解码）：序列布局 = 视觉 token 区 + 文本 token 区，
    visual token 位置与特征 token 数对齐；合成 decode 不宣称真实语义。
    """
    safe_manifest = validate_mm1_model_manifest(manifest)
    if vision_feature.get("feature_kind") != "qwen3_visual_feature_placeholder":
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_feature_invalid",
            "synthetic text decode requires a visual feature summary",
        )
    tensor = vision_feature.get("tensor") or {}
    feature_shape = tensor.get("shape")
    if not isinstance(feature_shape, (list, tuple)) or len(feature_shape) != 3:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_feature_shape_invalid",
            "visual feature shape must be [batch, tokens, hidden]",
        )
    visual_tokens = int(feature_shape[1])
    hidden_dim = int(feature_shape[2])
    try:
        prompt_tokens = int(prompt_tokens)
        batch_size = int(batch_size)
        sequence_length = int(sequence_length)
    except (TypeError, ValueError):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_dims_invalid",
            "decode dimensions are invalid",
        )
    if (
        not 1 <= prompt_tokens <= 8192
        or not 1 <= batch_size <= 64
        or not 1 <= sequence_length <= 65536
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_dims_invalid",
            "decode dimensions are outside the MM1 contract limits",
        )
    text_hidden = int(safe_manifest["text"]["hidden_size"])
    if hidden_dim != text_hidden:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_hidden_mismatch",
            "visual feature hidden must match the text segment hidden",
        )
    # visual token 位置对齐：视觉区 [0:visual_tokens] + 文本区
    layout = {
        "visual_span": [0, visual_tokens],
        "text_span": [visual_tokens, visual_tokens + prompt_tokens],
        "total_sequence": visual_tokens + prompt_tokens,
        "batch_size": batch_size,
    }
    if layout["total_sequence"] > sequence_length:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sequence_overflow",
            "visual + text tokens exceed the sequence budget",
        )
    return {
        "schema_version": 1,
        "decode_kind": "qwen3_synthetic_text_decode",
        "model_id": safe_manifest["model_id"],
        "text_chain_id": text_chain_id,
        "generation": generation,
        "input": {
            "visual_tokens": visual_tokens,
            "prompt_tokens": prompt_tokens,
            "hidden_size": hidden_dim,
            "layout": layout,
        },
        "synthetic": True,
        "text_weights_loaded": False,
        "weight_materialized": False,
        "full_model_materialized": False,
    }


def run_mm1_synthetic_hybrid_chain(
    *,
    vision_feature: Mapping[str, Any],
    manifest: Mapping[str, Any],
    media_smoke: Mapping[str, Any],
    node_capacity_bytes: int,
    prompt_tokens: int = 4,
    sequence_length: int = 128,
    text_chain_id: str = "a" * 64,
    generation: int = 3,
) -> dict[str, Any]:
    """MM1.17：CPU 混合链端到端合成回归——真实视觉特征 + 合成媒体链 +
    文本段合成解码（视觉塔真实、文本零权重、token 对齐）。

    媒体链（摘要→参考→账本→放置→骨架）走 MM1.14 合成路径；占位执行
    由真实视觉特征替代（MM1.15 产出）；文本段合成解码（MM1.16）。
    """
    safe_manifest = validate_mm1_model_manifest(manifest)
    if vision_feature.get("feature_kind") != "qwen3_visual_feature_placeholder":
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_feature_invalid",
            "hybrid chain requires a visual feature summary",
        )
    # 1) 合成媒体链（MM1.14 前置：摘要 → 张量参考 → 账本 → 放置 → 骨架）
    summary = _synthetic_media_summary(media_smoke)
    reference = build_mm1_media_tensor_reference(
        summary,
        model_id=safe_manifest["model_id"],
        component_ids=[str(item) for item in safe_manifest.get("component_ids", [])]
        or ["vision_tower", "text_segment_0"],
    )
    vision = safe_manifest["vision"]
    tower_bytes = int(vision.get("hidden_size", 0) * 4096 * vision.get("depth", 1) * 2) or 400_000_000
    ledger = mm1_ledger_commit(
        [
            {"entry_id": "vision_tower", "kind": "vision_tower_weights", "bytes": tower_bytes},
            {"entry_id": "media_1", "kind": "media_input",
             "bytes": int(summary["output_bytes_estimate"])},
            {"entry_id": "text_0", "kind": "text_segment",
             "bytes": int(safe_manifest["text"]["hidden_size"]) * 2048 * 2},
        ],
        ledger_id="node-hybrid",
        node_capacity_bytes=int(node_capacity_bytes),
    )
    placement = mm1_vision_tower_placement(
        ledger, request_has_media=True, vision_tower_bytes=tower_bytes,
    )
    skeleton = run_mm1_visual_tower_skeleton(placement, reference, text_only=False)

    # 2) 真实视觉特征（替代占位执行——MM1.15 产出）接入文本段合成解码
    decode = run_mm1_synthetic_text_decode(
        vision_feature=vision_feature,
        manifest=safe_manifest,
        prompt_tokens=prompt_tokens,
        sequence_length=sequence_length,
        text_chain_id=text_chain_id,
        generation=generation,
    )
    feature_tokens = int(
        (vision_feature.get("tensor") or {}).get("shape", [0, 0, 0])[1],
    )
    return {
        "chain_kind": "qwen3_synthetic_hybrid_chain",
        "model_id": safe_manifest["model_id"],
        "ledger_admitted": bool(ledger["capacity"]["admitted"]),
        "vision_tower_active": bool(placement["vision_tower_active"]),
        "visual_path": skeleton["visual_path"],
        "media_tokens": int(reference["capacity"]["total_media_tokens"]),
        "feature_tokens": feature_tokens,
        "decode": decode,
        "consistency": {
            "tokens_match": bool(feature_tokens == decode["input"]["visual_tokens"]),
        },
        "vision_tower_weight_materialized": bool(
            vision_feature.get("weight_materialized", False),
        ),
        "text_weight_materialized": False,
        "full_model_materialized": False,
    }


def _staged_digest(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text contract is not canonical JSON",
        ) from exc
    if len(encoded) > 64 * 1024:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_oversize",
            "staged text contract exceeds 64 KiB",
        )
    return hashlib.sha256(encoded).hexdigest()


def _staged_id(value: Any, field: str) -> str:
    result = str(value or "")
    if _STAGED_ID.fullmatch(result) is None:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            f"{field} is invalid",
        )
    return result


def _staged_int(value: Any, field: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            f"{field} must be an integer",
        )
    minimum = 1 if positive else 0
    if not minimum <= value <= _STAGED_MAX_BYTES:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            f"{field} is outside limits",
        )
    return int(value)


def _reject_staged_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                "path" in lowered
                or "file" in lowered
                or "pixel" in lowered
                or lowered in {"prompt", "prompt_text", "prompt_content"}
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_staged_sensitive",
                    "staged text metadata contains a sensitive field",
                )
            _reject_staged_sensitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_staged_sensitive(item)


def _normalise_staged_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    total_layers: int,
) -> list[dict[str, Any]]:
    if isinstance(segments, (str, bytes)):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text segment plan is invalid",
        )
    try:
        raw_segments = [dict(item) if isinstance(item, Mapping) else item for item in segments]
    except TypeError as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text segment plan is invalid",
        ) from exc
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_contract_invalid",
                "staged text segment must be an object",
            )
        if not isinstance(raw.get("has_embedding"), bool) or not isinstance(raw.get("has_lm_head"), bool):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_contract_invalid",
                "staged text ownership flags must be boolean",
            )
        _reject_staged_sensitive(raw)
    try:
        topology = validate_segment_plan(raw_segments, total_layers=total_layers)
    except (Qwen3PipelineContractError, TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_topology_invalid",
            "staged text segment topology is invalid",
        ) from exc
    result: list[dict[str, Any]] = []
    for raw, segment in zip(raw_segments, topology):
        node_id = _staged_id(raw.get("node_id"), "node_id")
        assignment_sha = str(raw.get("assignment_manifest_sha256") or "")
        if _SHA256.fullmatch(assignment_sha) is None:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_contract_invalid",
                "assignment manifest digest is invalid",
            )
        required = _staged_int(raw.get("required_bytes"), "required_bytes")
        activation = _staged_int(raw.get("activation_bytes"), "activation_bytes", positive=False)
        capacity = _staged_int(raw.get("node_capacity_bytes"), "node_capacity_bytes")
        peak = required + activation
        if peak > capacity:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_capacity_rejected",
                "staged text segment exceeds node capacity",
            )
        dtype = str(raw.get("dtype") or "").lower().removeprefix("torch.")
        device = str(raw.get("device") or "").lower()
        if dtype not in _STAGED_DTYPES or device not in {"cpu", "cuda"}:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_contract_invalid",
                "staged text segment dtype or device is unsupported",
            )
        result.append({
            "segment_index": int(segment["segment_index"]),
            "node_id": node_id,
            "layer_range": [int(value) for value in segment["layer_range"]],
            "has_embedding": bool(segment["has_embedding"]),
            "has_lm_head": bool(segment["has_lm_head"]),
            "dtype": dtype,
            "device": device,
            "required_bytes": required,
            "activation_bytes": activation,
            "peak_bytes": peak,
            "node_capacity_bytes": capacity,
            "remaining_bytes": capacity - peak,
            "admitted": True,
            "assignment_manifest_sha256": assignment_sha,
        })
    return result


def _normalise_staged_feature(
    vision_feature: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(vision_feature, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "staged text execution requires a visual feature summary",
        )
    _reject_staged_sensitive(vision_feature)
    if vision_feature.get("feature_kind") != "qwen3_visual_feature_placeholder":
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature kind is unsupported",
        )
    if vision_feature.get("model_id") != manifest["model_id"]:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature model identity does not match",
        )
    if vision_feature.get("full_model_materialized") is not False:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_full_model_forbidden",
            "visual feature cannot materialize the full model",
        )
    if not isinstance(vision_feature.get("weight_materialized"), bool):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature weight state is invalid",
        )
    if not isinstance(vision_feature.get("synthetic"), bool):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature synthetic state is invalid",
        )
    digest = str(vision_feature.get("media_reference_sha256") or "")
    if _SHA256.fullmatch(digest) is None:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature digest is invalid",
        )
    tensor = vision_feature.get("tensor")
    if not isinstance(tensor, Mapping) or set(tensor) != {"shape", "dtype", "device"}:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature tensor metadata is invalid",
        )
    shape = tensor.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature shape must be [batch,tokens,hidden]",
        )
    if int(shape[0]) > 64 or int(shape[1]) > MM1_MAX_VISUAL_TOKENS:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature dimensions exceed limits",
        )
    if int(shape[2]) != int(manifest["text"]["hidden_size"]):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_hidden_mismatch",
            "visual feature hidden must match the first text segment",
        )
    dtype = str(tensor.get("dtype") or "").lower().removeprefix("torch.")
    device = str(tensor.get("device") or "").lower()
    if dtype not in _STAGED_DTYPES or device not in {"cpu", "cuda"}:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_feature_invalid",
            "visual feature dtype or device is unsupported",
        )
    size_bytes = int(shape[0]) * int(shape[1]) * int(shape[2]) * _STAGED_DTYPES[dtype]
    return {
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": manifest["model_id"],
        "media_reference_sha256": digest,
        "tensor": {"shape": [int(item) for item in shape], "dtype": dtype, "device": device},
        "synthetic": bool(vision_feature.get("synthetic", False)),
        "weight_materialized": bool(vision_feature["weight_materialized"]),
        "full_model_materialized": False,
        "size_bytes": size_bytes,
    }


def _build_staged_input_layout(
    feature: Mapping[str, Any],
    *,
    prompt_tokens: Any,
    sequence_length: Any,
) -> dict[str, Any]:
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or not 1 <= prompt_tokens <= 8192
        or isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or not 1 <= sequence_length <= 65_536
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_layout_invalid",
            "staged text sequence dimensions are invalid",
        )
    shape = feature["tensor"]["shape"]
    visual_tokens = int(shape[1])
    total_sequence = visual_tokens + int(prompt_tokens)
    if total_sequence > sequence_length:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_sequence_overflow",
            "visual and text tokens exceed the staged sequence budget",
        )
    dtype = str(feature["tensor"]["dtype"])
    return {
        "batch_size": int(shape[0]),
        "visual_tokens": visual_tokens,
        "prompt_tokens": int(prompt_tokens),
        "visual_span": [0, visual_tokens],
        "text_span": [visual_tokens, total_sequence],
        "total_sequence": total_sequence,
        "sequence_budget": int(sequence_length),
        "hidden_size": int(shape[2]),
        "dtype": dtype,
        "device": str(feature["tensor"]["device"]),
        "minimum_activation_bytes": int(shape[0]) * total_sequence * int(shape[2]) * _STAGED_DTYPES[dtype],
    }


def build_mm1_staged_text_contract(
    *,
    vision_feature: Mapping[str, Any],
    manifest: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    text_chain_id: str,
    generation: int,
    phase: str = "prefill",
    source_node_id: str = "vision-node",
    prompt_tokens: int = 4,
    sequence_length: int = 128,
) -> dict[str, Any]:
    """Bind a visual feature reference to the first staged text segment."""
    safe_manifest = validate_mm1_model_manifest(manifest)
    feature = _normalise_staged_feature(vision_feature, manifest=safe_manifest)
    input_layout = _build_staged_input_layout(
        feature, prompt_tokens=prompt_tokens, sequence_length=sequence_length,
    )
    total_layers = int(safe_manifest["text"]["num_hidden_layers"])
    plan = _normalise_staged_segments(segments, total_layers=total_layers)
    if any(segment["activation_bytes"] < input_layout["minimum_activation_bytes"] for segment in plan):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_capacity_rejected",
            "staged text activation budget is below the combined input requirement",
        )
    first = plan[0]
    source = _staged_id(source_node_id, "source_node_id")
    if source == first["node_id"]:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "visual and first text nodes must differ",
        )
    if feature["tensor"]["dtype"] != first["dtype"] or feature["tensor"]["device"] != first["device"]:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_handoff_mismatch",
            "visual feature dtype/device does not match the first text segment",
        )
    try:
        handoff = build_mm1_handoff_contract(
            manifest=safe_manifest,
            text_chain_id=text_chain_id,
            generation=generation,
            phase=phase,
            source_node_id=source,
            target_node_id=first["node_id"],
            artifact={
                "artifact_id": feature["media_reference_sha256"],
                "mode": "local",
                "size_bytes": feature["size_bytes"],
                "sha256": feature["media_reference_sha256"],
                "status": "committed",
            },
            shape=feature["tensor"]["shape"],
            dtype=feature["tensor"]["dtype"],
            device=feature["tensor"]["device"],
            modality="image",
            item_count=1,
            frame_count=0,
        )
    except Qwen3MultimodalContractError as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_handoff_mismatch",
            "staged visual handoff could not be constructed",
        ) from exc
    values = {
        "schema_version": 1,
        "contract_kind": "qwen3_mm1_staged_text_execution",
        "model_id": safe_manifest["model_id"],
        "model_family": safe_manifest["model_family"],
        "manifest_sha256": safe_manifest["manifest_sha256"],
        "text_chain_id": handoff["text_chain_id"],
        "generation": handoff["generation"],
        "phase": handoff["phase"],
        "total_layers": total_layers,
        "entry_segment_index": 0,
        "segment_plan": plan,
        "visual_feature": feature,
        "visual_handoff": handoff,
        "input_layout": input_layout,
        "execution": {
            "mode": "staged_segment",
            "state": "planned",
            "text_weights_loaded": False,
            "segment_materialized": False,
            "full_model_materialized": False,
        },
        "cleanup": {"required": True, "completed": False},
    }
    values["contract_sha256"] = _staged_digest(values)
    return validate_mm1_staged_text_contract(values, manifest=safe_manifest)


def validate_mm1_staged_text_contract(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    safe_manifest = validate_mm1_model_manifest(manifest)
    required = {
        "schema_version", "contract_kind", "model_id", "model_family", "manifest_sha256",
        "text_chain_id", "generation", "phase", "total_layers", "entry_segment_index",
        "segment_plan", "visual_feature", "visual_handoff", "execution", "cleanup",
        "input_layout", "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text contract fields are invalid",
        )
    try:
        contract = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text contract is not JSON serializable",
        ) from exc
    if contract["schema_version"] != 1 or contract["contract_kind"] != "qwen3_mm1_staged_text_execution":
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text contract version or kind is invalid",
        )
    if (
        contract["model_id"] != safe_manifest["model_id"]
        or contract["model_family"] != safe_manifest["model_family"]
        or contract["manifest_sha256"] != safe_manifest["manifest_sha256"]
        or contract["total_layers"] != safe_manifest["text"]["num_hidden_layers"]
        or contract["entry_segment_index"] != 0
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text contract model or entry identity does not match",
        )
    plan = _normalise_staged_segments(
        contract["segment_plan"], total_layers=int(contract["total_layers"]),
    )
    if plan != contract["segment_plan"]:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text segment plan is not canonical",
        )
    feature = _normalise_staged_feature(contract["visual_feature"], manifest=safe_manifest)
    if feature != contract["visual_feature"]:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged visual feature is not canonical",
        )
    input_layout = contract["input_layout"]
    if not isinstance(input_layout, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_layout_invalid",
            "staged text input layout is invalid",
        )
    expected_layout = _build_staged_input_layout(
        feature,
        prompt_tokens=input_layout.get("prompt_tokens"),
        sequence_length=input_layout.get("sequence_budget"),
    )
    if dict(input_layout) != expected_layout or any(
        segment["activation_bytes"] < expected_layout["minimum_activation_bytes"]
        for segment in plan
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_layout_invalid",
            "staged text input layout or activation budget does not match",
        )
    try:
        handoff = validate_mm1_handoff_contract(contract["visual_handoff"], safe_manifest)
    except Qwen3MultimodalContractError as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_handoff_mismatch",
            "staged visual handoff is invalid",
        ) from exc
    first = plan[0]
    if (
        handoff["text_chain_id"] != contract["text_chain_id"]
        or handoff["generation"] != contract["generation"]
        or handoff["phase"] != contract["phase"]
        or handoff["target_node_id"] != first["node_id"]
        or handoff["tensor"] != feature["tensor"]
        or handoff["artifact"]["sha256"] != feature["media_reference_sha256"]
        or handoff["artifact"]["size_bytes"] != feature["size_bytes"]
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_handoff_mismatch",
            "staged visual handoff does not match the first text segment",
        )
    if contract["execution"] != {
        "mode": "staged_segment",
        "state": "planned",
        "text_weights_loaded": False,
        "segment_materialized": False,
        "full_model_materialized": False,
    } or contract["cleanup"] != {"required": True, "completed": False}:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged execution lifecycle is invalid",
        )
    unsigned = dict(contract)
    digest = str(unsigned.pop("contract_sha256", ""))
    if _SHA256.fullmatch(digest) is None or digest != _staged_digest(unsigned):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_contract_invalid",
            "staged text contract digest does not match",
        )
    _reject_staged_sensitive(contract)
    return contract


class Qwen3MultimodalStagedTextFixture:
    """Hardware-free first-segment lifecycle fixture for MM1.20 contracts."""

    def __init__(self, *, fail_execution: bool = False, fail_cleanup: bool = False) -> None:
        if not isinstance(fail_execution, bool) or not isinstance(fail_cleanup, bool):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_fixture_invalid",
                "staged text fixture flags must be boolean",
            )
        self.fail_execution = fail_execution
        self.fail_cleanup = fail_cleanup
        self.fixture_segment_materialized = False
        self.cleanup_reasons: list[str] = []

    def __call__(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        if self.fixture_segment_materialized:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_fixture_busy",
                "staged text fixture already owns a segment",
            )
        self.fixture_segment_materialized = True
        if self.fail_execution:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_execution_failed",
                "staged text fixture execution failed",
            )
        first = contract["segment_plan"][0]
        next_segment = contract["segment_plan"][1]
        layout = contract["input_layout"]
        return {
            "status": "executed",
            "full_model_materialized": False,
            "execution": {
                "segment_index": 0,
                "layer_range": list(first["layer_range"]),
                "input_handoff_sha256": contract["visual_handoff"]["contract_sha256"],
                "evidence_kind": "cpu_fixture",
                "planned_segment_peak_bytes": int(first["peak_bytes"]),
                "text_weights_loaded": False,
                "segment_materialized": False,
                "fixture_segment_materialized": True,
                "full_model_materialized": False,
            },
            "hidden_handoff": {
                "from_segment": 0,
                "to_segment": 1,
                "shape": [layout["batch_size"], layout["total_sequence"], layout["hidden_size"]],
                "dtype": next_segment["dtype"],
                "device": next_segment["device"],
            },
        }

    def cleanup(self, _contract: Mapping[str, Any], reason_code: str) -> dict[str, Any]:
        self.cleanup_reasons.append(str(reason_code))
        if not self.fail_cleanup:
            self.fixture_segment_materialized = False
        return {
            "completed": not self.fail_cleanup,
            "text_weights_loaded": False,
            "segment_materialized": False,
            "fixture_segment_materialized": self.fixture_segment_materialized,
            "full_model_materialized": False,
        }


def _validate_staged_cleanup(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_cleanup_failed",
            "staged text cleanup returned no metadata",
        )
    _reject_staged_sensitive(value)
    expected = {
        "completed": True,
        "text_weights_loaded": False,
        "segment_materialized": False,
        "fixture_segment_materialized": False,
        "full_model_materialized": False,
    }
    if any(value.get(key) is not expected_value for key, expected_value in expected.items()):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_cleanup_failed",
            "staged text cleanup is incomplete",
        )
    return dict(expected)


def execute_mm1_staged_text_contract(
    contract: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    executor: Any,
) -> dict[str, Any]:
    """Exercise the first text-segment boundary through a bounded CPU fixture."""
    safe = validate_mm1_staged_text_contract(contract, manifest=manifest)
    callback = executor if callable(executor) else getattr(executor, "execute", None)
    cleanup = getattr(executor, "cleanup", None)
    if not callable(callback) or not callable(cleanup):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_executor_invalid",
            "staged text executor must provide execute and cleanup",
        )
    primary_error: Qwen3MultimodalRuntimeError | None = None
    report: Mapping[str, Any] | None = None
    first = safe["segment_plan"][0]
    next_segment = safe["segment_plan"][1]
    execution: Mapping[str, Any] | None = None
    hidden: Mapping[str, Any] | None = None
    try:
        report = callback(safe)
        if not isinstance(report, Mapping):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_execution_failed",
                "staged text executor returned no metadata",
            )
        _reject_staged_sensitive(report)
        execution_value = report.get("execution")
        hidden_value = report.get("hidden_handoff")
        if report.get("status") != "executed" or report.get("full_model_materialized") is not False:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_execution_failed",
                "staged text execution status is invalid",
            )
        if not isinstance(execution_value, Mapping) or (
            execution_value.get("segment_index") != 0
            or execution_value.get("layer_range") != first["layer_range"]
            or execution_value.get("input_handoff_sha256") != safe["visual_handoff"]["contract_sha256"]
            or execution_value.get("planned_segment_peak_bytes") != first["peak_bytes"]
            or execution_value.get("evidence_kind") != "cpu_fixture"
            or execution_value.get("text_weights_loaded") is not False
            or execution_value.get("segment_materialized") is not False
            or execution_value.get("fixture_segment_materialized") is not True
            or execution_value.get("full_model_materialized") is not False
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_execution_failed",
                "staged text execution metadata does not match the contract",
            )
        layout = safe["input_layout"]
        expected_shape = [layout["batch_size"], layout["total_sequence"], layout["hidden_size"]]
        if not isinstance(hidden_value, Mapping) or (
            hidden_value.get("from_segment") != 0
            or hidden_value.get("to_segment") != 1
            or hidden_value.get("shape") != expected_shape
            or hidden_value.get("dtype") != next_segment["dtype"]
            or hidden_value.get("device") != next_segment["device"]
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_staged_handoff_mismatch",
                "staged text output handoff does not match the next segment",
            )
        execution = execution_value
        hidden = hidden_value
    except Qwen3MultimodalRuntimeError as exc:
        primary_error = exc
    except Exception as exc:
        primary_error = Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_execution_failed",
            "staged text executor failed",
        )
        primary_error.__cause__ = exc

    try:
        cleanup_report = _validate_staged_cleanup(
            cleanup(safe, "execution_failed" if primary_error else "completed"),
        )
    except Qwen3MultimodalRuntimeError:
        raise
    except Exception as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_staged_cleanup_failed",
            "staged text cleanup failed",
        ) from exc
    if primary_error is not None:
        raise primary_error
    assert execution is not None and hidden is not None
    return {
        "schema_version": 1,
        "status": "staged_text_segment_fixture_executed",
        "execution_kind": "qwen3_mm1_staged_text_fixture",
        "contract_sha256": safe["contract_sha256"],
        "execution": {
            "segment_index": 0,
            "layer_range": list(first["layer_range"]),
            "evidence_kind": "cpu_fixture",
            "planned_segment_peak_bytes": int(first["peak_bytes"]),
            "text_weights_loaded": False,
            "segment_materialized": False,
            "fixture_segment_materialized": True,
            "full_model_materialized": False,
        },
        "next_segment_request": {
            "segment_index": 1,
            "node_id": next_segment["node_id"],
            "layer_range": list(next_segment["layer_range"]),
            "text_chain_id": safe["text_chain_id"],
            "generation": safe["generation"],
            "phase": safe["phase"],
            "hidden_handoff": {
                "from_segment": 0,
                "to_segment": 1,
                "shape": list(hidden["shape"]),
                "dtype": hidden["dtype"],
                "device": hidden["device"],
            },
        },
        "cleanup": cleanup_report,
        "full_model_materialized": False,
    }


def build_mm1_first_segment_artifact_binding(
    staged_contract: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifact_id: str,
    size_bytes: int,
    sha256: str,
) -> dict[str, Any]:
    """Bind a controller-owned local tensor artifact without exposing its path."""
    safe = validate_mm1_staged_text_contract(staged_contract, manifest=manifest)
    artifact_size = _staged_int(size_bytes, "artifact.size_bytes")
    artifact_sha256 = str(sha256 or "")
    if _SHA256.fullmatch(artifact_sha256) is None:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_artifact_invalid",
            "first-segment input artifact digest is invalid",
        )
    layout = safe["input_layout"]
    first = safe["segment_plan"][0]
    next_segment = safe["segment_plan"][1]
    values = {
        "schema_version": 1,
        "contract_kind": "qwen3_mm1_first_segment_artifact",
        "staged_contract_sha256": safe["contract_sha256"],
        "model_id": safe["model_id"],
        "manifest_sha256": safe["manifest_sha256"],
        "text_chain_id": safe["text_chain_id"],
        "generation": safe["generation"],
        "phase": safe["phase"],
        "segment_index": 0,
        "node_id": first["node_id"],
        "layer_range": list(first["layer_range"]),
        "input_artifact": {
            "artifact_id": _staged_id(artifact_id, "artifact_id"),
            "size_bytes": artifact_size,
            "sha256": artifact_sha256,
            "status": "committed",
            "serialization": "torch_pt",
            "content_kind": "combined_hidden_states",
        },
        "tensor": {
            "shape": [
                layout["batch_size"], layout["total_sequence"], layout["hidden_size"],
            ],
            "dtype": first["dtype"],
            "storage_device": "cpu",
        },
        "next_segment": {
            "segment_index": 1,
            "node_id": next_segment["node_id"],
            "layer_range": list(next_segment["layer_range"]),
            "dtype": next_segment["dtype"],
            "device": next_segment["device"],
        },
        "full_model_materialized": False,
    }
    values["contract_sha256"] = _staged_digest(values)
    return validate_mm1_first_segment_artifact_binding(
        values, staged_contract=safe, manifest=manifest,
    )


def validate_mm1_first_segment_artifact_binding(
    value: Mapping[str, Any],
    *,
    staged_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the path-free local artifact binding for MM1 segment zero."""
    safe = validate_mm1_staged_text_contract(staged_contract, manifest=manifest)
    required = {
        "schema_version", "contract_kind", "staged_contract_sha256", "model_id",
        "manifest_sha256", "text_chain_id", "generation", "phase", "segment_index",
        "node_id", "layer_range", "input_artifact", "tensor", "next_segment",
        "full_model_materialized", "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_binding_invalid",
            "first-segment artifact binding fields are invalid",
        )
    try:
        binding = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_binding_invalid",
            "first-segment artifact binding is not JSON serializable",
        ) from exc
    first = safe["segment_plan"][0]
    next_segment = safe["segment_plan"][1]
    layout = safe["input_layout"]
    if (
        binding["schema_version"] != 1
        or binding["contract_kind"] != "qwen3_mm1_first_segment_artifact"
        or binding["staged_contract_sha256"] != safe["contract_sha256"]
        or binding["model_id"] != safe["model_id"]
        or binding["manifest_sha256"] != safe["manifest_sha256"]
        or binding["text_chain_id"] != safe["text_chain_id"]
        or binding["generation"] != safe["generation"]
        or binding["phase"] != "prefill"
        or binding["phase"] != safe["phase"]
        or binding["segment_index"] != 0
        or binding["node_id"] != first["node_id"]
        or binding["layer_range"] != first["layer_range"]
        or binding["full_model_materialized"] is not False
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_binding_invalid",
            "first-segment artifact binding identity does not match the staged contract",
        )
    artifact = binding["input_artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "artifact_id", "size_bytes", "sha256", "status", "serialization", "content_kind",
    }:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_artifact_invalid",
            "first-segment input artifact metadata is invalid",
        )
    _staged_id(artifact.get("artifact_id"), "artifact_id")
    _staged_int(artifact.get("size_bytes"), "artifact.size_bytes")
    if (
        _SHA256.fullmatch(str(artifact.get("sha256") or "")) is None
        or artifact.get("status") != "committed"
        or artifact.get("serialization") != "torch_pt"
        or artifact.get("content_kind") != "combined_hidden_states"
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_artifact_invalid",
            "first-segment input artifact evidence is invalid",
        )
    expected_tensor = {
        "shape": [layout["batch_size"], layout["total_sequence"], layout["hidden_size"]],
        "dtype": first["dtype"],
        "storage_device": "cpu",
    }
    expected_next = {
        "segment_index": 1,
        "node_id": next_segment["node_id"],
        "layer_range": list(next_segment["layer_range"]),
        "dtype": next_segment["dtype"],
        "device": next_segment["device"],
    }
    if binding["tensor"] != expected_tensor or binding["next_segment"] != expected_next:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_binding_invalid",
            "first-segment tensor or next-segment binding does not match",
        )
    unsigned = dict(binding)
    digest = str(unsigned.pop("contract_sha256", ""))
    if _SHA256.fullmatch(digest) is None or digest != _staged_digest(unsigned):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sidecar_binding_invalid",
            "first-segment artifact binding digest does not match",
        )
    _reject_staged_sensitive(binding)
    return binding


def _mm1_local_file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class Qwen3MultimodalFirstSegmentSidecarAdapter:
    """Drive one MM1 first text segment through a node-local Qwen3 sidecar."""

    def __init__(self, session: Any, *, artifact_root: str | Path) -> None:
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_artifact_root_missing",
                "first-segment sidecar artifact root is unavailable",
            )
        self.session = session
        self.artifact_root = root
        self.lifecycle: list[str] = []
        self._outputs: dict[str, Path] = {}

    def _local_ref(self, value: str | Path, *, must_exist: bool) -> Path:
        candidate = Path(value).expanduser().absolute().resolve(strict=False)
        try:
            candidate.relative_to(self.artifact_root)
        except ValueError as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_artifact_scope",
                "first-segment artifact escapes the local data-plane root",
            ) from exc
        if must_exist and not candidate.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_artifact_missing",
                "first-segment input artifact is unavailable",
            )
        return candidate

    def _validate_session_identity(self, staged: Mapping[str, Any]) -> None:
        identity = getattr(self.session, "identity", None)
        if not isinstance(identity, Mapping) or getattr(self.session, "phase", None) != "idle":
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_session_invalid",
                "first-segment sidecar session is unavailable or already used",
            )
        first = staged["segment_plan"][0]
        dtype = str(identity.get("dtype") or "").lower().removeprefix("torch.")
        device = str(identity.get("execution_device") or "").lower()
        expected = {
            "model_id": staged["model_id"],
            "node_id": first["node_id"],
            "layer_range": first["layer_range"],
            "total_layers": staged["total_layers"],
            "has_embedding": first["has_embedding"],
            "has_lm_head": first["has_lm_head"],
            "generation": staged["generation"],
            "assignment_manifest_sha256": first["assignment_manifest_sha256"],
        }
        if any(identity.get(key) != expected_value for key, expected_value in expected.items()):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_session_mismatch",
                "first-segment sidecar assignment does not match the staged contract",
            )
        if dtype != first["dtype"] or device != first["device"]:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_session_mismatch",
                "first-segment sidecar dtype or device does not match",
            )

    @staticmethod
    def _normalised_dtype(value: Any) -> str:
        return str(value or "").lower().removeprefix("torch.")

    def execute(
        self,
        staged_contract: Mapping[str, Any],
        *,
        manifest: Mapping[str, Any],
        artifact_binding: Mapping[str, Any],
        input_ref: str | Path,
        cancel_after_commit: bool = False,
    ) -> dict[str, Any]:
        """Execute prefill and retain its output only in the local data plane."""
        if not isinstance(cancel_after_commit, bool):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_contract_invalid", "cancel flag must be boolean",
            )
        staged = validate_mm1_staged_text_contract(staged_contract, manifest=manifest)
        binding = validate_mm1_first_segment_artifact_binding(
            artifact_binding, staged_contract=staged, manifest=manifest,
        )
        self._validate_session_identity(staged)
        input_path = self._local_ref(input_ref, must_exist=True)
        artifact = binding["input_artifact"]
        if _mm1_local_file_evidence(input_path) != (
            artifact["size_bytes"], artifact["sha256"],
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_artifact_mismatch",
                "first-segment input artifact evidence does not match the binding",
            )
        output_path = self.artifact_root / f"mm1-first-{secrets.token_hex(16)}.pt"
        first = staged["segment_plan"][0]
        layout = staged["input_layout"]
        try:
            self.session.prepare()
            self.lifecycle.append("prepare")
            self.session.commit()
            self.lifecycle.append("commit")
            if cancel_after_commit:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sidecar_cancelled",
                    "first-segment sidecar execution was cancelled after commit",
                )
            report = self.session.execute(
                phase="prefill",
                artifact_root=self.artifact_root,
                input_ref=input_path,
                output_ref=output_path,
                chain_id=staged["text_chain_id"],
                segment_index=0,
                sequence_length=layout["total_sequence"],
                batch_size=layout["batch_size"],
                has_next_segment=True,
                generation=staged["generation"],
                dtype=first["dtype"],
                device=first["device"],
            )
            self.lifecycle.append("prefill")
            _reject_staged_sensitive(report)
            execution = report.get("execution")
            hidden = report.get("hidden_handoff")
            expected_shape = [
                layout["batch_size"], layout["total_sequence"], layout["hidden_size"],
            ]
            if not isinstance(execution, Mapping) or (
                execution.get("data_plane") != "local_artifact"
                or execution.get("segment_materialized") is not True
                or execution.get("full_model_materialized") is not False
                or isinstance(execution.get("artifact_bytes"), bool)
                or not isinstance(execution.get("artifact_bytes"), int)
                or execution.get("artifact_bytes", 0) <= 0
                or _SHA256.fullmatch(str(execution.get("artifact_sha256") or "")) is None
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sidecar_execution_failed",
                    "first-segment sidecar execution evidence is invalid",
                )
            if not isinstance(hidden, Mapping) or (
                hidden.get("chain_id") != staged["text_chain_id"]
                or hidden.get("from_segment") != 0
                or hidden.get("to_segment") != 1
                or hidden.get("shape") != expected_shape
                or hidden.get("batch_size") != layout["batch_size"]
                or hidden.get("sequence_length") != layout["total_sequence"]
                or hidden.get("hidden_size") != layout["hidden_size"]
                or self._normalised_dtype(hidden.get("dtype")) != first["dtype"]
                or str(hidden.get("device") or "").lower() != first["device"]
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sidecar_handoff_mismatch",
                    "first-segment sidecar hidden handoff does not match the next segment",
                )
            output_evidence = _mm1_local_file_evidence(output_path)
            if output_evidence != (
                execution["artifact_bytes"], execution["artifact_sha256"],
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sidecar_artifact_mismatch",
                    "first-segment output artifact evidence does not match",
                )
            release_report = self.session.release()
            self.lifecycle.append("release")
            if not isinstance(release_report, Mapping) or (
                release_report.get("cleanup_complete") is not True
                or release_report.get("segment_materialized") is not False
                or release_report.get("full_model_materialized") is not False
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sidecar_cleanup_failed",
                    "first-segment sidecar release did not prove cleanup",
                )
            output_id = f"mm1out_{output_evidence[1][:32]}"
            self._outputs[output_id] = output_path
            result = {
                "schema_version": 1,
                "status": "first_segment_sidecar_executed",
                "contract_sha256": staged["contract_sha256"],
                "artifact_binding_sha256": binding["contract_sha256"],
                "lifecycle": list(self.lifecycle),
                "input_artifact": dict(artifact),
                "output_artifact": {
                    "artifact_id": output_id,
                    "size_bytes": output_evidence[0],
                    "sha256": output_evidence[1],
                    "status": "committed",
                },
                "hidden_handoff": dict(hidden),
                "next_segment": {
                    **dict(binding["next_segment"]),
                    "hidden_handoff": dict(hidden),
                },
                "sidecar_cleanup_complete": True,
                "artifact_cleanup_required": True,
                "segment_materialized": False,
                "full_model_materialized": False,
            }
            _reject_staged_sensitive(result)
            return result
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            if getattr(self.session, "phase", None) in {"prepared", "committed"}:
                try:
                    abort_report = self.session.abort()
                    self.lifecycle.append("abort")
                    if not isinstance(abort_report, Mapping) or (
                        abort_report.get("cleanup_complete") is not True
                        or abort_report.get("segment_materialized") is not False
                        or abort_report.get("full_model_materialized") is not False
                    ):
                        raise Qwen3MultimodalRuntimeError(
                            "qwen3_mm1_sidecar_cleanup_failed",
                            "first-segment sidecar abort did not prove cleanup",
                        )
                except Exception as cleanup_exc:
                    if isinstance(cleanup_exc, Qwen3MultimodalRuntimeError):
                        raise
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_sidecar_cleanup_failed",
                        "first-segment sidecar abort did not complete",
                    ) from cleanup_exc
            if isinstance(exc, Qwen3MultimodalRuntimeError):
                raise
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_execution_failed",
                "first-segment sidecar execution failed",
            ) from exc

    def output_path(self, artifact_id: str) -> Path:
        """Resolve a retained output for an in-process downstream data-plane step."""
        path = self._outputs.get(str(artifact_id))
        if path is None or not path.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_sidecar_artifact_missing",
                "first-segment output artifact is unavailable",
            )
        return path

    def cleanup(self, reason_code: str = "completed") -> dict[str, Any]:
        """Release retained local outputs without returning their paths."""
        reason = _staged_id(reason_code, "reason_code")
        removed = 0
        for artifact_id, path in list(self._outputs.items()):
            if path.exists():
                path.unlink()
                removed += 1
            self._outputs.pop(artifact_id, None)
        if getattr(self.session, "phase", None) in {"prepared", "committed"}:
            try:
                abort_report = self.session.abort()
                self.lifecycle.append("abort")
                if not isinstance(abort_report, Mapping) or (
                    abort_report.get("cleanup_complete") is not True
                    or abort_report.get("segment_materialized") is not False
                    or abort_report.get("full_model_materialized") is not False
                ):
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_sidecar_cleanup_failed",
                        "first-segment sidecar cleanup did not prove release",
                    )
            except Exception as exc:
                if isinstance(exc, Qwen3MultimodalRuntimeError):
                    raise
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sidecar_cleanup_failed",
                    "first-segment sidecar cleanup did not complete",
                ) from exc
        return {
            "completed": True,
            "reason_code": reason,
            "removed_artifacts": removed,
            "retained_artifacts": 0,
            "segment_materialized": False,
            "full_model_materialized": False,
        }


__all__ = [
    "Qwen3MultimodalRuntimeError",
    "Qwen3MultimodalSidecarAdapter",
    "Qwen3MultimodalSyntheticExecutor",
    "run_mm1_visual_tower_skeleton",
    "run_mm1_visual_placeholder_execution",
    "bind_mm1_visual_feature_handoff",
    "run_mm1_synthetic_visual_chain",
    "run_mm1_synthetic_text_decode",
    "run_mm1_synthetic_hybrid_chain",
    "build_mm1_staged_text_contract",
    "validate_mm1_staged_text_contract",
    "execute_mm1_staged_text_contract",
    "Qwen3MultimodalStagedTextFixture",
    "build_mm1_first_segment_artifact_binding",
    "validate_mm1_first_segment_artifact_binding",
    "Qwen3MultimodalFirstSegmentSidecarAdapter",
]
