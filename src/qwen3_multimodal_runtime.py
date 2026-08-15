"""Hardware-free MM1 target executor for the Qwen3 network consume boundary.

This executor deliberately does not load a vision model.  It exercises the
same target-local callback used by an isolated sidecar, constructs and checks
the MM1 visual handoff after the receiver has committed the input, and emits a
small deterministic artifact for downstream lifecycle tests.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from qwen3_multimodal_contract import validate_mm1_model_manifest  # noqa: E402
from qwen3_multimodal_preflight import validate_mm1_media_tensor_reference  # noqa: E402
from qwen3_multimodal_contract import (
    MM1_MAX_VISUAL_TOKENS,
    Qwen3MultimodalContractError,
    build_mm1_handoff_contract,
    build_mm1_transfer_binding,
    validate_mm1_model_manifest,
)


_TRANSFER_ID = re.compile(r"^qtx_[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    if bool(placement.get("request_has_media")) == bool(text_only):
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
    active = bool(placement.get("vision_tower_active"))
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
    reference = validate_mm1_media_tensor_reference(
        media_tensor_reference,
        model_id=str(media_tensor_reference["model_id"]),
        component_ids=list(media_tensor_reference["component_ids"]),
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
    reference = validate_mm1_media_tensor_reference(
        media_tensor_reference,
        model_id=str(media_tensor_reference["model_id"]),
        component_ids=list(media_tensor_reference["component_ids"]),
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


__all__ = [
    "Qwen3MultimodalRuntimeError",
    "Qwen3MultimodalSidecarAdapter",
    "Qwen3MultimodalSyntheticExecutor",
    "run_mm1_visual_tower_skeleton",
    "run_mm1_visual_placeholder_execution",
    "bind_mm1_visual_feature_handoff",
]
