"""Persistent offline worker for one Gemma 4 Unified text assignment."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gemma4_pipeline_adapter import (  # noqa: E402
    Gemma4PipelineAdapter,
    load_gemma4_text_layer_assignment,
    validate_gemma4_assignment,
)


SCHEMA_VERSION = 1
OPERATION = "gemma4_pipeline_sidecar"
TRANSFORMERS_VERSION = "5.10.1"
MAX_FRAME_BYTES = 256 * 1024
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _response(phase: str, *, status: str, gate_passed: bool, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": OPERATION,
        "phase": phase,
        "status": status,
        "gate_passed": bool(gate_passed),
        "read_only": True,
        "network_access": "disabled",
        "full_model_materialized": False,
        "multimodal_materialized": False,
        "segment_materialized": False,
        "cleanup_complete": False,
        "errors": [],
        **extra,
    }


def _error(phase: str, code: str, message: str, *, status: str = "rejected") -> dict[str, Any]:
    result = _response(phase, status=status, gate_passed=False)
    result["errors"] = [{"code": str(code), "message": str(message)[:2048]}]
    return result


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid local metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"local metadata is not an object: {path.name}")
    return value


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class _RuntimeSession:
    def __init__(self) -> None:
        self.phase = "idle"
        self.root: Path | None = None
        self.request: dict[str, Any] | None = None
        self.adapter: Gemma4PipelineAdapter | None = None
        self.assignment: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.runtime: dict[str, Any] = {}

    def _same_identity(self, request: dict[str, Any], *, include_generation: bool = True) -> bool:
        if self.request is None:
            return False
        keys = (
            "model_id", "model_sha256", "config_id", "plan_id", "node_id",
            "layer_range", "total_layers", "has_embedding", "has_lm_head",
            "required_shared_kv_types", "produced_shared_kv_types",
            "assignment_manifest_sha256", "execution_device", "dtype",
        )
        if include_generation:
            keys += ("generation",)
        return all(self.request.get(key) == request.get(key) for key in keys)

    def _cleanup(self) -> None:
        self.adapter = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _manifest_gate(request: dict[str, Any], root: Path) -> None:
        expected = str(request.get("assignment_manifest_sha256", "") or "")
        if not expected:
            return
        from pipeline_assignment_manifest import build_assignment_manifest

        layer_range = request["layer_range"]
        manifest = build_assignment_manifest(
            root,
            model_id=str(request["model_id"]),
            model_sha256=str(request["model_sha256"]),
            config_id=str(request["config_id"]),
            plan_id=str(request["plan_id"]),
            node_id=str(request["node_id"]),
            start_layer=int(layer_range[0]),
            end_layer=int(layer_range[1]),
            total_layers=int(request["total_layers"]),
            has_embedding=bool(request["has_embedding"]),
            has_lm_head=bool(request["has_lm_head"]),
        )
        if manifest.get("manifest_sha256") != expected:
            raise ValueError("local assignment manifest differs from signed contract")

    @staticmethod
    def _assignment_bytes(root: Path, weight_map: dict[str, str], dtype: str) -> tuple[int, int]:
        from safetensors import safe_open

        target_bytes_per_element = {
            "float16": 2, "bfloat16": 2, "float32": 4,
        }.get(str(dtype))
        if target_bytes_per_element is None:
            raise ValueError("unsupported assignment dtype")
        source_bytes = 0
        target_bytes = 0
        by_file: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            by_file.setdefault(str(filename), []).append(str(key))
        for filename, keys in by_file.items():
            relative = Path(filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe assignment shard path")
            shard = (root / relative).resolve(strict=False)
            try:
                shard.relative_to(root)
            except ValueError as exc:
                raise ValueError("assignment shard escapes model root") from exc
            if not shard.is_file():
                raise ValueError("assignment shard is unavailable")
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                if not set(keys).issubset(available):
                    raise ValueError("assignment shard is missing declared keys")
                for key in keys:
                    view = handle.get_slice(key)
                    shape = tuple(int(value) for value in view.get_shape())
                    elements = 1
                    for value in shape:
                        elements *= value
                    item_size = _DTYPE_BYTES.get(str(view.get_dtype()))
                    if item_size is None:
                        raise ValueError("unsupported Safetensors dtype")
                    source_bytes += elements * item_size
                    target_bytes += elements * target_bytes_per_element
        return source_bytes, target_bytes

    @staticmethod
    def _available_bytes(device: str) -> int:
        if str(device).startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                return 0
            free_bytes, _ = torch.cuda.mem_get_info()
            return max(0, int(free_bytes))
        import psutil

        return max(0, int(psutil.virtual_memory().available))

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.phase != "idle":
            return _error("prepare", "gemma4_sidecar_phase_invalid", "sidecar is not idle")
        try:
            import accelerate
            import safetensors
            import torch
            import transformers
            from transformers import (
                Gemma4UnifiedConfig,
                Gemma4UnifiedForConditionalGeneration,
            )
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            del Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
            del create_causal_mask, create_sliding_window_causal_mask
            if transformers.__version__ != TRANSFORMERS_VERSION:
                raise ValueError(
                    f"Gemma 4 sidecar requires transformers=={TRANSFORMERS_VERSION}"
                )
            controller_python = Path(str(request.get("controller_python", ""))).resolve(strict=False)
            if controller_python == Path(sys.executable).resolve(strict=False):
                raise ValueError("Gemma 4 sidecar is not isolated from the controller")
            root = Path(str(request.get("model_path", ""))).expanduser().absolute().resolve(strict=False)
            if not root.is_dir():
                raise ValueError("model assignment directory is unavailable")
            layer_range = request.get("layer_range")
            if not isinstance(layer_range, list) or len(layer_range) != 2:
                raise ValueError("layer_range is invalid")
            config = _json_object(root / "config.json")
            index = _json_object(root / "model.safetensors.index.json")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("filtered Safetensors index has no weight_map")
            text_config = config.get("text_config")
            if not isinstance(text_config, dict):
                raise ValueError("Gemma 4 text_config is unavailable")
            validation = validate_gemma4_assignment(
                model_type=str(config.get("model_type", "")),
                total_layers=int(text_config.get("num_hidden_layers", 0) or 0),
                start_layer=int(layer_range[0]),
                end_layer=int(layer_range[1]),
                has_embedding=bool(request.get("has_embedding", False)),
                has_lm_head=bool(request.get("has_lm_head", False)),
                keys=weight_map.keys(),
                tie_word_embeddings=bool(text_config.get("tie_word_embeddings", False)),
            )
            if set(validation["selected_keys"]) != set(weight_map):
                raise ValueError("Safetensors index contains unassigned keys")
            self._manifest_gate(request, root)
            source_bytes, target_bytes = self._assignment_bytes(
                root, {str(key): str(value) for key, value in weight_map.items()},
                str(request.get("dtype", "float32")),
            )
            margin = float(request.get("safety_margin", 1.2))
            reserve = int(request.get("reserve_bytes", 0))
            required_bytes = int(max(source_bytes, target_bytes) * margin) + reserve
            device = str(request.get("execution_device", "cpu"))
            available_bytes = self._available_bytes(device)
            self.assignment = {
                "layer_range": list(layer_range),
                "selected_tensor_count": len(weight_map),
                "source_tensor_bytes": source_bytes,
                "target_tensor_bytes": target_bytes,
                "full_model_materialized": False,
                "multimodal_materialized": False,
            }
            self.resources = {
                "device": device,
                "available_bytes": available_bytes,
                "required_bytes": required_bytes,
                "safety_margin": margin,
                "reserve_bytes": reserve,
            }
            self.runtime = {
                "python_isolated": True,
                "transformers_version": transformers.__version__,
                "torch_version": str(torch.__version__),
                "accelerate_version": str(accelerate.__version__),
                "safetensors_version": str(safetensors.__version__),
                "offline": True,
                "trust_remote_code": False,
            }
            if available_bytes < required_bytes:
                result = _error(
                    "prepare", "gemma4_sidecar_resource_rejected",
                    "assignment exceeds available node-local memory",
                    status="resource_rejected",
                )
                result.update({
                    "assignment": self.assignment,
                    "resources": self.resources,
                    "runtime": self.runtime,
                })
                return result
            self.root = root
            self.request = dict(request)
            self.phase = "prepared"
            return _response(
                "prepare", status="prepared", gate_passed=True,
                assignment=self.assignment,
                resources=self.resources,
                runtime=self.runtime,
            )
        except Exception as exc:
            self._cleanup()
            return _error(
                "prepare", "gemma4_sidecar_prepare_failed", str(exc),
                status="artifact_rejected",
            )

    def commit(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.phase != "prepared" or not self._same_identity(request) or self.root is None:
            return _error(
                "commit", "gemma4_sidecar_not_prepared",
                "commit does not match a prepared assignment",
            )
        try:
            layer_range = request["layer_range"]
            model, metrics = load_gemma4_text_layer_assignment(
                self.root,
                start_layer=int(layer_range[0]),
                end_layer=int(layer_range[1]),
                has_embedding=bool(request.get("has_embedding", False)),
                has_lm_head=bool(request.get("has_lm_head", False)),
                device=str(self.resources.get("device", "cpu")),
                dtype=str(request.get("dtype", "float32")),
            )
            self.adapter = Gemma4PipelineAdapter(
                model,
                start_layer=int(layer_range[0]),
                end_layer=int(layer_range[1]),
                has_embedding=bool(request.get("has_embedding", False)),
                has_lm_head=bool(request.get("has_lm_head", False)),
                total_layers=int(request.get("total_layers", 0)),
            )
            execution = {
                key: value for key, value in metrics.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            execution.update({
                "full_model_materialized": False,
                "multimodal_materialized": False,
                "segment_materialized": True,
            })
            self.phase = "committed"
            return _response(
                "commit", status="committed", gate_passed=True,
                assignment=self.assignment,
                resources=self.resources,
                runtime=self.runtime,
                execution=execution,
                segment_materialized=True,
            )
        except Exception as exc:
            self._cleanup()
            self.phase = "idle"
            return _error(
                "commit", "gemma4_sidecar_commit_failed", str(exc),
                status="execution_failed",
            )

    def release(self, request: dict[str, Any], *, aborted: bool = False) -> dict[str, Any]:
        phase = "abort" if aborted else "release"
        if self.phase not in {"prepared", "committed"}:
            if aborted and self.phase == "idle":
                return _response(phase, status="aborted", gate_passed=True, cleanup_complete=True)
            return _error(
                phase, "gemma4_sidecar_release_invalid", "release does not match active state",
            )
        self._cleanup()
        self.phase = "idle"
        return _response(
            phase,
            status="aborted" if aborted else "released",
            gate_passed=True,
            cleanup_complete=True,
            assignment=self.assignment,
        )

    @staticmethod
    def _artifact_path(request: dict[str, Any], value: Any, *, must_exist: bool) -> Path:
        root = Path(str(request.get("artifact_root", ""))).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise ValueError("local artifact root is unavailable")
        path = Path(str(value or "")).expanduser().absolute().resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("local artifact escapes artifact root") from exc
        if must_exist and not path.is_file():
            raise ValueError("local artifact is unavailable")
        if not must_exist:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _load_artifact(path: Path) -> dict[str, Any]:
        import torch

        value = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(value, dict):
            raise ValueError("local tensor artifact must contain an object")
        return value

    @staticmethod
    def _save_artifact(path: Path, payload: dict[str, Any]) -> tuple[int, str]:
        import torch

        fd, temporary = tempfile.mkstemp(
            prefix=".gemma4-artifact-", suffix=".pt", dir=str(path.parent),
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            torch.save(payload, str(temporary_path))
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return _file_evidence(path)

    @staticmethod
    def _shared_to_cpu(value: Any) -> dict[str, tuple[Any, Any]]:
        if not isinstance(value, dict):
            value = getattr(value, "data", value)
        if not isinstance(value, dict):
            raise ValueError("shared-KV output is not a mapping")
        return {
            str(layer_type): (
                pair[0].detach().to("cpu"),
                pair[1].detach().to("cpu"),
            )
            for layer_type, pair in value.items()
        }

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        phase = str(request.get("phase", ""))
        if (
            self.phase != "committed"
            or self.adapter is None
            or not self._same_identity(request, include_generation=False)
        ):
            return _error(
                phase, "gemma4_sidecar_not_committed",
                "execution does not match a committed assignment",
            )
        if phase not in {"prefill", "decode"} or request.get("data_plane") != "local_artifact":
            return _error(
                phase, "gemma4_sidecar_data_plane_invalid",
                "execution requires prefill/decode local artifacts",
            )
        try:
            import torch

            input_path = self._artifact_path(request, request.get("input_ref"), must_exist=True)
            output_path = self._artifact_path(request, request.get("output_ref"), must_exist=False)
            if _file_evidence(input_path) != (
                int(request.get("input_bytes", -1)), str(request.get("input_sha256", "")),
            ):
                raise ValueError("input artifact evidence differs")
            payload = self._load_artifact(input_path)
            input_ids = payload.get("input_ids")
            hidden_states = payload.get("hidden_states")
            if (input_ids is None) == (hidden_states is None):
                raise ValueError("artifact requires exactly one input tensor")
            tensor = input_ids if input_ids is not None else hidden_states
            expected_rank = 2 if input_ids is not None else 3
            if not hasattr(tensor, "shape") or len(tuple(tensor.shape)) != expected_rank:
                raise ValueError("input tensor rank is invalid")
            batch_size = int(tensor.shape[0])
            current_length = int(tensor.shape[1])
            if batch_size != int(request.get("batch_size", 0) or 0):
                raise ValueError("input batch differs from contract")
            requested_length = int(request.get("sequence_length", 0) or 0)
            past_length = requested_length - current_length
            if requested_length <= 0 or past_length < 0:
                raise ValueError("sequence length is invalid")
            cache = None
            if phase == "decode":
                kv_path = self._artifact_path(request, request.get("kv_ref"), must_exist=True)
                if _file_evidence(kv_path) != (
                    int(request.get("kv_bytes", -1)), str(request.get("kv_sha256", "")),
                ):
                    raise ValueError("KV artifact evidence differs")
                cache = self._load_artifact(kv_path).get("past_key_values")
                if cache is None:
                    raise ValueError("KV artifact has no native Cache")
            shared = payload.get("shared_kv_states")
            shared_types = set(shared or {})
            required_types = set(request.get("required_shared_kv_types", []))
            if not required_types.issubset(shared_types):
                raise ValueError("required shared-KV handoff is missing")
            cache_position = torch.arange(past_length, requested_length)
            result = self.adapter.forward(
                input_ids=input_ids,
                hidden_states=hidden_states,
                past_key_values=cache,
                shared_kv_states=shared,
                use_cache=True,
                cache_position=cache_position,
            )
            if int(result.get("sequence_length", -1)) != requested_length:
                raise ValueError("Gemma 4 logical sequence length differs from contract")
            output_hidden = result.get("hidden_states")
            output_logits = result.get("logits")
            output_tensor = output_hidden if output_hidden is not None else output_logits
            if output_tensor is None:
                raise ValueError("Gemma 4 segment returned no output tensor")
            output_shared = self._shared_to_cpu(result.get("shared_kv_states"))
            if bool(request.get("has_next_segment")):
                expected_shared = set(request.get("produced_shared_kv_types", [])) | required_types
                if not expected_shared.issubset(output_shared):
                    raise ValueError("Gemma 4 segment did not produce/propagate shared-KV")
            output_payload = {
                "hidden_states": output_hidden.detach().to("cpu") if output_hidden is not None else None,
                "logits": output_logits.detach().to("cpu") if output_logits is not None else None,
                "past_key_values": result.get("past_key_values"),
                "shared_kv_states": output_shared,
            }
            output_bytes, output_sha256 = self._save_artifact(output_path, output_payload)
            generation = int(request.get("generation", -1))
            segment_index = int(request.get("segment_index", -1))
            layer_range = list(request.get("layer_range", []))
            if generation < 0 or segment_index < 0 or len(layer_range) != 2:
                raise ValueError("execution identity is invalid")
            handoff = None
            if bool(request.get("has_next_segment")):
                handoff = {
                    "chain_id": str(request.get("chain_id", "")),
                    "from_segment": segment_index,
                    "to_segment": segment_index + 1,
                    "shape": [int(value) for value in output_hidden.shape],
                    "sequence_length": current_length,
                    "shared_kv_types": sorted(output_shared),
                }
            return _response(
                phase, status="executed", gate_passed=True,
                execution={
                    "data_plane": "local_artifact",
                    "artifact_bytes": output_bytes,
                    "artifact_sha256": output_sha256,
                    "full_model_materialized": False,
                    "multimodal_materialized": False,
                    "segment_materialized": True,
                },
                hidden_handoff=handoff,
                kv_contract={
                    "chain_id": str(request.get("chain_id", "")),
                    "segment_index": segment_index,
                    "layer_range": layer_range,
                    "sequence_length": requested_length,
                    "batch_size": batch_size,
                    "phase": phase,
                    "generation": generation,
                    "ownership": "node_local",
                },
                shared_kv_contract={
                    "types": sorted(output_shared),
                    "sequence_lengths": dict(result.get("shared_kv_sequence_lengths", {})),
                    "sequence_axis": -2,
                },
            )
        except Exception as exc:
            return _error(
                phase, "gemma4_sidecar_execution_failed", str(exc),
                status="execution_failed",
            )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(request, dict)
            or request.get("schema_version") != SCHEMA_VERSION
            or request.get("operation") != OPERATION
            or request.get("read_only") is not True
            or request.get("network_access") != "disabled"
        ):
            phase = str(request.get("phase", "")) if isinstance(request, dict) else ""
            return _error(
                phase, "gemma4_sidecar_protocol_invalid",
                "unsupported or unsafe sidecar protocol",
                status="invalid_request",
            )
        phase = str(request.get("phase", ""))
        if phase == "prepare":
            return self.prepare(request)
        if phase == "commit":
            return self.commit(request)
        if phase in {"prefill", "decode"}:
            return self.execute(request)
        if phase == "release":
            return self.release(request)
        if phase == "abort":
            return self.release(request, aborted=True)
        return _error(
            phase, "gemma4_sidecar_phase_invalid", "unsupported sidecar phase",
            status="invalid_request",
        )


def main() -> int:
    session = _RuntimeSession()
    try:
        for raw in sys.stdin.buffer:
            if len(raw) > MAX_FRAME_BYTES:
                report = _error(
                    "", "gemma4_sidecar_frame_oversize",
                    "sidecar frame exceeds 256 KiB",
                    status="invalid_request",
                )
            else:
                try:
                    request = json.loads(raw.decode("utf-8"))
                    report = session.handle(request)
                except Exception as exc:
                    report = _error(
                        "", "gemma4_sidecar_worker_failed", exc.__class__.__name__,
                        status="worker_failed",
                    )
            print(json.dumps(report, ensure_ascii=True, separators=(",", ":")), flush=True)
    finally:
        session._cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
