"""Persistent isolated worker for one Qwen3 node-local assignment.

The worker accepts one bounded JSON object per line.  It never exposes paths
or tensors in its reports.  Prepare creates a filtered, hard-link-first
assignment; commit loads only that assignment; release/abort drops all local
state and removes the staging directory.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_pipeline_smoke_worker import (  # noqa: E402
    _load_object,
    _prepare_filtered_assignment,
    execute_request,
    select_qwen3_assignment_keys,
)
from qwen3_pipeline_adapter import load_qwen3_layer_assignment, render_without_thinking  # noqa: E402
from qwen3_pipeline_chain import build_hidden_handoff, build_kv_contract  # noqa: E402


SCHEMA_VERSION = 1
OPERATION = "qwen3_pipeline_sidecar"
MAX_FRAME_BYTES = 256 * 1024


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
        "segment_materialized": False,
        "cleanup_complete": False,
        "errors": [],
        **extra,
    }


def _error(phase: str, code: str, message: str, *, status: str = "rejected") -> dict[str, Any]:
    result = _response(phase, status=status, gate_passed=False)
    result["errors"] = [{"code": str(code), "message": str(message)[:2048]}]
    return result


class _RuntimeSession:
    def __init__(self) -> None:
        self.phase = "idle"
        self.root: Path | None = None
        self.stage: Path | None = None
        self.request: dict[str, Any] | None = None
        self.adapter: Any = None
        self.assignment: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.execution: dict[str, Any] = {}

    def _same_identity(self, request: dict[str, Any], *, include_generation: bool = True) -> bool:
        if self.request is None:
            return False
        keys = (
            "model_id", "model_sha256", "config_id", "plan_id", "node_id",
            "layer_range", "total_layers", "has_embedding", "has_lm_head",
            "assignment_manifest_sha256", "execution_device", "dtype",
        )
        if include_generation:
            keys += ("generation",)
        return all(self.request.get(key) == request.get(key) for key in keys)

    def _cleanup(self) -> None:
        self.adapter = None
        if self.stage is not None:
            import shutil

            shutil.rmtree(self.stage, ignore_errors=True)
        self.stage = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _manifest_gate(self, request: dict[str, Any], root: Path) -> None:
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
            raise ValueError("local assignment manifest digest differs from signed contract")

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.phase != "idle":
            return _error("prepare", "qwen3_sidecar_phase_invalid", "sidecar is not idle")
        try:
            root = Path(str(request.get("model_path", ""))).expanduser().absolute().resolve(strict=False)
            if not root.is_dir():
                raise ValueError("model directory is unavailable")
            layer_range = request.get("layer_range")
            if not isinstance(layer_range, list) or len(layer_range) != 2:
                raise ValueError("layer_range is invalid")
            self._manifest_gate(request, root)
            preflight = {
                "schema_version": 1,
                "operation": "qwen3_pipeline_smoke",
                "read_only": True,
                "network_access": "disabled",
                "model_path": str(root),
                "layer_range": layer_range,
                "has_embedding": bool(request.get("has_embedding", False)),
                "has_lm_head": bool(request.get("has_lm_head", False)),
                "execute": False,
                "device": str(request.get("execution_device", "cpu")),
                "dtype": request.get("dtype"),
                "safety_margin": request.get("safety_margin", 1.2),
                "reserve_bytes": request.get("reserve_bytes", 512 * 1024**2),
                "controller_python": request.get("controller_python", ""),
            }
            for key in ("available_ram_bytes", "available_vram_bytes"):
                if key in request:
                    preflight[key] = request[key]
            report = execute_request(preflight)
            self.assignment = dict(report.get("assignment", {}))
            self.resources = dict(report.get("resources", {}))
            if report.get("gate_passed") is not True:
                result = _error("prepare", "qwen3_sidecar_preflight_rejected", "assignment resource/runtime gate rejected")
                result.update({"status": report.get("status", "rejected"), "assignment": self.assignment, "resources": self.resources, "runtime": report.get("runtime", {})})
                return result
            config = _load_object(root / "config.json")
            index = _load_object(root / "model.safetensors.index.json")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Safetensors index has no weight_map")
            selected = select_qwen3_assignment_keys(
                weight_map.keys(),
                start_layer=int(layer_range[0]), end_layer=int(layer_range[1]),
                has_embedding=bool(request.get("has_embedding", False)),
                has_lm_head=bool(request.get("has_lm_head", False)),
                tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            )
            self.stage = _prepare_filtered_assignment(
                root, selected, {str(key): str(value) for key, value in weight_map.items()},
            )
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(root), local_files_only=True, trust_remote_code=False,
                )
                rendered = render_without_thinking(
                    tokenizer, [{"role": "user", "content": "Reply with the word OK."}],
                )
                encoded = tokenizer(rendered, add_special_tokens=False)
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
                if input_ids is None or not hasattr(input_ids, "__len__"):
                    raise ValueError("tokenizer produced no input ids")
                tokenizer_report = {
                    "checked": True,
                    "thinking_disabled": True,
                    "input_token_count": len(input_ids),
                }
            except Exception as exc:
                raise ValueError(f"tokenizer thinking-off preflight failed: {exc}") from exc
            self.root = root
            self.request = dict(request)
            self.phase = "prepared"
            return _response(
                "prepare", status="prepared", gate_passed=True,
                assignment=self.assignment,
                resources=self.resources,
                tokenizer=tokenizer_report,
                staged=True,
            )
        except Exception as exc:
            self._cleanup()
            return _error("prepare", "qwen3_sidecar_prepare_failed", str(exc), status="artifact_rejected")

    def commit(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.phase != "prepared" or not self._same_identity(request) or self.stage is None:
            return _error("commit", "qwen3_sidecar_not_prepared", "commit does not match a prepared assignment")
        try:
            layer_range = request["layer_range"]
            self.adapter, metrics = load_qwen3_layer_assignment(
                self.stage,
                start_layer=int(layer_range[0]), end_layer=int(layer_range[1]),
                has_embedding=bool(request.get("has_embedding", False)),
                has_lm_head=bool(request.get("has_lm_head", False)),
                device=str(self.resources.get("device", request.get("execution_device", "cpu"))),
                dtype=request.get("dtype"),
            )
            self.execution = {key: value for key, value in dict(metrics).items() if isinstance(value, (str, int, float, bool, type(None)))}
            self.execution["full_model_materialized"] = False
            self.execution["segment_materialized"] = True
            self.phase = "committed"
            return _response(
                "commit", status="committed", gate_passed=True,
                assignment=self.assignment,
                resources=self.resources,
                execution=self.execution,
                segment_materialized=True,
            )
        except Exception as exc:
            self._cleanup()
            self.phase = "idle"
            return _error("commit", "qwen3_sidecar_commit_failed", str(exc), status="execution_failed")

    def release(self, request: dict[str, Any], *, aborted: bool = False) -> dict[str, Any]:
        if self.phase not in {"prepared", "committed"}:
            if aborted and self.phase == "idle":
                return _response("abort", status="aborted", gate_passed=True, cleanup_complete=True)
            return _error("release" if not aborted else "abort", "qwen3_sidecar_release_invalid", "release does not match active state")
        self._cleanup()
        self.phase = "idle"
        return _response(
            "abort" if aborted else "release",
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
        candidate = Path(str(value or "")).expanduser().absolute().resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("local artifact escapes artifact root") from exc
        if must_exist and not candidate.is_file():
            raise ValueError("local input artifact is unavailable")
        if not must_exist:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    @staticmethod
    def _load_artifact(path: Path) -> dict[str, Any]:
        try:
            import torch

            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError("local tensor artifact could not be read") from exc
        if not isinstance(payload, dict):
            raise ValueError("local tensor artifact must contain an object")
        return payload

    @staticmethod
    def _file_evidence(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        return size, digest.hexdigest()

    @staticmethod
    def _save_artifact(path: Path, payload: dict[str, Any]) -> tuple[int, str]:
        import os
        import tempfile
        import torch

        fd, temporary = tempfile.mkstemp(prefix=".qwen3-artifact-", suffix=".pt", dir=str(path.parent))
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            torch.save(payload, str(temporary_path))
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return _RuntimeSession._file_evidence(path)

    @staticmethod
    def _tensor(value: Any, label: str) -> Any:
        if value is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError(f"{label} tensor is missing")
        if len(tuple(value.shape)) != 3 and label == "hidden_states":
            raise ValueError("hidden_states must be [batch, sequence, hidden]")
        return value

    @staticmethod
    def _cache_sequence_length(cache: Any, expected: int | None = None) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            try:
                return int(getter())
            except (TypeError, ValueError, IndexError):
                pass
        candidates: list[int] = []

        def visit(value: Any) -> None:
            shape = getattr(value, "shape", None)
            if shape is not None:
                try:
                    candidates.extend(int(item) for item in shape)
                except (TypeError, ValueError):
                    return
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        visit(cache)
        if expected is not None and int(expected) in candidates:
            return int(expected)
        if len(set(candidates)) == 1:
            return int(candidates[0])
        if not candidates:
            raise ValueError("KV cache sequence length is unavailable")
        raise ValueError("KV cache sequence length is ambiguous")

    @staticmethod
    def _dtype_matches(actual: Any, expected: Any) -> bool:
        aliases = {
            "float16": "torch.float16", "fp16": "torch.float16", "torch.float16": "torch.float16",
            "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16", "torch.bfloat16": "torch.bfloat16",
            "float32": "torch.float32", "fp32": "torch.float32", "torch.float32": "torch.float32",
        }
        return aliases.get(str(actual).lower(), str(actual).lower()) == aliases.get(str(expected).lower(), str(expected).lower())

    @staticmethod
    def _device_matches(actual: Any, expected: Any) -> bool:
        actual_value = str(actual).lower()
        expected_value = str(expected).lower()
        if expected_value == "cuda":
            return actual_value == "cuda" or actual_value.startswith("cuda:")
        return actual_value == expected_value

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        phase = str(request.get("phase", ""))
        if self.phase != "committed" or not self._same_identity(request, include_generation=False):
            return _error(phase, "qwen3_sidecar_not_committed", "execution does not match a committed assignment")
        if phase not in {"prefill", "decode"}:
            return _error(phase, "qwen3_sidecar_phase_invalid", "unsupported sidecar execution phase")
        if request.get("data_plane") != "local_artifact":
            return _error(phase, "qwen3_sidecar_data_plane_invalid", "sidecar execution requires local artifacts")
        try:
            import torch

            execution_generation = int(request.get("generation", -1))
            assignment_generation = int((self.request or {}).get("generation", -1))
            if execution_generation < assignment_generation:
                raise ValueError("execution generation is older than assignment generation")

            input_path = self._artifact_path(request, request.get("input_ref"), must_exist=True)
            output_path = self._artifact_path(request, request.get("output_ref"), must_exist=False)
            if self._file_evidence(input_path) != (
                int(request.get("input_bytes", -1)), str(request.get("input_sha256", "")),
            ):
                raise ValueError("local input artifact evidence does not match")
            input_payload = self._load_artifact(input_path)
            input_ids = input_payload.get("input_ids")
            hidden_states = input_payload.get("hidden_states")
            if (input_ids is None) == (hidden_states is None):
                raise ValueError("artifact must contain exactly one of input_ids or hidden_states")
            if input_ids is not None and phase != "prefill" and request.get("segment_index") != 0:
                raise ValueError("only the first segment may receive decode input_ids")
            past_key_values = None
            kv_ref = request.get("kv_ref")
            if phase == "decode":
                if not kv_ref:
                    raise ValueError("decode requires a segment KV artifact")
                kv_path = self._artifact_path(request, kv_ref, must_exist=True)
                if self._file_evidence(kv_path) != (
                    int(request.get("kv_bytes", -1)), str(request.get("kv_sha256", "")),
                ):
                    raise ValueError("local KV artifact evidence does not match")
                kv_payload = self._load_artifact(kv_path)
                past_key_values = kv_payload.get("past_key_values")
                if past_key_values is None:
                    raise ValueError("segment KV artifact is missing past_key_values")
            if input_ids is not None:
                input_ids = self._tensor(input_ids, "input_ids")
                if len(tuple(input_ids.shape)) != 2:
                    raise ValueError("input_ids must be [batch, sequence]")
                result = self.adapter.forward(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                current_batch = int(input_ids.shape[0])
                current_sequence = int(input_ids.shape[1])
            else:
                hidden_states = self._tensor(hidden_states, "hidden_states")
                result = self.adapter.forward(hidden_states=hidden_states, past_key_values=past_key_values, use_cache=True)
                current_batch = int(hidden_states.shape[0])
                current_sequence = int(hidden_states.shape[1])
            if current_batch != int(request.get("batch_size", 0) or 0):
                raise ValueError("artifact batch size does not match contract")
            cache = result.get("past_key_values")
            if cache is None:
                raise ValueError("sidecar execution returned no KV cache")
            requested_length = int(request.get("sequence_length", 0) or 0)
            past_length = (
                self._cache_sequence_length(past_key_values, requested_length - current_sequence)
                if phase == "decode" else 0
            )
            expected_length = current_sequence if phase == "prefill" else past_length + current_sequence
            cache_length = self._cache_sequence_length(cache, expected_length)
            if cache_length != expected_length or cache_length != requested_length:
                raise ValueError("sidecar KV sequence length does not match contract")
            output_hidden = result.get("hidden_states")
            output_logits = result.get("logits")
            output_tensor = output_hidden if output_hidden is not None else output_logits
            if output_tensor is None:
                raise ValueError("sidecar execution returned no output tensor")
            actual_dtype = str(getattr(output_tensor, "dtype", ""))
            actual_device = str(getattr(output_tensor, "device", ""))
            expected_dtype = str(request.get("dtype", ""))
            expected_device = str(request.get("device", ""))
            if expected_dtype and not self._dtype_matches(actual_dtype, expected_dtype):
                raise ValueError("sidecar output dtype does not match contract")
            if expected_device and not self._device_matches(actual_device, expected_device):
                raise ValueError("sidecar output device does not match contract")
            # Keep artifacts CPU-resident between local processes; the next
            # sidecar moves the handoff to its assigned device explicitly.
            payload = {
                "hidden_states": output_hidden.detach().to("cpu") if output_hidden is not None else None,
                "logits": output_logits.detach().to("cpu") if output_logits is not None else None,
                "past_key_values": cache,
            }
            output_bytes, output_sha256 = self._save_artifact(output_path, payload)
            chain_id = str(request.get("chain_id", ""))
            segment_index = int(request.get("segment_index", -1))
            layer_range = request.get("layer_range")
            if not isinstance(layer_range, list) or len(layer_range) != 2:
                raise ValueError("execution layer range is invalid")
            generation = execution_generation
            if generation < 0:
                raise ValueError("execution generation is invalid")
            handoff = None
            if bool(request.get("has_next_segment")):
                if output_hidden is None:
                    raise ValueError("non-final sidecar did not return hidden states")
                handoff = build_hidden_handoff(
                    output_hidden, chain_id=chain_id, from_segment=segment_index,
                    to_segment=segment_index + 1, sequence_length=current_sequence,
                )
            kv_contract = build_kv_contract(
                chain_id=chain_id, segment_index=segment_index,
                layer_range=layer_range, sequence_length=cache_length,
                batch_size=current_batch, dtype=actual_dtype, device=actual_device,
                phase=phase, generation=generation,
            )
            return _response(
                phase, status="executed", gate_passed=True,
                execution={
                    "data_plane": "local_artifact",
                    "artifact_bytes": output_bytes,
                    "artifact_sha256": output_sha256,
                    "full_model_materialized": False,
                    "segment_materialized": True,
                },
                hidden_handoff=handoff,
                kv_contract=kv_contract,
            )
        except Exception as exc:
            return _error(phase, "qwen3_sidecar_execution_failed", str(exc), status="execution_failed")

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict) or request.get("schema_version") != SCHEMA_VERSION or request.get("operation") != OPERATION:
            return _error(str(request.get("phase", "")) if isinstance(request, dict) else "", "qwen3_sidecar_protocol_invalid", "unsupported sidecar protocol", status="invalid_request")
        if request.get("read_only") is not True or request.get("network_access") != "disabled":
            return _error(str(request.get("phase", "")), "qwen3_sidecar_protocol_invalid", "sidecar must be read-only and network-disabled", status="invalid_request")
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
        return _error(phase, "qwen3_sidecar_phase_invalid", "unsupported sidecar phase", status="invalid_request")


def main() -> int:
    session = _RuntimeSession()
    try:
        for raw in sys.stdin.buffer:
            if len(raw) > MAX_FRAME_BYTES:
                print(json.dumps(_error("", "qwen3_sidecar_frame_oversize", "sidecar frame exceeds 256 KiB", status="invalid_request"), separators=(",", ":")), flush=True)
                continue
            try:
                request = json.loads(raw.decode("utf-8"))
                report = session.handle(request)
            except Exception as exc:
                report = _error("", "qwen3_sidecar_worker_failed", exc.__class__.__name__, status="worker_failed")
            print(json.dumps(report, ensure_ascii=True, separators=(",", ":")), flush=True)
    finally:
        session._cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
