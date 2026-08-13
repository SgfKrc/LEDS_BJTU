"""Qwen3 text pipeline adapter used by the isolated Transformers sidecar.

The main application deliberately does not import this module.  It contains
the small, architecture-specific contract needed to execute a selected
``model.layers.N`` range without materializing the other layers.  Real weight
loading remains a sidecar concern; this module only accepts an already built
model object and a filtered assignment.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable


QWEN3_ADAPTER_SCHEMA_VERSION = 1
QWEN3_MODEL_TYPE = "qwen3"
QWEN3_LAYER_PREFIX = "model.layers."
_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
_THINK_MARKERS = ("<|think|>", "</think>", "<think>")


class Qwen3AdapterError(ValueError):
    """The sidecar cannot prove that a Qwen3 request is safe to execute."""


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(value).split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def _has_nonempty_thinking(rendered: str) -> bool:
    """Allow the official empty ``<think></think>`` scaffold."""
    lower = str(rendered).lower()
    start = lower.find("<think>")
    end = lower.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start >= 0 and end >= 0:
        if str(rendered)[start + len("<think>"):end].strip():
            return True
        lower = lower[:start] + lower[end + len("</think>"):]
    return any(marker in lower for marker in _THINK_MARKERS)


def render_without_thinking(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Render a chat prompt with a hard API-level thinking disable.

    A tokenizer that silently ignores ``enable_thinking`` or emits a non-empty
    thinking block is rejected.  This is intentionally stricter than relying
    on a model card default.
    """
    if not isinstance(messages, list) or not messages:
        raise Qwen3AdapterError("messages must be a non-empty list")
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise Qwen3AdapterError("Qwen3 tokenizer has no chat template API")
    try:
        signature = inspect.signature(apply_template)
        parameters = signature.parameters.values()
        accepts_thinking = (
            "enable_thinking" in signature.parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        )
    except (TypeError, ValueError):
        accepts_thinking = True
    if not accepts_thinking:
        raise Qwen3AdapterError(
            "Qwen3 tokenizer does not accept enable_thinking=False"
        )
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise Qwen3AdapterError(
            "Qwen3 tokenizer does not accept enable_thinking=False"
        ) from exc
    rendered = str(rendered)
    if _has_nonempty_thinking(rendered):
        raise Qwen3AdapterError(
            "enable_thinking=False still produced a non-empty thinking marker"
        )
    return rendered


def _key_kind(key: str) -> tuple[str, int | None]:
    match = _LAYER_KEY.match(str(key))
    if match:
        return "layer", int(match.group(1))
    if str(key).startswith("model.embed_tokens."):
        return "embedding", None
    if str(key).startswith("model.norm."):
        return "norm", None
    if str(key).startswith("lm_head."):
        return "lm_head", None
    return "unknown", None


def select_qwen3_assignment_keys(
    keys: Iterable[str],
    *,
    start_layer: int,
    end_layer: int,
    has_embedding: bool,
    has_lm_head: bool,
    tie_word_embeddings: bool = False,
) -> list[str]:
    """Return only keys owned by one Qwen3 text pipeline segment.

    Unknown text/vision/MTP keys are rejected instead of being silently
    dropped.  Keys belonging to another layer or component are skipped, as
    they are expected to be present in another assignment.
    """
    if int(start_layer) < 0 or int(end_layer) <= int(start_layer):
        raise Qwen3AdapterError("invalid Qwen3 layer range")
    selected: list[str] = []
    for raw_key in keys:
        key = str(raw_key)
        kind, index = _key_kind(key)
        if kind == "unknown":
            raise Qwen3AdapterError(f"unsupported Qwen3 assignment key: {key}")
        if kind == "layer":
            if int(start_layer) <= int(index) < int(end_layer):
                selected.append(key)
        elif kind == "embedding" and (
            has_embedding or (has_lm_head and tie_word_embeddings)
        ):
            selected.append(key)
        elif kind == "norm":
            selected.append(key)
        elif kind == "lm_head" and has_lm_head:
            selected.append(key)
    return selected


def validate_qwen3_assignment(
    *,
    model_type: str,
    total_layers: int,
    start_layer: int,
    end_layer: int,
    has_embedding: bool,
    has_lm_head: bool,
    keys: Iterable[str],
    tie_word_embeddings: bool = False,
) -> dict[str, Any]:
    """Validate a metadata-only assignment before any tensor is read."""
    if str(model_type).lower() != QWEN3_MODEL_TYPE:
        raise Qwen3AdapterError("Qwen3 adapter requires model_type=qwen3")
    total_layers = int(total_layers)
    start_layer = int(start_layer)
    end_layer = int(end_layer)
    if total_layers <= 0 or start_layer < 0 or end_layer > total_layers or start_layer >= end_layer:
        raise Qwen3AdapterError("Qwen3 assignment range is outside model bounds")
    selected = select_qwen3_assignment_keys(
        keys,
        start_layer=start_layer,
        end_layer=end_layer,
        has_embedding=bool(has_embedding),
        has_lm_head=bool(has_lm_head),
        tie_word_embeddings=bool(tie_word_embeddings),
    )
    layer_indices = sorted({index for key in selected for kind, index in [_key_kind(key)] if kind == "layer"})
    missing_layers = [index for index in range(start_layer, end_layer) if index not in layer_indices]
    if missing_layers:
        raise Qwen3AdapterError(f"Qwen3 assignment is missing layers: {missing_layers[:8]}")
    selected_kinds = {_key_kind(key)[0] for key in selected}
    if has_embedding and "embedding" not in selected_kinds:
        raise Qwen3AdapterError("Qwen3 assignment is missing embedding weights")
    if has_lm_head and not (
        "lm_head" in selected_kinds
        or (tie_word_embeddings and "embedding" in selected_kinds)
    ):
        raise Qwen3AdapterError("Qwen3 assignment is missing LM Head weights")
    return {
        "schema_version": QWEN3_ADAPTER_SCHEMA_VERSION,
        "model_type": QWEN3_MODEL_TYPE,
        "layer_range": [start_layer, end_layer],
        "has_embedding": bool(has_embedding),
        "has_lm_head": bool(has_lm_head),
        "tie_word_embeddings": bool(tie_word_embeddings),
        "selected_key_count": len(selected),
        "selected_keys": selected,
    }


def load_qwen3_layer_assignment(
    model_path: str | Path,
    *,
    start_layer: int,
    end_layer: int,
    has_embedding: bool,
    has_lm_head: bool,
    device: str | None = None,
    dtype: str | None = None,
) -> tuple["Qwen3PipelineAdapter", dict[str, Any]]:
    """Materialize one Qwen3 text segment from a filtered Safetensors index.

    This function is only for the isolated Qwen3 execution environment.  It
    never calls ``from_pretrained`` and checks every key before ``get_tensor``.
    The caller must provide an assignment directory whose index is already
    bound to a C3 assignment manifest.
    """
    try:
        import torch
        import transformers
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise Qwen3AdapterError(
            "Qwen3 pipeline sidecar requires torch, accelerate, safetensors and transformers"
        ) from exc
    if _version_tuple(getattr(transformers, "__version__", "0")) < (4, 51, 0):
        raise Qwen3AdapterError("Qwen3 pipeline sidecar requires transformers>=4.51")

    root = Path(model_path).expanduser().absolute().resolve(strict=False)
    if not root.is_dir():
        raise Qwen3AdapterError("Qwen3 assignment directory does not exist")
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise Qwen3AdapterError("Qwen3 assignment requires a filtered Safetensors index")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise Qwen3AdapterError("Qwen3 assignment index is invalid") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise Qwen3AdapterError("Qwen3 assignment index has no weight map")
    if any(
        not isinstance(key, str) or not isinstance(filename, str)
        for key, filename in weight_map.items()
    ):
        raise Qwen3AdapterError("Qwen3 assignment index entries are invalid")

    config = AutoConfig.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False,
    )
    model_type = str(getattr(config, "model_type", "") or "").lower()
    total_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    assignment = validate_qwen3_assignment(
        model_type=model_type,
        total_layers=total_layers,
        start_layer=start_layer,
        end_layer=end_layer,
        has_embedding=has_embedding,
        has_lm_head=has_lm_head,
        keys=weight_map.keys(),
        tie_word_embeddings=tied,
    )
    selected_keys = set(assignment["selected_keys"])
    # A C3 filtered index must contain only this segment's keys.  Reject a
    # full-model index so an operator cannot bypass assignment scoping.
    unexpected = sorted(set(weight_map) - selected_keys)
    if unexpected:
        raise Qwen3AdapterError(
            f"Qwen3 assignment index contains unassigned keys: {unexpected[:4]}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise Qwen3AdapterError("Qwen3 CUDA assignment requested without CUDA")
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype is None:
        target_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        target_dtype = dtype_map.get(str(dtype).lower())
        if target_dtype is None:
            raise Qwen3AdapterError("unsupported Qwen3 assignment dtype")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    files_to_keys: dict[str, list[str]] = {}
    for key in assignment["selected_keys"]:
        filename = str(weight_map[key]).replace("\\", "/")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise Qwen3AdapterError("Qwen3 assignment shard path is unsafe")
        shard = (root / relative).resolve(strict=False)
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise Qwen3AdapterError("Qwen3 assignment shard escapes model root") from exc
        if not shard.is_file():
            raise Qwen3AdapterError(f"Qwen3 assignment shard is missing: {filename}")
        files_to_keys.setdefault(filename, []).append(key)

    started = time.monotonic()
    loaded_keys: set[str] = set()
    source_bytes = 0
    for filename, keys in files_to_keys.items():
        shard = root / filename
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(set(keys) - available)
            if missing:
                raise Qwen3AdapterError(
                    f"Qwen3 assignment shard is missing keys: {missing[:4]}"
                )
            for key in keys:
                if key not in selected_keys:
                    raise Qwen3AdapterError(
                        f"refusing to materialize unassigned Qwen3 key: {key}"
                    )
                tensor = handle.get_tensor(key)
                source_bytes += tensor.numel() * tensor.element_size()
                set_module_tensor_to_device(
                    model, key, device, value=tensor, dtype=target_dtype,
                )
                loaded_keys.add(key)
                del tensor
    if loaded_keys != selected_keys:
        raise Qwen3AdapterError("Qwen3 assignment materialization is incomplete")

    body = model.model
    kept_layers = list(body.layers[int(start_layer):int(end_layer)])
    body.layers = torch.nn.ModuleList(kept_layers)
    if tied and has_lm_head:
        source_weight = getattr(getattr(body, "embed_tokens", None), "weight", None)
        if source_weight is None or source_weight.device.type == "meta":
            raise Qwen3AdapterError("tied Qwen3 output weight was not materialized")
        model.lm_head.weight = source_weight
    if not has_embedding:
        body.embed_tokens = None
    if not has_lm_head:
        model.lm_head = None
    retained_meta = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    if retained_meta:
        raise Qwen3AdapterError(
            f"Qwen3 assignment retained meta parameters: {retained_meta[:4]}"
        )
    model.eval()
    adapter = Qwen3PipelineAdapter(
        model,
        start_layer=int(start_layer),
        end_layer=int(end_layer),
        has_embedding=bool(has_embedding),
        has_lm_head=bool(has_lm_head),
        total_layers=total_layers,
    )
    metrics = {
        "schema_version": QWEN3_ADAPTER_SCHEMA_VERSION,
        "mode": "filtered_safetensors_assignment",
        "model_type": QWEN3_MODEL_TYPE,
        "layer_range": [int(start_layer), int(end_layer)],
        "selected_tensor_count": len(loaded_keys),
        "source_tensor_bytes": source_bytes,
        "materialized_tensor_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "device": device,
        "dtype": str(target_dtype),
        "tie_word_embeddings": tied,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "full_model_materialized": False,
    }
    return adapter, metrics


def _layer_result(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple):
        return value[0], value[1] if len(value) > 1 else None
    if isinstance(value, list):
        return value[0], value[1] if len(value) > 1 else None
    hidden = getattr(value, "hidden_states", None)
    if hidden is not None:
        return hidden, getattr(value, "past_key_value", None)
    return value, None


class Qwen3PipelineAdapter:
    """Execute one Qwen3 text layer range against an isolated model object."""

    def __init__(
        self,
        model: Any,
        *,
        start_layer: int,
        end_layer: int,
        has_embedding: bool,
        has_lm_head: bool,
        total_layers: int | None = None,
    ) -> None:
        self.model = model
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.has_embedding = bool(has_embedding)
        self.has_lm_head = bool(has_lm_head)
        base = getattr(model, "model", None)
        if base is None:
            raise Qwen3AdapterError("Qwen3ForCausalLM model body is missing")
        self.body = base
        layers = getattr(base, "layers", None)
        if layers is None:
            raise Qwen3AdapterError("Qwen3 model.layers is missing")
        actual_total = len(layers)
        self.total_layers = int(total_layers if total_layers is not None else actual_total)
        expected_local_layers = self.end_layer - self.start_layer
        if actual_total not in {self.total_layers, expected_local_layers}:
            raise Qwen3AdapterError("Qwen3 model layer count does not match assignment")
        if self.start_layer < 0 or self.end_layer > self.total_layers or self.start_layer >= self.end_layer:
            raise Qwen3AdapterError("Qwen3 adapter layer range is invalid")
        if self.has_embedding and getattr(base, "embed_tokens", None) is None:
            raise Qwen3AdapterError("Qwen3 embedding is required but unavailable")
        if self.has_lm_head and getattr(model, "lm_head", None) is None:
            raise Qwen3AdapterError("Qwen3 LM Head is required but unavailable")
        if getattr(base, "norm", None) is None:
            raise Qwen3AdapterError("Qwen3 final norm is missing")
        self._layers = layers[self.start_layer:self.end_layer] if actual_total == self.total_layers else layers

    @staticmethod
    def _device_dtype(model: Any) -> tuple[Any, Any]:
        try:
            parameter = next(model.parameters())
        except (AttributeError, StopIteration):
            return None, None
        return parameter.device, parameter.dtype

    def _call_layer(
        self,
        layer: Any,
        hidden_states: Any,
        *,
        past: Any,
        use_cache: bool,
        attention_mask: Any,
        position_ids: Any,
        position_embeddings: Any,
        cache_position: Any,
    ) -> tuple[Any, Any]:
        try:
            parameters = inspect.signature(layer.forward).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        candidates = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "cache_position": cache_position,
            "use_cache": use_cache,
        }
        for name, value in candidates.items():
            if name in parameters and value is not None:
                kwargs[name] = value
        for name in ("past_key_values", "past_key_value", "layer_past"):
            if name in parameters and past is not None:
                kwargs[name] = past
                break
        output = layer(hidden_states, **kwargs)
        return _layer_result(output)

    def _shared_transformers_cache(self, past_key_values: Any, use_cache: bool) -> Any:
        if not use_cache:
            return None
        try:
            parameters = inspect.signature(self._layers[0].forward).parameters
        except (IndexError, TypeError, ValueError):
            parameters = {}
        if "past_key_values" not in parameters:
            return None
        try:
            from transformers.cache_utils import Cache, DynamicCache
        except (ImportError, AttributeError):
            return None
        if isinstance(past_key_values, Cache):
            return past_key_values
        if isinstance(past_key_values, (tuple, list)):
            return DynamicCache.from_legacy_cache(tuple(past_key_values))
        if past_key_values is not None:
            raise Qwen3AdapterError("unsupported Qwen3 KV cache representation")
        return DynamicCache()

    def _attention_masks(
        self,
        hidden_states: Any,
        attention_mask: Any,
        cache_position: Any,
        cache: Any,
        position_ids: Any,
    ) -> dict[str, Any]:
        try:
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
        except (ImportError, AttributeError):
            return {"full_attention": attention_mask, "sliding_attention": attention_mask}
        config = getattr(self.model, "config", None)
        if config is None:
            raise Qwen3AdapterError("Qwen3 model config is unavailable")
        kwargs = {
            "config": config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        masks = {"full_attention": create_causal_mask(**kwargs)}
        if any(getattr(layer, "attention_type", "full_attention") == "sliding_attention" for layer in self._layers):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
        return masks

    def forward(
        self,
        *,
        input_ids: Any = None,
        hidden_states: Any = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        apply_lm_head: bool = True,
        attention_mask: Any = None,
        position_ids: Any = None,
        cache_position: Any = None,
    ) -> dict[str, Any]:
        """Run the assigned segment and return hidden states/logits plus KV."""
        if (input_ids is None) == (hidden_states is None):
            raise Qwen3AdapterError("provide exactly one of input_ids or hidden_states")
        if input_ids is not None and not self.has_embedding:
            raise Qwen3AdapterError("input_ids require an embedding-owning segment")
        device, dtype = self._device_dtype(self.model)
        if input_ids is not None:
            if device is not None:
                input_ids = input_ids.to(device)
            hidden_states = self.body.embed_tokens(input_ids)
        elif device is not None and hidden_states.device != device:
            hidden_states = hidden_states.to(device)
        if dtype is not None and getattr(hidden_states, "dtype", dtype) != dtype:
            hidden_states = hidden_states.to(dtype=dtype)
        shared_cache = self._shared_transformers_cache(past_key_values, use_cache)
        past_seen_tokens = 0
        if shared_cache is not None:
            try:
                past_seen_tokens = int(shared_cache.get_seq_length())
            except (AttributeError, TypeError, IndexError) as exc:
                raise Qwen3AdapterError("Qwen3 KV cache length is unavailable") from exc
        if cache_position is None:
            try:
                import torch

                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + hidden_states.shape[1],
                    device=hidden_states.device,
                )
            except Exception:
                cache_position = None
        if position_ids is None:
            position_ids = (
                cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1)
                if cache_position is not None else None
            )
        position_embeddings = None
        rotary = getattr(self.body, "rotary_emb", None)
        if callable(rotary) and position_ids is not None:
            try:
                position_embeddings = rotary(hidden_states, position_ids)
            except (TypeError, RuntimeError):
                position_embeddings = None

        masks = self._attention_masks(
            hidden_states, attention_mask, cache_position, shared_cache, position_ids,
        ) if shared_cache is not None else {
            "full_attention": attention_mask,
            "sliding_attention": attention_mask,
        }
        cache_values: list[Any] = []
        saved_layer_indices: list[tuple[Any, int]] = []
        if shared_cache is not None:
            for local_index, layer in enumerate(self._layers):
                attention = getattr(layer, "self_attn", None)
                if attention is not None and hasattr(attention, "layer_idx"):
                    saved_layer_indices.append((attention, attention.layer_idx))
                    attention.layer_idx = local_index
        try:
            with _no_grad():
                for local_index, layer in enumerate(self._layers):
                    if shared_cache is not None:
                        past = shared_cache
                    elif isinstance(past_key_values, (tuple, list)) and local_index < len(past_key_values):
                        past = past_key_values[local_index]
                    else:
                        past = past_key_values
                    layer_mask = masks.get(
                        getattr(layer, "attention_type", "full_attention"),
                        masks["full_attention"],
                    )
                    hidden_states, present = self._call_layer(
                        layer,
                        hidden_states,
                        past=past,
                        use_cache=use_cache,
                        attention_mask=layer_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        cache_position=cache_position,
                    )
                    if use_cache and shared_cache is None:
                        cache_values.append(present)

                result: dict[str, Any] = {}
                if self.has_lm_head and apply_lm_head:
                    normalized = self.body.norm(hidden_states)
                    result["logits"] = self.model.lm_head(normalized)
                else:
                    result["hidden_states"] = hidden_states
                if use_cache:
                    if shared_cache is not None:
                        result["past_key_values"] = shared_cache.to_legacy_cache()
                    else:
                        result["past_key_values"] = tuple(cache_values)
                return result
        finally:
            for attention, original_index in saved_layer_indices:
                attention.layer_idx = original_index


class _no_grad:
    """Tiny lazy torch.no_grad context; importing this module stays torch-free."""

    def __enter__(self):
        import torch

        self._context = torch.no_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        return self._context.__exit__(exc_type, exc, traceback)


__all__ = [
    "QWEN3_ADAPTER_SCHEMA_VERSION",
    "QWEN3_LAYER_PREFIX",
    "QWEN3_MODEL_TYPE",
    "Qwen3AdapterError",
    "Qwen3PipelineAdapter",
    "load_qwen3_layer_assignment",
    "render_without_thinking",
    "select_qwen3_assignment_keys",
    "validate_qwen3_assignment",
]
