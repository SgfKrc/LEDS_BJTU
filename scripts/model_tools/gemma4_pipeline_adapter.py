"""Isolated Gemma 4 Unified text-layer assignment loader.

The main QLH runtime intentionally does not import this module. Official
Gemma 4 PyTorch checkpoints require a newer Transformers runtime and use the
``model.language_model.*`` namespace, so they must not reuse the Qwen2 loader.
"""

from __future__ import annotations

from collections import UserDict
import inspect
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable


GEMMA4_ADAPTER_SCHEMA_VERSION = 1
GEMMA4_MODEL_TYPE = "gemma4_unified"
GEMMA4_LAYER_PREFIX = "model.language_model.layers."
_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class Gemma4AdapterError(ValueError):
    """The isolated runtime cannot prove a Gemma 4 assignment is safe."""


class _MaskCacheView:
    """Read-only cache geometry for a pure shared-KV decode segment."""

    def __init__(self, source: Any, *, past_length: int, key_length: int) -> None:
        self._source = source
        self._past_length = int(past_length)
        self._key_length = int(key_length)
        self.is_sliding = getattr(source, "is_sliding", None)
        self.is_compileable = bool(getattr(source, "is_compileable", False))

    def get_seq_length(self, *args: Any, **kwargs: Any) -> int:
        return self._past_length

    def get_mask_sizes(self, query_length: int, layer_idx: int | None = None) -> tuple[int, int]:
        del query_length, layer_idx
        return self._key_length, 0


def _sequence_axis(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(tuple(shape)) < 3:
        raise Gemma4AdapterError("Gemma 4 shared-KV tensor shape is invalid")
    return int(shape[-2])


def _key_kind(key: str) -> tuple[str, int | None]:
    value = str(key)
    match = _LAYER_KEY.match(value)
    if match:
        return "layer", int(match.group(1))
    if value.startswith("model.language_model.embed_tokens."):
        return "embedding", None
    if value.startswith("model.language_model.norm."):
        return "norm", None
    if value.startswith("lm_head."):
        return "lm_head", None
    if value.startswith(("model.embed_vision.", "model.embed_audio.")):
        return "multimodal", None
    if value.startswith("model.language_model."):
        return "text_aux", None
    return "unknown", None


def select_gemma4_assignment_keys(
    keys: Iterable[str],
    *,
    start_layer: int,
    end_layer: int,
    has_embedding: bool,
    has_lm_head: bool,
    tie_word_embeddings: bool = False,
) -> list[str]:
    """Select one text-only Gemma 4 segment without admitting modalities."""
    start_layer = int(start_layer)
    end_layer = int(end_layer)
    if start_layer < 0 or end_layer <= start_layer:
        raise Gemma4AdapterError("invalid Gemma 4 layer range")
    selected: list[str] = []
    for raw_key in keys:
        key = str(raw_key)
        kind, index = _key_kind(key)
        if kind == "unknown":
            raise Gemma4AdapterError(f"unsupported Gemma 4 assignment key: {key}")
        if kind == "layer" and start_layer <= int(index) < end_layer:
            selected.append(key)
        elif kind == "embedding" and (
            has_embedding or (has_lm_head and tie_word_embeddings)
        ):
            selected.append(key)
        elif kind == "norm":
            selected.append(key)
        elif kind in {"lm_head", "text_aux"} and has_lm_head:
            selected.append(key)
    return selected


def validate_gemma4_assignment(
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
    """Validate a filtered Gemma 4 text assignment before tensor reads."""
    if str(model_type).lower() != GEMMA4_MODEL_TYPE:
        raise Gemma4AdapterError(
            "Gemma 4 adapter requires model_type=gemma4_unified"
        )
    total_layers = int(total_layers)
    start_layer = int(start_layer)
    end_layer = int(end_layer)
    if (
        total_layers <= 0
        or start_layer < 0
        or end_layer > total_layers
        or start_layer >= end_layer
    ):
        raise Gemma4AdapterError("Gemma 4 assignment range is outside model bounds")
    selected = select_gemma4_assignment_keys(
        keys,
        start_layer=start_layer,
        end_layer=end_layer,
        has_embedding=bool(has_embedding),
        has_lm_head=bool(has_lm_head),
        tie_word_embeddings=bool(tie_word_embeddings),
    )
    layer_indices = sorted({
        index
        for key in selected
        for kind, index in [_key_kind(key)]
        if kind == "layer"
    })
    missing_layers = [
        index for index in range(start_layer, end_layer)
        if index not in layer_indices
    ]
    if missing_layers:
        raise Gemma4AdapterError(
            f"Gemma 4 assignment is missing layers: {missing_layers[:8]}"
        )
    kinds = {_key_kind(key)[0] for key in selected}
    if has_embedding and "embedding" not in kinds:
        raise Gemma4AdapterError("Gemma 4 assignment is missing embedding weights")
    if has_lm_head and not (
        "lm_head" in kinds
        or (tie_word_embeddings and "embedding" in kinds)
    ):
        raise Gemma4AdapterError("Gemma 4 assignment is missing LM Head weights")
    return {
        "schema_version": GEMMA4_ADAPTER_SCHEMA_VERSION,
        "model_type": GEMMA4_MODEL_TYPE,
        "layer_prefix": GEMMA4_LAYER_PREFIX,
        "layer_range": [start_layer, end_layer],
        "has_embedding": bool(has_embedding),
        "has_lm_head": bool(has_lm_head),
        "tie_word_embeddings": bool(tie_word_embeddings),
        "selected_key_count": len(selected),
        "selected_keys": selected,
        "multimodal_materialized": False,
        "full_model_materialized": False,
    }


def load_gemma4_text_layer_assignment(
    model_path: str | Path,
    *,
    start_layer: int,
    end_layer: int,
    has_embedding: bool,
    has_lm_head: bool,
    device: str | None = None,
    dtype: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Materialize a filtered text segment in an isolated Transformers runtime."""
    try:
        import torch
        import transformers
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
    except ImportError as exc:
        raise Gemma4AdapterError(
            "Gemma 4 sidecar requires torch, accelerate, safetensors and transformers"
        ) from exc

    auto_config = getattr(transformers, "AutoConfig", None)
    auto_model = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_config is None or auto_model is None:
        raise Gemma4AdapterError(
            "Gemma 4 sidecar Transformers lacks AutoModelForImageTextToText"
        )

    root = Path(model_path).expanduser().absolute().resolve(strict=False)
    if not root.is_dir():
        raise Gemma4AdapterError("Gemma 4 assignment directory does not exist")
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise Gemma4AdapterError(
            "Gemma 4 assignment requires a filtered Safetensors index"
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise Gemma4AdapterError("Gemma 4 assignment index is invalid") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise Gemma4AdapterError("Gemma 4 assignment index has no weight map")
    if any(
        not isinstance(key, str) or not isinstance(filename, str)
        for key, filename in weight_map.items()
    ):
        raise Gemma4AdapterError("Gemma 4 assignment index entries are invalid")

    try:
        config = auto_config.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False,
        )
    except Exception as exc:
        raise Gemma4AdapterError(
            "Gemma 4 sidecar cannot construct gemma4_unified config"
        ) from exc
    text_config = getattr(config, "text_config", None)
    total_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    tied = bool(getattr(text_config, "tie_word_embeddings", False))
    assignment = validate_gemma4_assignment(
        model_type=getattr(config, "model_type", ""),
        total_layers=total_layers,
        start_layer=start_layer,
        end_layer=end_layer,
        has_embedding=has_embedding,
        has_lm_head=has_lm_head,
        keys=weight_map.keys(),
        tie_word_embeddings=tied,
    )
    selected_keys = set(assignment["selected_keys"])
    unexpected = sorted(set(weight_map) - selected_keys)
    if unexpected:
        raise Gemma4AdapterError(
            f"Gemma 4 assignment index contains unassigned keys: {unexpected[:4]}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise Gemma4AdapterError("Gemma 4 CUDA assignment requested without CUDA")
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
            raise Gemma4AdapterError("unsupported Gemma 4 assignment dtype")

    try:
        with init_empty_weights():
            model = auto_model.from_config(config, trust_remote_code=False)
    except Exception as exc:
        raise Gemma4AdapterError(
            "Gemma 4 sidecar cannot build a meta model skeleton"
        ) from exc

    files_to_keys: dict[str, list[str]] = {}
    for key in assignment["selected_keys"]:
        filename = str(weight_map[key]).replace("\\", "/")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise Gemma4AdapterError("Gemma 4 assignment shard path is unsafe")
        shard = (root / relative).resolve(strict=False)
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise Gemma4AdapterError(
                "Gemma 4 assignment shard escapes model root"
            ) from exc
        if not shard.is_file():
            raise Gemma4AdapterError(
                f"Gemma 4 assignment shard is missing: {filename}"
            )
        files_to_keys.setdefault(filename, []).append(key)

    started = time.monotonic()
    loaded_keys: set[str] = set()
    source_bytes = 0
    for filename, keys in files_to_keys.items():
        shard = root / filename
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            missing = sorted(set(keys) - set(handle.keys()))
            if missing:
                raise Gemma4AdapterError(
                    f"Gemma 4 assignment shard is missing keys: {missing[:4]}"
                )
            for key in keys:
                if key not in selected_keys:
                    raise Gemma4AdapterError(
                        f"refusing to materialize unassigned Gemma 4 key: {key}"
                    )
                tensor = handle.get_tensor(key)
                source_bytes += tensor.numel() * tensor.element_size()
                set_module_tensor_to_device(
                    model, key, device, value=tensor, dtype=target_dtype,
                )
                loaded_keys.add(key)
                del tensor
    if loaded_keys != selected_keys:
        raise Gemma4AdapterError("Gemma 4 assignment materialization is incomplete")

    wrapper = getattr(model, "model", None)
    body = getattr(wrapper, "language_model", None)
    if body is None or getattr(body, "layers", None) is None:
        raise Gemma4AdapterError("Gemma 4 language_model skeleton is unavailable")
    body.layers = torch.nn.ModuleList(
        list(body.layers[int(start_layer):int(end_layer)])
    )
    if tied and has_lm_head:
        source_weight = getattr(getattr(body, "embed_tokens", None), "weight", None)
        if source_weight is None or source_weight.device.type == "meta":
            raise Gemma4AdapterError("tied Gemma 4 output weight was not materialized")
        model.lm_head.weight = source_weight
    if not has_embedding:
        body.embed_tokens = None
    if not has_lm_head:
        model.lm_head = None
    if hasattr(wrapper, "embed_vision"):
        wrapper.embed_vision = None
    if hasattr(wrapper, "embed_audio"):
        wrapper.embed_audio = None
    retained_meta = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    if retained_meta:
        raise Gemma4AdapterError(
            f"Gemma 4 assignment retained meta parameters: {retained_meta[:4]}"
        )
    model.eval()
    metrics = {
        **assignment,
        "mode": "filtered_safetensors_assignment",
        "selected_tensor_count": len(loaded_keys),
        "source_tensor_bytes": source_bytes,
        "materialized_tensor_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "device": device,
        "dtype": str(target_dtype),
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "transformers_version": str(getattr(transformers, "__version__", "")),
    }
    return model, metrics


class Gemma4PipelineAdapter:
    """Execute one official Gemma 4 Unified text segment.

    Gemma 4 keeps absolute layer identities because layer type, cache policy,
    and shared-KV ownership are derived from the original layer index.  The
    adapter therefore never applies Qwen3's local layer-index rewrite.
    """

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
        wrapper = getattr(model, "model", None)
        body = getattr(wrapper, "language_model", None)
        if body is None:
            raise Gemma4AdapterError("Gemma 4 language_model body is missing")
        self.body = body
        self.text_config = getattr(getattr(model, "config", None), "text_config", None)
        if self.text_config is None:
            raise Gemma4AdapterError("Gemma 4 text_config is missing")
        configured_total = int(
            total_layers
            if total_layers is not None
            else getattr(self.text_config, "num_hidden_layers", 0) or 0
        )
        if (
            configured_total <= 0
            or self.start_layer < 0
            or self.end_layer > configured_total
            or self.start_layer >= self.end_layer
        ):
            raise Gemma4AdapterError("Gemma 4 adapter layer range is invalid")
        self.total_layers = configured_total
        layers = getattr(body, "layers", None)
        if layers is None:
            raise Gemma4AdapterError("Gemma 4 language_model.layers is missing")
        expected_local = self.end_layer - self.start_layer
        actual_count = len(layers)
        if actual_count == configured_total:
            self._layers = list(layers[self.start_layer:self.end_layer])
        elif actual_count == expected_local:
            self._layers = list(layers)
        else:
            raise Gemma4AdapterError("Gemma 4 model layer count does not match assignment")
        layer_types = list(getattr(self.text_config, "layer_types", []) or [])
        if len(layer_types) != configured_total:
            raise Gemma4AdapterError("Gemma 4 layer_types does not cover the model")
        self._layer_indices = list(range(self.start_layer, self.end_layer))
        self._layer_types = [layer_types[index] for index in self._layer_indices]
        for layer, absolute_index in zip(self._layers, self._layer_indices):
            declared = getattr(layer, "layer_idx", absolute_index)
            attention = getattr(layer, "self_attn", None)
            attention_index = getattr(attention, "layer_idx", absolute_index)
            if int(declared) != absolute_index or int(attention_index) != absolute_index:
                raise Gemma4AdapterError("Gemma 4 segment lost its absolute layer identity")
        if self.has_embedding and getattr(body, "embed_tokens", None) is None:
            raise Gemma4AdapterError("Gemma 4 embedding is required but unavailable")
        if self.has_lm_head and getattr(model, "lm_head", None) is None:
            raise Gemma4AdapterError("Gemma 4 LM Head is required but unavailable")
        if getattr(body, "norm", None) is None:
            raise Gemma4AdapterError("Gemma 4 final norm is missing")
        if not callable(getattr(body, "rotary_emb", None)):
            raise Gemma4AdapterError("Gemma 4 rotary embedding is unavailable")

    @staticmethod
    def _device_dtype(model: Any) -> tuple[Any, Any]:
        try:
            parameter = next(model.parameters())
        except (AttributeError, StopIteration):
            return None, None
        return parameter.device, parameter.dtype

    def _cache(self, past_key_values: Any, use_cache: bool) -> Any:
        if not use_cache:
            if past_key_values is not None:
                raise Gemma4AdapterError("Gemma 4 past KV requires use_cache=true")
            return None
        try:
            from transformers.cache_utils import Cache, DynamicCache
        except (ImportError, AttributeError) as exc:
            raise Gemma4AdapterError("Gemma 4 sidecar has no Transformers Cache API") from exc
        if isinstance(past_key_values, Cache):
            return past_key_values
        if past_key_values is not None:
            raise Gemma4AdapterError("Gemma 4 requires the native Cache representation")
        try:
            return DynamicCache(config=self.text_config)
        except TypeError:
            return DynamicCache()

    def cache_sequence_length(self, cache: Any) -> int:
        if cache is None:
            return 0
        getter = getattr(cache, "get_seq_length", None)
        if not callable(getter):
            raise Gemma4AdapterError("Gemma 4 KV cache length is unavailable")
        lengths: set[int] = set()
        for layer, absolute_index in zip(self._layers, self._layer_indices):
            attention = getattr(layer, "self_attn", None)
            if bool(getattr(attention, "is_kv_shared_layer", False)):
                continue
            try:
                value = int(getter(absolute_index))
            except (TypeError, ValueError, IndexError):
                try:
                    value = int(getter(layer_idx=absolute_index))
                except (TypeError, ValueError, IndexError) as exc:
                    raise Gemma4AdapterError("Gemma 4 KV cache length is unavailable") from exc
            if value > 0:
                lengths.add(value)
        if len(lengths) > 1:
            raise Gemma4AdapterError("Gemma 4 segment KV cache lengths diverged")
        return next(iter(lengths), 0)

    @staticmethod
    def _cache_global_sequence_length(cache: Any) -> int:
        if cache is None:
            return 0
        getter = getattr(cache, "get_seq_length", None)
        if not callable(getter):
            raise Gemma4AdapterError("Gemma 4 KV cache length is unavailable")
        try:
            return int(getter())
        except (TypeError, ValueError, IndexError) as exc:
            raise Gemma4AdapterError("Gemma 4 global KV cache length is unavailable") from exc

    @staticmethod
    def _shared_kv(value: Any) -> UserDict:
        if value is None:
            return UserDict()
        if not isinstance(value, (dict, UserDict)):
            raise Gemma4AdapterError("Gemma 4 shared-KV state must be a mapping")
        normalized = UserDict()
        for layer_type, pair in value.items():
            if (
                not isinstance(layer_type, str)
                or not isinstance(pair, (tuple, list))
                or len(pair) != 2
            ):
                raise Gemma4AdapterError("Gemma 4 shared-KV entry is invalid")
            key, item = pair
            key_length = _sequence_axis(key)
            if _sequence_axis(item) != key_length:
                raise Gemma4AdapterError("Gemma 4 shared-KV key/value lengths differ")
            normalized[layer_type] = (key, item)
        return normalized

    def _attention_masks(
        self,
        hidden_states: Any,
        attention_mask: Any,
        cache: Any,
        position_ids: Any,
    ) -> dict[str, Any]:
        required = set(self._layer_types)
        if isinstance(attention_mask, dict):
            missing = sorted(required - set(attention_mask))
            if missing:
                raise Gemma4AdapterError(
                    f"Gemma 4 attention mask is missing layer types: {missing}"
                )
            return dict(attention_mask)
        try:
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
        except (ImportError, AttributeError) as exc:
            raise Gemma4AdapterError("Gemma 4 sidecar has no official masking API") from exc
        try:
            parameters = inspect.signature(create_causal_mask).parameters
        except (TypeError, ValueError):
            parameters = {}
        embeds_name = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
        kwargs = {
            "config": self.text_config,
            embeds_name: hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        result: dict[str, Any] = {}
        if "full_attention" in required:
            result["full_attention"] = create_causal_mask(**kwargs)
        if "sliding_attention" in required:
            result["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
        missing = sorted(required - set(result))
        if missing:
            raise Gemma4AdapterError(
                f"Gemma 4 layer type has no supported mask builder: {missing}"
            )
        return result

    def forward(
        self,
        *,
        input_ids: Any = None,
        hidden_states: Any = None,
        past_key_values: Any = None,
        shared_kv_states: Any = None,
        use_cache: bool = True,
        apply_lm_head: bool = True,
        attention_mask: Any = None,
        position_ids: Any = None,
        cache_position: Any = None,
    ) -> dict[str, Any]:
        """Run one text segment and preserve native cache/shared-KV semantics."""
        if (input_ids is None) == (hidden_states is None):
            raise Gemma4AdapterError("provide exactly one of input_ids or hidden_states")
        if input_ids is not None and not self.has_embedding:
            raise Gemma4AdapterError("input_ids require the embedding-owning segment")
        device, dtype = self._device_dtype(self.model)
        if input_ids is not None:
            if device is not None:
                input_ids = input_ids.to(device)
            hidden_states = self.body.embed_tokens(input_ids)
        elif device is not None and hidden_states.device != device:
            hidden_states = hidden_states.to(device)
        if dtype is not None and getattr(hidden_states, "dtype", dtype) != dtype:
            hidden_states = hidden_states.to(dtype=dtype)
        cache = self._cache(past_key_values, bool(use_cache))
        shared = self._shared_kv(shared_kv_states)
        current_length = int(hidden_states.shape[1])
        # The segment-local cache may be empty for shared-KV layers, while the
        # native cache still exposes the upstream sequence globally.  Mask and
        # position decisions must use that global watermark.
        shared_lengths = {_sequence_axis(pair[0]) for pair in shared.values()}
        if len(shared_lengths) > 1:
            raise Gemma4AdapterError("Gemma 4 shared-KV layer types have divergent lengths")
        cache_length = self._cache_global_sequence_length(cache)
        pure_shared = bool(shared_lengths) and all(
            bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False))
            for layer in self._layers
        )
        # A later segment receives the upstream cache object for eventual decode,
        # but its prefill input is still the complete sequence.  In that case
        # the shared KV length and current input length match, so positions and
        # masks must start at zero instead of counting upstream cache entries.
        past_length = cache_length
        shared_length = next(iter(shared_lengths), 0)
        if pure_shared:
            if shared_length < current_length:
                raise Gemma4AdapterError("Gemma 4 shared-KV sequence is shorter than input")
            past_length = shared_length - current_length
        elif cache_length == current_length and shared_lengths == {current_length}:
            past_length = 0
        if past_length == 0 and shared_lengths:
            if shared_length < current_length:
                raise Gemma4AdapterError("Gemma 4 shared-KV sequence is shorter than input")
            past_length = shared_length - current_length
        try:
            import torch
        except ImportError as exc:
            raise Gemma4AdapterError("Gemma 4 sidecar requires torch") from exc
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + current_length,
                device=hidden_states.device,
            )
        elif (
            len(tuple(cache_position.shape)) != 1
            or int(cache_position.shape[0]) != current_length
            or int(cache_position[0]) != past_length
        ):
            raise Gemma4AdapterError("Gemma 4 cache_position does not match KV state")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1)
        # When positions explicitly describe a cross-segment prefill, the
        # native cache still contains upstream layers and would make the
        # official mask builder allocate ``past + current`` keys.  The segment
        # layers consume the complete shared KV directly, so build a fresh
        # current-sequence mask while retaining the cache for later decode.
        if pure_shared and cache is not None and past_length > 0:
            mask_cache = _MaskCacheView(
                cache, past_length=past_length, key_length=shared_length,
            )
        else:
            mask_cache = cache if past_length == cache_length else None
        masks = self._attention_masks(
            hidden_states, attention_mask, mask_cache, position_ids,
        )
        position_embeddings = {
            layer_type: self.body.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self._layer_types)
        }
        with torch.no_grad():
            for layer, layer_type in zip(self._layers, self._layer_types):
                attention = getattr(layer, "self_attn", None)
                if (
                    bool(getattr(attention, "is_kv_shared_layer", False))
                    and layer_type not in shared
                ):
                    raise Gemma4AdapterError(
                        f"Gemma 4 shared-KV source is missing for {layer_type}"
                    )
                hidden_states = layer(
                    hidden_states,
                    shared_kv_states=shared,
                    position_embeddings=position_embeddings[layer_type],
                    attention_mask=masks[layer_type],
                    position_ids=position_ids,
                    past_key_values=cache,
                )
                if isinstance(hidden_states, (tuple, list)):
                    hidden_states = hidden_states[0]
        logical_length = past_length + current_length
        output_shared_lengths: dict[str, int] = {}
        for layer_type, pair in shared.items():
            length = _sequence_axis(pair[0])
            if _sequence_axis(pair[1]) != length or length != logical_length:
                raise Gemma4AdapterError("Gemma 4 shared-KV output length is inconsistent")
            output_shared_lengths[layer_type] = length
        result: dict[str, Any] = {
            "past_key_values": cache,
            "shared_kv_states": shared,
            "shared_kv_sequence_lengths": output_shared_lengths,
            "sequence_length": logical_length,
        }
        if self.has_lm_head and apply_lm_head:
            result["logits"] = self.model.lm_head(self.body.norm(hidden_states))
        else:
            result["hidden_states"] = hidden_states
        return result


__all__ = [
    "GEMMA4_ADAPTER_SCHEMA_VERSION",
    "GEMMA4_LAYER_PREFIX",
    "GEMMA4_MODEL_TYPE",
    "Gemma4AdapterError",
    "Gemma4PipelineAdapter",
    "load_gemma4_text_layer_assignment",
    "select_gemma4_assignment_keys",
    "validate_gemma4_assignment",
]
