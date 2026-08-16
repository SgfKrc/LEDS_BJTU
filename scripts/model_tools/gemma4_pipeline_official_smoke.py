"""CPU-only official Gemma 4 adapter smoke for the isolated Transformers venv."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


def run_smoke() -> dict[str, Any]:
    import torch
    from transformers import (
        Gemma4UnifiedConfig,
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedTextConfig,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.model_tools.gemma4_pipeline_adapter import Gemma4PipelineAdapter

    text = Gemma4UnifiedTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_global_key_value_heads=2,
        global_head_dim=8,
        num_kv_shared_layers=2,
        sliding_window=16,
        attention_dropout=0.0,
    )
    config = Gemma4UnifiedConfig(
        text_config=text,
        vision_config=None,
        audio_config=None,
    )
    model = Gemma4UnifiedForConditionalGeneration(config).eval()
    first = Gemma4PipelineAdapter(
        model,
        start_layer=0,
        end_layer=2,
        has_embedding=True,
        has_lm_head=False,
    )
    last = Gemma4PipelineAdapter(
        model,
        start_layer=2,
        end_layer=4,
        has_embedding=False,
        has_lm_head=True,
    )
    prefill_ids = torch.tensor([[1, 2, 3, 4]])
    decode_ids = torch.tensor([[5]])
    with torch.no_grad():
        direct_prefill = model(
            input_ids=prefill_ids,
            use_cache=True,
            return_shared_kv_states=True,
        )
        direct_decode = model(
            input_ids=decode_ids,
            use_cache=True,
            past_key_values=direct_prefill.past_key_values,
            shared_kv_states=direct_prefill.shared_kv_states,
            return_shared_kv_states=True,
        )
        first_prefill = first.forward(input_ids=prefill_ids)
        last_prefill = last.forward(
            hidden_states=first_prefill["hidden_states"],
            past_key_values=first_prefill["past_key_values"],
            shared_kv_states=first_prefill["shared_kv_states"],
        )
        first_decode = first.forward(
            input_ids=decode_ids,
            past_key_values=last_prefill["past_key_values"],
            shared_kv_states=last_prefill["shared_kv_states"],
        )
        last_decode = last.forward(
            hidden_states=first_decode["hidden_states"],
            past_key_values=first_decode["past_key_values"],
            shared_kv_states=first_decode["shared_kv_states"],
        )

    prefill_error = float(
        (direct_prefill.logits - last_prefill["logits"]).abs().max()
    )
    decode_error = float(
        (direct_decode.logits - last_decode["logits"]).abs().max()
    )
    if prefill_error > 1e-6 or decode_error > 1e-6:
        raise AssertionError(
            f"Gemma 4 segmented parity failed: prefill={prefill_error} decode={decode_error}"
        )
    return {
        "status": "passed",
        "transformers_version": __import__("transformers").__version__,
        "model_type": config.model_type,
        "prefill_max_abs_error": prefill_error,
        "decode_max_abs_error": decode_error,
        "sequence_length": int(last_decode["sequence_length"]),
        "shared_kv_sequence_lengths": dict(
            last_decode["shared_kv_sequence_lengths"]
        ),
        "full_model_materialized": False,
        "multimodal_materialized": False,
    }


def main() -> int:
    print(json.dumps(run_smoke(), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
