from __future__ import annotations

import types

import pytest
import torch

from scripts.model_tools.qwen3_pipeline_adapter import Qwen3AdapterError
from scripts.model_tools.qwen3_pipeline_chain import (
    build_hidden_handoff,
    build_kv_contract,
    execute_segment_chain,
    validate_hidden_handoff,
    validate_kv_contract,
    validate_segment_plan,
)
from scripts.model_tools.qwen3_pipeline_chain_smoke import run_qwen3_pipeline_chain_smoke


class _Block(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **_kwargs):
        value = hidden_states * (1 + self.scale)
        present = None
        if use_cache:
            current = value.unsqueeze(2)
            if past_key_value is not None:
                current = torch.cat((past_key_value[0], current), dim=1)
            present = (current, current)
        return value, present


class _Model(torch.nn.Module):
    def __init__(self, start: int, end: int, embedding: bool, head: bool):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4) if embedding else None
        self.model.layers = torch.nn.ModuleList([_Block((i + 1) / 10) for i in range(start, end)])
        self.model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False) if head else None
        self.config = types.SimpleNamespace(model_type="qwen3", num_hidden_layers=4)


class _Adapter:
    def __init__(self, start: int, end: int, embedding: bool, head: bool):
        self.model = _Model(start, end, embedding, head)
        from scripts.model_tools.qwen3_pipeline_adapter import Qwen3PipelineAdapter

        self.adapter = Qwen3PipelineAdapter(
            self.model, start_layer=start, end_layer=end,
            has_embedding=embedding, has_lm_head=head, total_layers=4,
        )

    def forward(self, **kwargs):
        return self.adapter.forward(**kwargs)

    def _device_dtype(self, model):
        return self.adapter._device_dtype(model)


def test_segment_plan_is_contiguous_and_component_owned():
    plan = validate_segment_plan([
        {"layer_range": [0, 2], "has_embedding": True, "has_lm_head": False},
        {"layer_range": [2, 4], "has_embedding": False, "has_lm_head": True},
    ], total_layers=4)
    assert plan[0]["segment_index"] == 0
    with pytest.raises(Qwen3AdapterError, match="contiguous"):
        validate_segment_plan([
            {"layer_range": [0, 1], "has_embedding": True},
            {"layer_range": [2, 4], "has_lm_head": True},
        ], total_layers=4)
    with pytest.raises(Qwen3AdapterError, match="only the first"):
        validate_segment_plan([
            {"layer_range": [0, 2], "has_embedding": False},
            {"layer_range": [2, 4], "has_embedding": True, "has_lm_head": True},
        ], total_layers=4)


def test_hidden_and_kv_contracts_fail_closed_on_boundary_change():
    hidden = torch.zeros(1, 3, 4)
    handoff = build_hidden_handoff(hidden, chain_id="c", from_segment=0, to_segment=1)
    validate_hidden_handoff(hidden, handoff, chain_id="c", expected_from=0, expected_to=1)
    handoff["sequence_length"] = 2
    with pytest.raises(Qwen3AdapterError, match="sequence_length"):
        validate_hidden_handoff(hidden, handoff, chain_id="c", expected_from=0, expected_to=1)
    contract = build_kv_contract(
        chain_id="c", segment_index=0, layer_range=[0, 2], sequence_length=3,
        batch_size=1, dtype="torch.float32", device="cpu", phase="prefill", generation=0,
    )
    validate_kv_contract(contract, chain_id="c", segment_index=0, layer_range=[0, 2], sequence_length=3,
                         batch_size=1, dtype="torch.float32", device="cpu", phase="prefill", generation=0)
    contract["generation"] = 1
    with pytest.raises(Qwen3AdapterError, match="KV cache contract"):
        validate_kv_contract(contract, chain_id="c", segment_index=0, layer_range=[0, 2], sequence_length=3,
                             batch_size=1, dtype="torch.float32", device="cpu", phase="prefill", generation=0)


def test_two_segment_chain_hands_off_hidden_and_tracks_each_kv():
    first = _Adapter(0, 2, True, False)
    last = _Adapter(2, 4, False, True)
    segments = [
        {"layer_range": [0, 2], "has_embedding": True, "has_lm_head": False},
        {"layer_range": [2, 4], "has_embedding": False, "has_lm_head": True},
    ]
    result = execute_segment_chain(
        [first, last],
        input_ids=torch.tensor([[1, 2, 3]]),
        segments=segments,
        decode_input_ids=torch.tensor([[3]]),
    )
    assert result["full_model_materialized"] is False
    assert len(result["hidden_handoffs"]) == 1
    assert len(result["kv_contracts"]["prefill"]) == 2
    assert len(result["kv_contracts"]["decode"]) == 2
    assert result["kv_contracts"]["decode"][0]["sequence_length"] == 4
    assert result["decode"]["logits"].shape == (1, 1, 8)


def test_chain_controller_forwards_structured_result(tmp_path):
    report = run_qwen3_pipeline_chain_smoke(
        model=tmp_path,
        segments=[
            {"layer_range": [0, 2], "has_embedding": True},
            {"layer_range": [2, 4], "has_lm_head": True},
        ],
        worker_runner=lambda request, timeout: {
            "schema_version": 1, "tool": "qwen3_pipeline_chain_smoke",
            "operation": "qwen3_pipeline_chain_smoke", "valid": True,
            "gate_passed": False, "status": "resource_rejected", "errors": [],
        },
    )
    assert report["status"] == "resource_rejected"

