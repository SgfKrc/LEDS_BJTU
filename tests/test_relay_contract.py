"""Contract tests for the L -> L relay handoff (CORE-RELAY-01).

The acceptance cases here are the frozen form of the experiment in
``docs/跨框架层接力重启评估-2026-09-15.md`` §7.8-§7.11.
"""

from src.relay_contract import (
    RELAY_ACCEPTANCE,
    RelayModelIdentity,
    RelayHiddenSpec,
    RelayTrimPlan,
    build_relay_handoff,
    judge_relay_generation,
    relay_fallback,
)


def _identity(**overrides) -> RelayModelIdentity:
    base = dict(
        model_sha256="a" * 64,
        architecture="qwen35",
        block_count=25,
        n_embd=2048,
        nextn_predict_layers=1,
        artifact_sha256="b" * 64,
    )
    base.update(overrides)
    return RelayModelIdentity(**base)


def test_identity_separates_block_count_from_forward_layers():
    identity = _identity()

    assert identity.n_layer == 24
    assert identity.last_handoff_layer == 22
    assert identity.to_dict()["n_layer"] == 24


def test_trim_maps_local_layer_to_source_layer_and_keeps_block_count():
    # K=4: blk.4..blk.24 are kept, block_count 25 -> 21 (experiment §7.9).
    trim = RelayTrimPlan(trim_layers=4, source=_identity())

    assert trim.kept_block_count == 21
    assert trim.local_to_source_layer(0) == 4
    assert trim.local_to_source_layer(19) == 23
    assert trim.is_valid() is True


def test_handoff_admits_matching_engines_and_carries_the_trim_offset():
    decision = build_relay_handoff(
        _identity(), _identity(block_count=21, artifact_sha256="c" * 64), 4
    )

    assert decision.admitted is True
    assert decision.reason == "admitted"
    handoff = decision.handoff
    assert handoff.cut_layer == 4
    assert handoff.trim.trim_layers == 4
    assert handoff.upstream_last_layer == 3
    assert handoff.downstream_first_source_layer == 4
    assert handoff.hidden.n_embd == 2048
    assert handoff.hidden.bytes_per_token == 2048 * 4
    assert handoff.downstream.artifact_sha256 != handoff.upstream.artifact_sha256
    assert handoff.trim.kept_block_count == handoff.downstream.block_count


def test_handoff_admits_highest_legal_cut_but_never_the_last_layer():
    legal = build_relay_handoff(_identity(), _identity(block_count=2), 23)
    illegal = build_relay_handoff(
        _identity(), _identity(block_count=1, nextn_predict_layers=0), 24
    )

    assert legal.admitted is True
    assert legal.handoff.upstream_last_layer == 22  # == last_handoff_layer
    assert illegal.admitted is False
    assert illegal.reason == "cut_layer_out_of_range"


def test_handoff_rejects_zero_cut_and_missing_identity():
    assert build_relay_handoff(_identity(), _identity(), 0).reason == "cut_layer_out_of_range"
    assert build_relay_handoff(_identity(), _identity(), -3).reason == "cut_layer_out_of_range"
    assert build_relay_handoff(None, _identity(), 4).reason == "invalid_identity"


def test_handoff_rejects_identity_architecture_and_shape_mismatch():
    other_model = build_relay_handoff(
        _identity(), _identity(model_sha256="c" * 64, block_count=21), 4
    )
    other_arch = build_relay_handoff(
        _identity(), _identity(architecture="llama", block_count=21), 4
    )
    other_shape = build_relay_handoff(
        _identity(), _identity(n_embd=4096, block_count=21), 4
    )

    assert other_model.reason == "model_identity_mismatch"
    assert other_arch.reason == "architecture_mismatch"
    assert other_shape.reason == "shape_mismatch"


def test_handoff_rejects_cross_engine_until_xframe_ticket():
    pytorch = _identity(engine="pytorch")

    decision = build_relay_handoff(_identity(), pytorch, 4)

    assert decision.admitted is False
    assert decision.reason == "cross_engine_not_admitted"


def test_handoff_rejects_unsupported_hidden_format():
    decision = build_relay_handoff(
        _identity(), _identity(block_count=21), 4, hidden_dtype="bfloat16"
    )

    assert decision.admitted is False
    assert decision.reason == "unsupported_hidden_format"


def test_handoff_rejects_a_downstream_artifact_that_is_not_the_declared_cut():
    decision = build_relay_handoff(_identity(), _identity(block_count=20), 4)

    assert decision.admitted is False
    assert decision.reason == "trim_layout_mismatch"


def test_handoff_rejects_a_different_mtp_layout():
    decision = build_relay_handoff(
        _identity(), _identity(block_count=21, nextn_predict_layers=0), 4
    )

    assert decision.admitted is False
    assert decision.reason == "trim_layout_mismatch"


def test_hidden_spec_reports_wire_size():
    assert RelayHiddenSpec(n_embd=2048).bytes_per_token == 8192
    assert RelayHiddenSpec(n_embd=2048, dtype="float16").bytes_per_token == 4096
    assert RelayHiddenSpec(n_embd=2048, dtype="int8").supported is False


def test_argmax_is_the_only_criterion_even_when_cosine_is_low():
    # §7.11.2: the embd path is not bit-reproducible inside one process. A relay run that
    # matches on argmax must still pass, otherwise the contract would reject healthy runs.
    verdict = judge_relay_generation(
        [11751, 13, 198, 32],
        [11751, 13, 198, 32],
        cosine=0.994,
        bitwise_equal=False,
    )

    assert verdict.accepted is True
    assert verdict.reason == "all_tokens_match"
    assert verdict.criterion == RELAY_ACCEPTANCE == "per_token_argmax"
    assert verdict.cosine == 0.994  # recorded as evidence, never used as a gate
    assert verdict.bitwise_equal is False
    assert verdict.to_dict()["matched_steps"] == 4


def test_divergence_is_reported_with_the_first_mismatch_step():
    verdict = judge_relay_generation([11751, 13, 198], [11751, 13, 999])

    assert verdict.accepted is False
    assert verdict.reason == "token_mismatch"
    assert verdict.matched_steps == 2
    assert verdict.total_steps == 3
    assert verdict.diagnostics == "first_divergence_step=2"


def test_sequence_length_mismatch_and_empty_inputs_are_rejected():
    length = judge_relay_generation([1, 2, 3], [1, 2])
    empty = judge_relay_generation([], [])

    assert length.accepted is False
    assert length.reason == "length_mismatch"
    assert length.total_steps == 2
    assert empty.accepted is False
    assert empty.reason == "empty_sequence"


def test_fallback_always_names_the_single_process_strategy():
    fallback = relay_fallback("cut_layer_out_of_range")

    assert fallback.engaged is True
    assert fallback.reason == "cut_layer_out_of_range"
    assert fallback.strategy == "single_process_llama_cpp"
    assert fallback.to_dict()["strategy"] == "single_process_llama_cpp"
