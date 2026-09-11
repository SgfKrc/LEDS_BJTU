import pytest

from harness_workbench.research import (
    CONTEXT_MEASURE_SCHEMA,
    STRATEGIES,
    build_context_measure_fixture,
    run_context_measure,
)


def test_context_measure_fixture_is_exactly_thirty_rounds_and_stable():
    first = build_context_measure_fixture()
    second = build_context_measure_fixture()

    assert first.rounds == 30
    assert len(first.messages) == 60
    assert len(first.early_facts) == 3
    assert first.digest == second.digest
    assert first.as_dict()["fixture_digest"] == first.digest


def test_context_measure_runs_three_strategies_without_model_or_network():
    report = run_context_measure(budgets=(64, 96, 128))

    assert report.schema == CONTEXT_MEASURE_SCHEMA
    assert len(report.observations) == 9
    assert {item.strategy for item in report.observations} == set(STRATEGIES)
    assert all(item.runner_kind == "fixture" for item in report.observations)
    assert all(not item.network_used and not item.weights_loaded for item in report.observations)
    assert all(item.input_tokens <= item.input_budget for item in report.observations)


def test_context_measure_replay_is_digest_stable_and_exposes_chart_series():
    first = run_context_measure(budgets=(64, 96, 128))
    second = run_context_measure(budgets=(64, 96, 128))

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    assert tuple(point["input_budget"] for point in first.series()["state"]) == (64, 96, 128)
    assert "early_fact_recall_rate" in first.series()["memory"][0]
    assert "| `window` |" in first.to_markdown()


def test_context_measure_curves_show_distinct_window_state_and_memory_behaviour():
    report = run_context_measure(budgets=(64, 96, 128, 192))
    rows = {(item.strategy, item.input_budget): item for item in report.observations}

    assert rows["window", 128].early_facts_recalled == 0
    assert rows["state", 128].early_facts_recalled == 3
    assert rows["memory", 192].early_facts_recalled == 3
    assert rows["memory", 192].memory_entries_written == 3
    assert rows["memory", 192].memory_entries_recalled == 3
    assert rows["state", 128].state_tokens > 0
    assert rows["memory", 192].memory_tokens > 0


def test_context_measure_rejects_unsorted_duplicate_or_mismatched_inputs():
    with pytest.raises(ValueError):
        run_context_measure(budgets=(96, 64))
    with pytest.raises(ValueError):
        run_context_measure(budgets=(64, 64))
    with pytest.raises(ValueError):
        run_context_measure(seed=18, fixture=build_context_measure_fixture(seed=17), budgets=(64,))
