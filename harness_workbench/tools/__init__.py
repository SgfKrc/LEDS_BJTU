"""Small, policy-first harness tools."""

from .network import (
    FetchResult,
    NetworkClient,
    NetworkPolicy,
    NetworkResponse,
    NetworkToolError,
    SearchResult,
    TOOL_RESULT_SCHEMA,
    UrllibTransport,
    validate_proxy,
)
from .remote import (
    DEFAULT_ENDPOINT,
    QLHToolAdapter,
    QLHToolAdapterConfig,
    QLHToolTransport,
    REMOTE_RESULT_SCHEMA,
    RemoteToolError,
    TOOL_REQUEST_SCHEMA,
)
from .context import TOOL_CONTEXT_SCHEMA, ToolContextBuilder, ToolContextError, ToolContextPolicy, build_tool_result_context
from .judge_policy import (
    JUDGE_POLICY_SCHEMA,
    JudgeDecision,
    JudgePolicy,
    JudgePolicyDiff,
    JudgePolicyReport,
    JudgeRubric,
    JudgeRubricEntry,
    V1_POLICY,
    V2_POLICY,
    builtin_judge_policy_fixture,
    load_judge_rubric,
    run_judge_policy_diff,
)
from .manifest_health import (
    MANIFEST_HEALTH_SCHEMA,
    MODEL_SUFFIXES,
    ManifestHealthReport,
    build_manifest_health_report,
    scan_manifest_health,
)

__all__ = [
    "FetchResult",
    "NetworkClient",
    "NetworkPolicy",
    "NetworkResponse",
    "NetworkToolError",
    "SearchResult",
    "TOOL_RESULT_SCHEMA",
    "UrllibTransport",
    "validate_proxy",
    "DEFAULT_ENDPOINT",
    "QLHToolAdapter",
    "QLHToolAdapterConfig",
    "QLHToolTransport",
    "REMOTE_RESULT_SCHEMA",
    "RemoteToolError",
    "TOOL_REQUEST_SCHEMA",
    "TOOL_CONTEXT_SCHEMA",
    "ToolContextBuilder",
    "ToolContextError",
    "ToolContextPolicy",
    "build_tool_result_context",
    "JUDGE_POLICY_SCHEMA",
    "JudgeDecision",
    "JudgePolicy",
    "JudgePolicyDiff",
    "JudgePolicyReport",
    "JudgeRubric",
    "JudgeRubricEntry",
    "V1_POLICY",
    "V2_POLICY",
    "builtin_judge_policy_fixture",
    "load_judge_rubric",
    "run_judge_policy_diff",
    "MANIFEST_HEALTH_SCHEMA",
    "MODEL_SUFFIXES",
    "ManifestHealthReport",
    "build_manifest_health_report",
    "scan_manifest_health",
    "MODEL_CARD_INPUT_SCHEMA",
    "MODEL_CARD_SCHEMA",
    "ModelCardArtifact",
    "ModelCardMetadata",
    "ModelCardReport",
    "build_model_card_from_health",
    "build_model_card_from_json",
    "DOWNLOAD_MANIFEST_SCHEMA",
    "DOWNLOAD_RUNNER_SCHEMA",
    "DownloadError",
    "DownloadFilePin",
    "DownloadFileResult",
    "DownloadManifest",
    "DownloadResponse",
    "DownloadRunReport",
    "DownloadRunner",
    "MemoryDownloadTransport",
    "UrllibDownloadTransport",
    "load_download_manifest",
    "CTX_RESSURE_SCHEMA",
    "ContextPressureReport",
    "PressureCheck",
    "build_context_pressure_report",
    "run_context_pressure",
    "RED_TEAM_LAB_SCHEMA",
    "RedTeamLabDecision",
    "RedTeamLabReport",
    "build_red_team_lab_report",
    "run_red_team_lab",
    "TRACE_EVENT_KINDS",
    "TRACE_INPUT_SCHEMA",
    "TRACE_REPLAY_SCHEMA",
    "TraceEvent",
    "TraceReplayReport",
    "TraceScenario",
    "build_trace_replay_report",
    "builtin_trace_scenarios",
    "load_trace_events",
    "run_trace_replay",
    "PROMPT_LAB_INPUT_SCHEMA",
    "PROMPT_LAB_SCHEMA",
    "PromptCaseDelta",
    "PromptLabCase",
    "PromptLabReport",
    "PromptProfileDiff",
    "PromptRenderResult",
    "build_prompt_lab_report",
    "builtin_prompt_lab_cases",
    "load_prompt_cases",
    "load_prompt_profiles",
    "run_prompt_lab",
    "BENCHMARK_LEDGER_INPUT_SCHEMA",
    "BENCHMARK_LEDGER_SCHEMA",
    "BenchmarkLedgerReport",
    "BenchmarkRecord",
    "LedgerGroup",
    "build_benchmark_ledger_report",
    "load_benchmark_payload",
    "run_benchmark_ledger",
    "APIResponse",
    "APITransport",
    "APIProbeResult",
    "APIWorkbenchError",
    "APIWorkbenchReport",
    "API_WORKBENCH_INPUT_SCHEMA",
    "API_WORKBENCH_SCHEMA",
    "MemoryAPITransport",
    "ProbeCase",
    "UrllibAPITransport",
    "builtin_api_cases",
    "run_api_workbench",
    "FUN_CLI_SCHEMA",
    "FUN_QUOTES_INPUT_SCHEMA",
    "FunCliError",
    "QuoteCard",
    "QuoteReport",
    "SayReport",
    "build_quote_report",
    "build_say_report",
    "builtin_quote_cards",
    "load_quote_cards",
    "render_banner",
    "render_progress",
    "render_quote_cards",
    "run_fun_cli",
]


def __getattr__(name: str):
    """Lazy-load the CLI module so ``python -m`` has no pre-import warning."""

    if name in {
        "DOWNLOAD_MANIFEST_SCHEMA",
        "DOWNLOAD_RUNNER_SCHEMA",
        "DownloadError",
        "DownloadFilePin",
        "DownloadFileResult",
        "DownloadManifest",
        "DownloadResponse",
        "DownloadRunReport",
        "DownloadRunner",
        "MemoryDownloadTransport",
        "UrllibDownloadTransport",
        "load_download_manifest",
    }:
        from . import download_runner

        return getattr(download_runner, name)
    if name in {
        "MODEL_CARD_INPUT_SCHEMA",
        "MODEL_CARD_SCHEMA",
        "ModelCardArtifact",
        "ModelCardMetadata",
        "ModelCardReport",
        "build_model_card_from_health",
        "build_model_card_from_json",
    }:
        from . import model_card

        return getattr(model_card, name)
    if name in {
        "CTX_RESSURE_SCHEMA",
        "ContextPressureReport",
        "PressureCheck",
        "build_context_pressure_report",
        "run_context_pressure",
    }:
        from . import ctx_ressure

        return getattr(ctx_ressure, name)
    if name in {
        "RED_TEAM_LAB_SCHEMA",
        "RedTeamLabDecision",
        "RedTeamLabReport",
        "build_red_team_lab_report",
        "run_red_team_lab",
    }:
        from . import red_team_lab

        return getattr(red_team_lab, name)
    if name in {
        "TRACE_EVENT_KINDS",
        "TRACE_INPUT_SCHEMA",
        "TRACE_REPLAY_SCHEMA",
        "TraceEvent",
        "TraceReplayReport",
        "TraceScenario",
        "build_trace_replay_report",
        "builtin_trace_scenarios",
        "load_trace_events",
        "run_trace_replay",
    }:
        from . import trace_replay

        return getattr(trace_replay, name)
    if name in {
        "PROMPT_LAB_INPUT_SCHEMA",
        "PROMPT_LAB_SCHEMA",
        "PromptCaseDelta",
        "PromptLabCase",
        "PromptLabReport",
        "PromptProfileDiff",
        "PromptRenderResult",
        "build_prompt_lab_report",
        "builtin_prompt_lab_cases",
        "load_prompt_cases",
        "load_prompt_profiles",
        "run_prompt_lab",
    }:
        from . import prompt_lab

        return getattr(prompt_lab, name)
    if name in {
        "BENCHMARK_LEDGER_INPUT_SCHEMA",
        "BENCHMARK_LEDGER_SCHEMA",
        "BenchmarkLedgerReport",
        "BenchmarkRecord",
        "LedgerGroup",
        "build_benchmark_ledger_report",
        "load_benchmark_payload",
        "run_benchmark_ledger",
    }:
        from . import benchmark_ledger

        return getattr(benchmark_ledger, name)
    if name in {
        "APIResponse",
        "APITransport",
        "APIProbeResult",
        "APIWorkbenchError",
        "APIWorkbenchReport",
        "API_WORKBENCH_INPUT_SCHEMA",
        "API_WORKBENCH_SCHEMA",
        "MemoryAPITransport",
        "ProbeCase",
        "UrllibAPITransport",
        "builtin_api_cases",
        "run_api_workbench",
    }:
        from . import api_workbench

        return getattr(api_workbench, name)
    if name in {
        "FUN_CLI_SCHEMA",
        "FUN_QUOTES_INPUT_SCHEMA",
        "FunCliError",
        "QuoteCard",
        "QuoteReport",
        "SayReport",
        "build_quote_report",
        "build_say_report",
        "builtin_quote_cards",
        "load_quote_cards",
        "render_banner",
        "render_progress",
        "render_quote_cards",
        "run_fun_cli",
    }:
        from . import fun_cli

        return getattr(fun_cli, name)
    raise AttributeError(name)
