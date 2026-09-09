# AUD-RT-01 Startup and Harness Follow-up

Date: 2026-09-10

## Scope

This follow-up closes the startup-flow gap found during the multi-agent quality audit:

- A missing local model must not terminate the normal application launcher.
- The model catalog must remain visible before any model is downloaded.
- Presets and download jobs must be usable from the frontend.
- A model selected in Harness must be forwarded to the QLH backend.

## Implemented

- `packaging/launcher.py` uses the non-interactive `ensure_model_or_warn()` check. A missing model is now a recoverable state; the API and model-download workspace can start. `--check-only` remains strict and fails without an installed model.
- `harness_workbench/adapters/qlh.py` exposes the QLH catalog, presets, download jobs, queue-download, and load-model operations. Catalog rows are normalized to the Harness `available` contract.
- `harness_workbench/api_layer/app.py` exposes the corresponding `/v1/model-assets`, `/v1/model-presets`, `/v1/model-downloads`, and `/v1/models/load` routes.
- `harness_workbench/ui_react` shows unavailable assets, presets, active download jobs, load/download actions, and polls active jobs. Chat refuses to send to an unavailable asset and points the user to the runtime view.
- The main frontend can select an unavailable model and queue its matching preset instead of disabling the catalog option. CyberGothic links empty local-asset state and ready download jobs into the model lab.
- `scripts/model_tools/llm_smoke_matrix.py` now rejects a non-zero worker exit even if stale stdout contains parseable JSON.

## Verification

- Python targeted regression: `9 passed` for Harness asset flow, launcher gate, and smoke-worker handling.
- Model/API/download/Harness regression: `111 passed, 1 skipped`.
- Full Python suite: `3396 passed, 20 skipped`.
- Harness UI build: passed.
- CyberGothic UI build: passed.
- Main frontend build: passed.
- `git diff --check`: passed.

## Remaining Work Count

The ticket is not closed as a real-runtime acceptance ticket. Four independently verifiable items remain:

1. Run the pinned llama.cpp and MiniCPM4 runtime smoke with real model assets, including template, thinking, and architecture evidence.
2. Run real model load/switch success, failure, rollback, and quantization-mismatch acceptance cases.
3. Add and execute complete MCP negative-path UI automation.
4. Add and execute the CyberGothic Playwright regression matrix.

No real model runtime result is claimed by this follow-up. The local environment can validate the contracts and fixture paths, but it is not the final runtime acceptance environment.
