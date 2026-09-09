"""Command-line entry point for the standalone harness workbench."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import LlamaServerAdapter, LlamaServerConfig, LlamaServerProcess
from .api_layer import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH small-model harness workbench")
    parser.add_argument("--model", required=True, help="local GGUF or backend model path")
    parser.add_argument("--llama-server", default="llama-server", help="llama-server executable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--mmproj")
    parser.add_argument("--no-jinja", action="store_true")
    parser.add_argument("--no-cache-prompt", action="store_true")
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8090)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LlamaServerConfig(
        executable=args.llama_server,
        model=Path(args.model),
        host=args.host,
        port=args.port,
        context_size=args.ctx_size,
        max_new_tokens=args.max_new_tokens,
        mmproj=args.mmproj,
        enable_jinja=not args.no_jinja,
        cache_prompt=not args.no_cache_prompt,
    )
    process = LlamaServerProcess(config)
    adapter = LlamaServerAdapter(config, process=process)
    adapter.start()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - environment-dependent
        adapter.close()
        raise SystemExit("uvicorn is required to serve the harness API") from exc
    try:
        uvicorn.run(create_app(adapter), host=args.serve_host, port=args.serve_port)
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
