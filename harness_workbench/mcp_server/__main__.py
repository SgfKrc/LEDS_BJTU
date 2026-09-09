"""Run the harness MCP server over stdio: ``python -m harness_workbench.mcp_server``."""

from . import StdioMCPTransport, create_harness_server


def main() -> None:
    StdioMCPTransport(create_harness_server()).run()


if __name__ == "__main__":
    main()
