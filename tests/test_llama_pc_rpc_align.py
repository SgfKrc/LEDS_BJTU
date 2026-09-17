"""跨路径数值对齐开关（`--align-numerics` / `-nr --no-repack`）的接入测试。

背景：本机 CPU 默认启用 CPU_REPACK，而 RPC 路径的权重落在远端 buffer 里无法 repack，
两条路径 logits 不同（max|Δ|≈0.98，见 local_docs §11/§13）。对齐开关必须同时覆盖
CLI 执行端（`host_command` 加 `--no-repack`）与 Python 引擎端（`use_extra_bufts=False`）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.llama_pc_rpc import build_plan  # noqa: E402


def test_host_command_adds_no_repack_only_when_requested():
    aligned = build_plan(total_layers=25, no_repack=True)
    default = build_plan(total_layers=25)

    assert "--no-repack" in aligned.host_command
    assert "--no-repack" not in default.host_command
    # 其余命令形状不变（只多这一个开关）
    assert [a for a in aligned.host_command if a != "--no-repack"] == default.host_command
    assert aligned.no_repack is True and default.no_repack is False


def test_plan_keeps_rpc_command_shape():
    plan = build_plan(total_layers=25, no_repack=True)
    command = plan.host_command
    assert "--rpc" in command
    assert command[command.index("--device") + 1] == "RPC0"
