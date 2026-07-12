#!/usr/bin/env python3
"""
验证 P0 问题修复的测试脚本

测试 C1 和 C2 的修复是否正确
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.simulation.framework import TestOrchestrator, TestConfig


async def test_c1_fix():
    """测试 C1: test_degradation.py 索引错误修复"""
    print("=" * 70)
    print("测试 C1: test_degradation.py 索引错误修复")
    print("=" * 70)

    # 创建测试编排器
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=3,
        master_api_port=8000,
        master_tcp_port=8888,
        slave_api_port_start=8001,
    )

    try:
        # 设置测试环境
        await orchestrator.setup(config)
        print(f"\n[OK] 启动了 {config.slave_count} 个从节点")

        # 测试停止从节点 0
        print("\n测试停止从节点 0...")
        await orchestrator.stop_slave(0)
        print(f"[OK] 停止了从节点 0，剩余 {len(orchestrator.backend_manager.slave_processes)} 个从节点")

        # 测试停止剩余从节点（C1 修复的关键部分）
        print("\n测试停止剩余从节点（C1 修复）...")
        initial_slave_count = config.slave_count
        remaining_count = initial_slave_count - 1
        for i in range(remaining_count):
            print(f"  停止从节点 {i}...")
            await orchestrator.stop_slave(i)
            print(f"  [OK] 停止了从节点 {i}，剩余 {len(orchestrator.backend_manager.slave_processes)} 个从节点")

        print(f"\n[OK] C1 修复验证通过：成功停止了所有从节点")

    finally:
        # 清理测试环境
        await orchestrator.teardown()
        print("\n[OK] 测试环境已清理")


async def test_c2_fix():
    """测试 C2: test_exceptions.py 索引错误修复"""
    print("\n" + "=" * 70)
    print("测试 C2: test_exceptions.py 索引错误修复")
    print("=" * 70)

    # 创建测试编排器
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=3,
        master_api_port=8000,
        master_tcp_port=8888,
        slave_api_port_start=8001,
    )

    try:
        # 设置测试环境
        await orchestrator.setup(config)
        print(f"\n[OK] 启动了 {config.slave_count} 个从节点")

        # 测试停止部分从节点
        print("\n测试停止部分从节点...")
        interruption_indices = list(range(min(2, config.slave_count)))
        for idx in interruption_indices:
            print(f"  停止从节点 {idx}...")
            await orchestrator.stop_slave(idx)
            print(f"  [OK] 停止了从节点 {idx}，剩余 {len(orchestrator.backend_manager.slave_processes)} 个从节点")

        # 测试重启从节点（C2 修复的关键部分）
        print("\n测试重启从节点（C2 修复）...")
        for i, idx in enumerate(interruption_indices):
            # 停止后，原来的索引 idx 现在变成了 idx - i
            actual_idx = idx - i
            print(f"  重启从节点 {idx} (实际索引 {actual_idx})...")
            await orchestrator.restart_slave(actual_idx)
            print(f"  [OK] 重启了从节点 {idx} (实际索引 {actual_idx})，当前 {len(orchestrator.backend_manager.slave_processes)} 个从节点")

        print(f"\n[OK] C2 修复验证通过：成功重启了所有从节点")

    finally:
        # 清理测试环境
        await orchestrator.teardown()
        print("\n[OK] 测试环境已清理")


async def main():
    """主函数"""
    print("开始验证 P0 问题修复\n")

    try:
        # 测试 C1 修复
        await test_c1_fix()

        # 测试 C2 修复
        await test_c2_fix()

        print("\n" + "=" * 70)
        print("[OK] 所有 P0 问题修复验证通过")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAIL] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
