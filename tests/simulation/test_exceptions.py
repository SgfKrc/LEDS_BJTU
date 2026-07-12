"""
异常测试

测试系统在各种异常场景下的恢复能力，包括网络中断、节点崩溃等。
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import random

from .framework import (
    TestOrchestrator,
    TestConfig,
    TestResult,
    RequestSender,
    HumanSimulator,
    ResponseValidator,
)
from .scenarios import SCENARIOS, Scenario


async def simulate_network_interruption(
    orchestrator: TestOrchestrator,
    slave_indices: List[int],
    duration: float = 5.0,
) -> List[Tuple[int, int]]:
    """模拟网络中断（停止指定从节点）

    Args:
        orchestrator: 测试编排器
        slave_indices: 要停止的从节点索引列表
        duration: 中断持续时间（秒）
    """
    print(f"\n模拟网络中断: 停止从节点 {slave_indices}, 持续 {duration}s")
    stopped: List[Tuple[int, int]] = []

    # 停止指定从节点
    for idx in sorted(slave_indices, reverse=True):
        try:
            port = await orchestrator.stop_slave(idx)
            stopped.append((idx, port))
            print(f"  [OK] 从节点 {idx} 已停止")
        except Exception as e:
            print(f"  [FAIL] 停止从节点 {idx} 失败: {e}")

    # 等待指定时间
    print(f"  等待 {duration}s...")
    await asyncio.sleep(duration)
    return stopped


async def simulate_node_crash(
    orchestrator: TestOrchestrator,
    slave_indices: List[int],
) -> None:
    """模拟节点崩溃（强制停止并立即重启）

    Args:
        orchestrator: 测试编排器
        slave_indices: 要重启的从节点索引列表
    """
    print(f"\n模拟节点崩溃: 重启从节点 {slave_indices}")
    stopped: List[Tuple[int, int]] = []

    # 停止从节点
    for idx in sorted(slave_indices, reverse=True):
        try:
            port = await orchestrator.stop_slave(idx)
            stopped.append((idx, port))
            print(f"  [OK] 从节点 {idx} 已停止")
        except Exception as e:
            print(f"  [FAIL] 停止从节点 {idx} 失败: {e}")

    # 等待一小段时间（模拟崩溃）
    print(f"  等待 2s (模拟崩溃)...")
    await asyncio.sleep(2)

    # 重启从节点
    for idx, port in stopped:
        try:
            await orchestrator.restart_slave(idx, slave_api_port=port)
            print(f"  [OK] 从节点 {idx} 已重启 (API:{port})")
        except Exception as e:
            print(f"  [FAIL] 重启从节点 {idx} 失败: {e}")

    # 等待集群稳定
    print(f"  等待 5s (等待集群稳定)...")
    await asyncio.sleep(5)


async def execute_inference_during_exception(
    sender: RequestSender,
    question: str,
    session_id: str = "exception-test",
    timeout: float = 30.0,
) -> dict:
    """在异常场景下执行推理请求

    Args:
        sender: 请求发送器
        question: 问题文本
        session_id: 会话 ID
        timeout: 超时时间（秒）

    Returns:
        推理结果
    """
    try:
        response_text = ""
        start_time = time.time()

        # 设置超时
        async def send_with_timeout():
            nonlocal response_text
            async for chunk in sender.send_chat_request(question, session_id, stream=True):
                if chunk["type"] == "chunk":
                    content = chunk.get("content", "")
                    response_text += content
                    print(content, end="", flush=True)

        await asyncio.wait_for(send_with_timeout(), timeout=timeout)

        latency = time.time() - start_time

        # 验证响应
        validation = ResponseValidator.validate_response(
            response=response_text,
            question=question,
            expected_length=(50, 500),
        )

        return {
            "status": "PASSED" if validation.is_valid else "FAILED",
            "response": response_text,
            "latency": latency,
            "response_length": len(response_text),
            "validation": validation,
            "timeout": False,
        }

    except asyncio.TimeoutError:
        validation = ResponseValidator.validate_response("", question, expected_length=(50, 500))
        validation.errors.append(f"请求超时 ({timeout}s)")
        return {
            "status": "TIMEOUT",
            "error": f"请求超时 ({timeout}s)",
            "response": "",
            "latency": timeout,
            "response_length": 0,
            "validation": validation,
            "timeout": True,
        }

    except Exception as e:
        validation = ResponseValidator.validate_response("", question, expected_length=(50, 500))
        validation.errors.append(str(e))
        return {
            "status": "ERROR",
            "error": str(e),
            "response": "",
            "latency": 0,
            "response_length": 0,
            "validation": validation,
            "timeout": False,
        }


async def test_network_interruption(
    scenario_name: str = "simple_qa",
    slave_count: int = 3,
    interruption_duration: float = 5.0,
    verbose: bool = True,
) -> List[TestResult]:
    """网络中断测试

    测试流程:
    1. 启动主节点 + 多个从节点
    2. 执行基线推理测试
    3. 模拟网络中断（停止部分从节点）
    4. 在中断期间执行推理测试
    5. 恢复网络（重启从节点）
    6. 执行恢复后推理测试

    Args:
        scenario_name: 测试场景名称
        slave_count: 从节点数量
        interruption_duration: 中断持续时间（秒）
        verbose: 是否打印详细信息

    Returns:
        测试结果列表
    """
    print("\n" + "=" * 70)
    print(f"网络中断测试 (从节点: {slave_count}, 中断持续: {interruption_duration}s)")
    print("=" * 70)

    # 获取测试场景
    if scenario_name not in SCENARIOS:
        print(f"错误: 场景 '{scenario_name}' 不存在")
        return []

    scenario = SCENARIOS[scenario_name]
    print(f"\n测试场景: {scenario.name}")
    print(f"描述: {scenario.description}")

    # 选择前3个问题进行测试
    test_questions = scenario.questions[:3]
    print(f"问题数: {len(test_questions)}")

    # 1. 启动后端
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=slave_count,
    )

    all_results = []

    try:
        await orchestrator.setup(config)

        async with RequestSender() as sender:
            # 2. 基线测试
            print("\n" + "-" * 70)
            print("阶段 1: 基线测试")
            print("-" * 70)

            baseline_question = test_questions[0]
            print(f"问题: {baseline_question}")

            baseline_result = await execute_inference_during_exception(
                sender=sender,
                question=baseline_question,
                session_id="baseline",
            )

            print(f"\n\n响应 (长度: {baseline_result['response_length']}):")
            print(f"  {baseline_result['response'][:200]}...")
            print(f"\n延迟: {baseline_result['latency']:.2f}s")

            test_result = TestResult(
                test_name=f"network_interruption_baseline",
                timestamp=datetime.now().isoformat(),
                status=baseline_result["status"],
                question=baseline_question,
                response=baseline_result["response"],
                latency=baseline_result["latency"],
                response_length=baseline_result["response_length"],
                validation=baseline_result["validation"],
                metrics={
                    "phase": "baseline",
                    "timeout": baseline_result.get("timeout", False),
                },
            )

            all_results.append(test_result)

            # 3. 模拟网络中断
            print("\n" + "-" * 70)
            print(f"阶段 2: 网络中断测试 (中断持续 {interruption_duration}s)")
            print("-" * 70)

            # 停止部分从节点（模拟网络中断）
            interruption_indices = list(range(min(2, slave_count)))
            stopped_slaves = await simulate_network_interruption(
                orchestrator=orchestrator,
                slave_indices=interruption_indices,
                duration=interruption_duration,
            )

            # 在中断期间执行推理
            interruption_question = test_questions[1]
            print(f"\n在中断期间执行推理...")
            print(f"问题: {interruption_question}")

            interruption_result = await execute_inference_during_exception(
                sender=sender,
                question=interruption_question,
                session_id="during-interruption",
                timeout=30.0,
            )

            print(f"\n\n响应 (长度: {interruption_result['response_length']}):")
            print(f"  {interruption_result['response'][:200]}...")
            print(f"\n延迟: {interruption_result['latency']:.2f}s")
            print(f"状态: {interruption_result['status']}")

            test_result = TestResult(
                test_name=f"network_interruption_during",
                timestamp=datetime.now().isoformat(),
                status=interruption_result["status"],
                question=interruption_question,
                response=interruption_result["response"],
                latency=interruption_result["latency"],
                response_length=interruption_result["response_length"],
                validation=interruption_result["validation"],
                metrics={
                    "phase": "during_interruption",
                    "timeout": interruption_result.get("timeout", False),
                },
            )

            all_results.append(test_result)

            # 4. 恢复网络
            print("\n" + "-" * 70)
            print("阶段 3: 恢复后测试")
            print("-" * 70)

            # 重启从节点，使用停止时记录的原 API 端口
            for idx, port in stopped_slaves:
                actual_idx = min(idx, len(orchestrator.backend_manager.slave_processes))
                try:
                    await orchestrator.restart_slave(actual_idx, slave_api_port=port)
                    print(f"  [OK] 从节点 {idx} (API:{port}) 已重启")
                except Exception as e:
                    print(f"  [FAIL] 重启从节点 {idx} 失败: {e}")

            # 等待集群稳定
            print("  等待 5s (等待集群稳定)...")
            await asyncio.sleep(5)

            # 执行恢复后测试
            recovery_question = test_questions[2]
            print(f"\n执行恢复后推理...")
            print(f"问题: {recovery_question}")

            recovery_result = await execute_inference_during_exception(
                sender=sender,
                question=recovery_question,
                session_id="after-recovery",
            )

            print(f"\n\n响应 (长度: {recovery_result['response_length']}):")
            print(f"  {recovery_result['response'][:200]}...")
            print(f"\n延迟: {recovery_result['latency']:.2f}s")

            test_result = TestResult(
                test_name=f"network_interruption_recovery",
                timestamp=datetime.now().isoformat(),
                status=recovery_result["status"],
                question=recovery_question,
                response=recovery_result["response"],
                latency=recovery_result["latency"],
                response_length=recovery_result["response_length"],
                validation=recovery_result["validation"],
                metrics={
                    "phase": "after_recovery",
                    "timeout": recovery_result.get("timeout", False),
                },
            )

            all_results.append(test_result)

            # 5. 生成报告
            print("\n" + "=" * 70)
            print("网络中断测试报告")
            print("=" * 70)

            report = generate_exception_report(
                all_results, scenario, "network_interruption", slave_count
            )
            print_exception_report(report)

            # 保存报告
            report_path = Path(__file__).parent / "results" / f"network_interruption_{scenario_name}.json"
            save_exception_report(report, report_path)
            print(f"\n[OK] 报告已保存: {report_path}")

            return all_results

    except Exception as e:
        print(f"\n[FAIL] 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        await orchestrator.teardown()


async def test_node_crash(
    scenario_name: str = "simple_qa",
    slave_count: int = 3,
    verbose: bool = True,
) -> List[TestResult]:
    """节点崩溃测试

    测试流程:
    1. 启动主节点 + 多个从节点
    2. 执行基线推理测试
    3. 模拟节点崩溃（停止并立即重启）
    4. 执行恢复后推理测试

    Args:
        scenario_name: 测试场景名称
        slave_count: 从节点数量
        verbose: 是否打印详细信息

    Returns:
        测试结果列表
    """
    print("\n" + "=" * 70)
    print(f"节点崩溃测试 (从节点: {slave_count})")
    print("=" * 70)

    # 获取测试场景
    if scenario_name not in SCENARIOS:
        print(f"错误: 场景 '{scenario_name}' 不存在")
        return []

    scenario = SCENARIOS[scenario_name]
    print(f"\n测试场景: {scenario.name}")
    print(f"描述: {scenario.description}")

    # 选择前2个问题进行测试
    test_questions = scenario.questions[:2]
    print(f"问题数: {len(test_questions)}")

    # 1. 启动后端
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=slave_count,
    )

    all_results = []

    try:
        await orchestrator.setup(config)

        async with RequestSender() as sender:
            # 2. 基线测试
            print("\n" + "-" * 70)
            print("阶段 1: 基线测试")
            print("-" * 70)

            baseline_question = test_questions[0]
            print(f"问题: {baseline_question}")

            baseline_result = await execute_inference_during_exception(
                sender=sender,
                question=baseline_question,
                session_id="baseline",
            )

            print(f"\n\n响应 (长度: {baseline_result['response_length']}):")
            print(f"  {baseline_result['response'][:200]}...")
            print(f"\n延迟: {baseline_result['latency']:.2f}s")

            test_result = TestResult(
                test_name=f"node_crash_baseline",
                timestamp=datetime.now().isoformat(),
                status=baseline_result["status"],
                question=baseline_question,
                response=baseline_result["response"],
                latency=baseline_result["latency"],
                response_length=baseline_result["response_length"],
                validation=baseline_result["validation"],
                metrics={
                    "phase": "baseline",
                },
            )

            all_results.append(test_result)

            # 3. 模拟节点崩溃
            print("\n" + "-" * 70)
            print("阶段 2: 节点崩溃和恢复测试")
            print("-" * 70)

            # 随机选择一个从节点进行崩溃测试
            crash_indices = [random.randint(0, slave_count - 1)]
            await simulate_node_crash(
                orchestrator=orchestrator,
                slave_indices=crash_indices,
            )

            # 执行恢复后测试
            recovery_question = test_questions[1]
            print(f"\n执行恢复后推理...")
            print(f"问题: {recovery_question}")

            recovery_result = await execute_inference_during_exception(
                sender=sender,
                question=recovery_question,
                session_id="after-crash",
            )

            print(f"\n\n响应 (长度: {recovery_result['response_length']}):")
            print(f"  {recovery_result['response'][:200]}...")
            print(f"\n延迟: {recovery_result['latency']:.2f}s")

            test_result = TestResult(
                test_name=f"node_crash_recovery",
                timestamp=datetime.now().isoformat(),
                status=recovery_result["status"],
                question=recovery_question,
                response=recovery_result["response"],
                latency=recovery_result["latency"],
                response_length=recovery_result["response_length"],
                validation=recovery_result["validation"],
                metrics={
                    "phase": "after_crash",
                },
            )

            all_results.append(test_result)

            # 4. 生成报告
            print("\n" + "=" * 70)
            print("节点崩溃测试报告")
            print("=" * 70)

            report = generate_exception_report(
                all_results, scenario, "node_crash", slave_count
            )
            print_exception_report(report)

            # 保存报告
            report_path = Path(__file__).parent / "results" / f"node_crash_{scenario_name}.json"
            save_exception_report(report, report_path)
            print(f"\n[OK] 报告已保存: {report_path}")

            return all_results

    except Exception as e:
        print(f"\n[FAIL] 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        await orchestrator.teardown()


def generate_exception_report(
    results: List[TestResult],
    scenario: Scenario,
    exception_type: str,
    slave_count: int,
) -> dict:
    """生成异常测试报告

    Args:
        results: 测试结果列表
        scenario: 测试场景
        exception_type: 异常类型 ("network_interruption" 或 "node_crash")
        slave_count: 从节点数量

    Returns:
        报告字典
    """
    # 按阶段分类
    baseline_results = [r for r in results if r.metrics.get("phase") == "baseline"]
    during_results = [r for r in results if r.metrics.get("phase") == "during_interruption"]
    recovery_results = [r for r in results if r.metrics.get("phase") in ["after_recovery", "after_crash"]]

    # 统计各阶段
    def calculate_stats(result_list):
        if not result_list:
            return {"count": 0}

        passed = sum(1 for r in result_list if r.status == "PASSED")
        total = len(result_list)
        latencies = [r.latency for r in result_list if r.status == "PASSED"]

        return {
            "count": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        }

    baseline_stats = calculate_stats(baseline_results)
    during_stats = calculate_stats(during_results)
    recovery_stats = calculate_stats(recovery_results)

    # 构建报告
    report = {
        "test_name": f"{exception_type}_{scenario.name}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "exception_type": exception_type,
            "slave_count": slave_count,
        },
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
        },
        "phases": {
            "baseline": {
                "description": "基线测试",
                "stats": baseline_stats,
                "results": [r.to_dict() for r in baseline_results],
            },
        },
        "summary": {
            "total_tests": len(results),
            "total_passed": sum(1 for r in results if r.status == "PASSED"),
            "total_failed": sum(1 for r in results if r.status != "PASSED"),
            "overall_success_rate": sum(1 for r in results if r.status == "PASSED") / len(results) if results else 0,
        },
    }

    if during_results:
        report["phases"]["during_exception"] = {
            "description": "异常期间测试",
            "stats": during_stats,
            "results": [r.to_dict() for r in during_results],
        }

    if recovery_results:
        phase_name = "after_recovery" if exception_type == "network_interruption" else "after_crash"
        report["phases"][phase_name] = {
            "description": "恢复后测试",
            "stats": recovery_stats,
            "results": [r.to_dict() for r in recovery_results],
        }

    return report


def print_exception_report(report: dict) -> None:
    """打印异常测试报告

    Args:
        report: 报告字典
    """
    print(f"\n测试名称: {report['test_name']}")
    print(f"执行时间: {report['timestamp']}")

    print(f"\n配置信息:")
    config = report["config"]
    print(f"  异常类型: {config['exception_type']}")
    print(f"  从节点数: {config['slave_count']}")

    print(f"\n场景信息:")
    scenario = report["scenario"]
    print(f"  名称: {scenario['name']}")
    print(f"  描述: {scenario['description']}")

    print(f"\n各阶段测试结果:")

    for phase_name, phase_data in report["phases"].items():
        print(f"\n  {phase_data['description']}:")
        stats = phase_data["stats"]

        if stats["count"] == 0:
            print(f"    无测试")
            continue

        print(f"    测试数: {stats['count']}")
        print(f"    通过: {stats['passed']}")
        print(f"    失败: {stats['failed']}")
        print(f"    成功率: {stats['success_rate']:.1%}")
        if 'avg_latency' in stats and stats['avg_latency'] > 0:
            print(f"    平均延迟: {stats['avg_latency']:.2f}s")

    print(f"\n整体统计:")
    summary = report["summary"]
    print(f"  总测试数: {summary['total_tests']}")
    print(f"  总通过: {summary['total_passed']}")
    print(f"  总失败: {summary['total_failed']}")
    print(f"  整体成功率: {summary['overall_success_rate']:.1%}")

    # 判断整体状态
    overall_success_rate = summary["overall_success_rate"]

    print(f"\n整体状态:")
    if overall_success_rate >= 0.95:
        print(f"  [PASS] 通过 (整体成功率 {overall_success_rate:.1%} >= 95%)")
    else:
        print(f"  [FAIL] 失败 (整体成功率 {overall_success_rate:.1%} < 95%)")


def save_exception_report(report: dict, path: Path) -> None:
    """保存异常测试报告到文件

    Args:
        report: 报告字典
        path: 文件路径
    """
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================================
# 主函数
# ============================================================================


async def main():
    """主函数"""
    import sys

    # 解析命令行参数
    test_type = "network_interruption"
    scenario_name = "simple_qa"
    slave_count = 3

    if len(sys.argv) > 1:
        test_type = sys.argv[1]

    if len(sys.argv) > 2:
        scenario_name = sys.argv[2]

    if len(sys.argv) > 3:
        try:
            slave_count = int(sys.argv[3])
        except ValueError:
            print(f"警告: 无效的从节点数 '{sys.argv[3]}'，使用默认值 3")
            slave_count = 3

    print(f"测试类型: {test_type}")
    print(f"使用场景: {scenario_name}, 从节点数: {slave_count}")

    # 执行测试
    if test_type == "network_interruption":
        results = await test_network_interruption(
            scenario_name=scenario_name,
            slave_count=slave_count,
            verbose=True,
        )
    elif test_type == "node_crash":
        results = await test_node_crash(
            scenario_name=scenario_name,
            slave_count=slave_count,
            verbose=True,
        )
    else:
        print(f"错误: 未知的测试类型 '{test_type}'")
        print(f"可用类型: network_interruption, node_crash")
        sys.exit(1)

    # 返回退出码
    if results:
        passed = sum(1 for r in results if r.status == "PASSED")
        total = len(results)
        success_rate = passed / total

        if success_rate >= 0.95:
            print(f"\n[SUCCESS] 测试通过 ({success_rate:.1%})")
            sys.exit(0)
        else:
            print(f"\n[FAIL] 测试失败 ({success_rate:.1%})")
            sys.exit(1)
    else:
        print("\n[FAIL] 测试未执行")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
