"""
分布式推理测试

测试主从节点协作推理的能力，验证分布式推理的正确性和性能。
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .framework import (
    TestOrchestrator,
    TestConfig,
    TestResult,
    RequestSender,
    HumanSimulator,
    ResponseValidator,
)
from .scenarios import SCENARIOS, Scenario


def summarize_cluster_metrics(metrics_history: List[dict], duration: int) -> dict:
    if not metrics_history:
        return {
            "error": "未收集到任何指标",
            "sample_count": 0,
            "duration": duration,
            "history": [],
        }

    return {
        "sample_count": len(metrics_history),
        "duration": duration,
        "avg_online_slaves": sum(m["online_slaves"] for m in metrics_history) / len(metrics_history),
        "avg_busy_slaves": sum(m["busy_slaves"] for m in metrics_history) / len(metrics_history),
        "max_total_tasks": max(m["total_tasks"] for m in metrics_history),
        "max_completed_tasks": max(m["completed_tasks"] for m in metrics_history),
        "history": metrics_history,
    }


async def monitor_cluster_metrics(
    sender: RequestSender,
    duration: int = 30,
    interval: float = 2.0,
) -> dict:
    """监控集群指标

    Args:
        sender: 请求发送器
        duration: 监控持续时间（秒）
        interval: 采样间隔（秒）

    Returns:
        集群指标统计
    """
    metrics_history = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            try:
                status = await sender.get_cluster_status()

                # 提取关键指标
                metrics = {
                    "timestamp": time.time(),
                    "mode": status.get("mode", "unknown"),
                    "master_state": status.get("master", {}).get("state", "unknown"),
                    "slave_count": len(status.get("slaves", [])),
                    "online_slaves": sum(
                        1 for s in status.get("slaves", [])
                        if s.get("state") == "online"
                    ),
                    "busy_slaves": sum(
                        1 for s in status.get("slaves", [])
                        if s.get("state") == "busy"
                    ),
                    "total_tasks": status.get("total_tasks", 0),
                    "completed_tasks": status.get("completed_tasks", 0),
                }

                metrics_history.append(metrics)

            except Exception as e:
                print(f"  警告: 获取集群状态失败: {e}")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass

    return summarize_cluster_metrics(metrics_history, duration)


async def execute_distributed_inference(
    sender: RequestSender,
    question: str,
    session_id: str = "test-distributed",
    simulator: HumanSimulator = None,
    monitor: bool = True,
) -> dict:
    """执行分布式推理请求

    Args:
        sender: 请求发送器
        question: 问题文本
        session_id: 会话 ID
        simulator: 人机模拟器（可选）
        monitor: 是否监控集群指标

    Returns:
        推理结果字典
    """
    # 如果提供了模拟器，使用模拟输入
    if simulator:
        full_input = await simulator.simulate_input_batch(question)
    else:
        full_input = question

    # 启动监控（后台任务）
    monitor_task = None
    if monitor:
        monitor_task = asyncio.create_task(
            monitor_cluster_metrics(sender, duration=60, interval=2.0)
        )

    # 发送请求
    response_text = ""
    start_time = time.time()

    async for chunk in sender.send_chat_request(full_input, session_id, stream=True):
        if chunk["type"] == "chunk":
            content = chunk.get("content", "")
            response_text += content
            # 实时打印（可选）
            print(content, end="", flush=True)

    latency = time.time() - start_time

    # 停止监控并获取结果
    cluster_metrics = None
    if monitor_task:
        monitor_task.cancel()
        try:
            cluster_metrics = await monitor_task
        except asyncio.CancelledError:
            cluster_metrics = None

    return {
        "question": question,
        "response": response_text,
        "latency": latency,
        "length": len(response_text),
        "cluster_metrics": cluster_metrics,
    }


async def test_distributed_inference(
    scenario_name: str = "simple_qa",
    slave_count: int = 2,
    verbose: bool = True,
) -> List[TestResult]:
    """分布式推理完整测试

    Args:
        scenario_name: 测试场景名称
        slave_count: 从节点数量
        verbose: 是否打印详细信息

    Returns:
        测试结果列表
    """
    print("\n" + "=" * 70)
    print(f"分布式推理测试 (从节点数: {slave_count})")
    print("=" * 70)

    # 获取测试场景
    if scenario_name not in SCENARIOS:
        print(f"错误: 场景 '{scenario_name}' 不存在")
        print(f"可用场景: {', '.join(SCENARIOS.keys())}")
        return []

    scenario = SCENARIOS[scenario_name]
    print(f"\n测试场景: {scenario.name}")
    print(f"描述: {scenario.description}")
    print(f"问题数: {len(scenario.questions)}")
    print(f"预期长度: {scenario.expected_length}")

    # 1. 启动后端（主节点 + 从节点）
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=slave_count,
    )

    try:
        await orchestrator.setup(config)

        # 2. 验证集群状态
        print("\n" + "-" * 70)
        print("验证集群状态")
        print("-" * 70)

        async with RequestSender() as sender:
            # 检查健康状态
            health = await sender.check_health()
            print(f"[OK] 后端健康状态: {health.get('status', 'unknown')}")

            # 检查集群状态
            status = await sender.get_cluster_status()
            mode = status.get("mode", "unknown")
            actual_slave_count = len(status.get("slaves", []))
            online_slaves = sum(
                1 for s in status.get("slaves", [])
                if s.get("state") == "online"
            )

            print(f"[OK] 运行模式: {mode}")
            print(f"[OK] 从节点数: {actual_slave_count} (在线: {online_slaves})")

            if mode != "distributed":
                print(f"[FAIL] 预期分布式模式，实际为 {mode}")
                return []

            if actual_slave_count != slave_count:
                print(f"[WARNING] 预期 {slave_count} 个从节点，实际为 {actual_slave_count}")

            if online_slaves < slave_count:
                print(f"[WARNING] 预期 {slave_count} 个在线从节点，实际为 {online_slaves}")

            # 检查分层配置
            layer_config = await sender.get_layer_config()
            print(f"\n[OK] 分层配置:")
            print(f"  总层数: {layer_config.get('total_layers', 'unknown')}")
            print(f"  分配节点数: {len(layer_config.get('assignments', []))}")

            for assignment in layer_config.get("assignments", []):
                node_id = assignment.get("node_id", "unknown")
                start_layer = assignment.get("start_layer", 0)
                end_layer = assignment.get("end_layer", 0)
                print(f"    {node_id}: layers {start_layer}-{end_layer}")

            # 3. 执行推理测试
            print("\n" + "-" * 70)
            print("开始推理测试")
            print("-" * 70)

            simulator = HumanSimulator(
                typing_speed=5.0,
                pause_probability=0.3,
                pause_duration=(0.5, 2.0),
            )

            results = []

            for i, question in enumerate(scenario.questions, 1):
                print(f"\n[{i}/{len(scenario.questions)}] 测试问题:")
                print(f"  {question}")
                print("\n模拟用户输入...")

                # 执行推理
                result_data = await execute_distributed_inference(
                    sender=sender,
                    question=question,
                    session_id="test-distributed",
                    simulator=simulator,
                    monitor=True,
                )

                print(f"\n\n响应 (长度: {result_data['length']}):")
                print(f"  {result_data['response'][:200]}...")
                print(f"\n延迟: {result_data['latency']:.2f}s")

                # 验证响应
                validation = ResponseValidator.validate_response(
                    response=result_data["response"],
                    question=question,
                    expected_length=scenario.expected_length,
                )

                if verbose:
                    print(f"\n{validation}")

                # 验证分布式推理指标
                distributed_validation = None
                if result_data.get("cluster_metrics"):
                    cluster_metrics = result_data["cluster_metrics"]

                    # 检查是否有节点参与
                    avg_busy = cluster_metrics.get("avg_busy_slaves", 0)
                    max_tasks = cluster_metrics.get("max_total_tasks", 0)

                    if avg_busy > 0 or max_tasks > 0:
                        print(f"\n[OK] 分布式推理指标:")
                        print(f"  平均忙碌从节点: {avg_busy:.1f}")
                        print(f"  最大任务数: {max_tasks}")

                        distributed_validation = ResponseValidator.validate_distributed_inference(
                            metrics={
                                "nodes_involved": int(avg_busy) + 1,  # +1 for master
                                "layer_forward_count": max_tasks,
                                "network_traffic": max_tasks,  # 简化
                            },
                            expected_nodes=slave_count + 1,  # slaves + master
                        )

                        if verbose:
                            print(f"\n{distributed_validation}")

                # 创建测试结果
                test_result = TestResult(
                    test_name=f"distributed_{scenario_name}_{i}",
                    timestamp=datetime.now().isoformat(),
                    status="PASSED" if validation.is_valid else "FAILED",
                    question=question,
                    response=result_data["response"],
                    latency=result_data["latency"],
                    response_length=result_data["length"],
                    validation=validation,
                    metrics={
                        "cluster_metrics": result_data.get("cluster_metrics"),
                        "distributed_validation": distributed_validation.to_dict() if distributed_validation else None,
                    },
                )

                results.append(test_result)

                # 短暂等待，避免过载
                await asyncio.sleep(2)

            # 4. 生成测试报告
            print("\n" + "=" * 70)
            print("测试报告")
            print("=" * 70)

            report = generate_distributed_report(results, scenario, slave_count)
            print_distributed_report(report)

            # 保存报告
            report_path = Path(__file__).parent / "results" / f"distributed_{scenario_name}_{slave_count}slaves.json"
            save_distributed_report(report, report_path)
            print(f"\n[OK] 报告已保存: {report_path}")

            return results

    except Exception as e:
        print(f"\n[FAIL] 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        # 5. 清理
        await orchestrator.teardown()


def generate_distributed_report(
    results: List[TestResult],
    scenario: Scenario,
    slave_count: int,
) -> dict:
    """生成分布式推理测试报告

    Args:
        results: 测试结果列表
        scenario: 测试场景
        slave_count: 从节点数量

    Returns:
        报告字典
    """
    # 统计
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASSED")
    failed = total - passed

    latencies = [r.latency for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0

    lengths = [r.response_length for r in results]
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    # 构建报告
    report = {
        "test_name": f"distributed_{scenario.name}_{slave_count}slaves",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "slave_count": slave_count,
            "mode": "distributed",
        },
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "question_count": len(scenario.questions),
            "expected_length": scenario.expected_length,
        },
        "summary": {
            "total_requests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
        },
        "performance": {
            "avg_latency": round(avg_latency, 2),
            "max_latency": round(max_latency, 2),
            "min_latency": round(min_latency, 2),
            "avg_response_length": round(avg_length, 2),
        },
        "results": [r.to_dict() for r in results],
    }

    return report


def print_distributed_report(report: dict) -> None:
    """打印分布式推理测试报告

    Args:
        report: 报告字典
    """
    print(f"\n测试名称: {report['test_name']}")
    print(f"执行时间: {report['timestamp']}")

    print(f"\n配置信息:")
    config = report["config"]
    print(f"  模式: {config['mode']}")
    print(f"  从节点数: {config['slave_count']}")

    print(f"\n场景信息:")
    scenario = report["scenario"]
    print(f"  名称: {scenario['name']}")
    print(f"  描述: {scenario['description']}")
    print(f"  问题数: {scenario['question_count']}")

    print(f"\n测试统计:")
    summary = report["summary"]
    print(f"  总请求数: {summary['total_requests']}")
    print(f"  通过: {summary['passed']}")
    print(f"  失败: {summary['failed']}")
    print(f"  成功率: {summary['success_rate']:.1%}")

    print(f"\n性能指标:")
    perf = report["performance"]
    print(f"  平均延迟: {perf['avg_latency']}s")
    print(f"  最大延迟: {perf['max_latency']}s")
    print(f"  最小延迟: {perf['min_latency']}s")
    print(f"  平均响应长度: {perf['avg_response_length']} 字符")

    # 判断整体状态
    success_rate = summary["success_rate"]
    avg_latency = perf["avg_latency"]

    print(f"\n整体状态:")
    if success_rate >= 0.95 and avg_latency < 10:
        print(f"  [PASS] 通过 (成功率 {success_rate:.1%} >= 95%, 平均延迟 {avg_latency}s < 10s)")
    else:
        print(f"  [FAIL] 失败")
        if success_rate < 0.95:
            print(f"    - 成功率 {success_rate:.1%} < 95%")
        if avg_latency >= 10:
            print(f"    - 平均延迟 {avg_latency}s >= 10s")


def save_distributed_report(report: dict, path: Path) -> None:
    """保存分布式推理测试报告到文件

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
    scenario_name = "simple_qa"
    slave_count = 2

    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            slave_count = int(sys.argv[2])
        except ValueError:
            print(f"警告: 无效的从节点数 '{sys.argv[2]}'，使用默认值 2")
            slave_count = 2

    print(f"使用场景: {scenario_name}, 从节点数: {slave_count}")

    # 执行测试
    results = await test_distributed_inference(
        scenario_name=scenario_name,
        slave_count=slave_count,
        verbose=True,
    )

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
