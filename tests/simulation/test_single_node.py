"""
单机推理测试

测试在无从节点的情况下，主节点独立完成推理的能力。
这是最基础的测试场景，确保系统在单机模式下可用。
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


async def execute_inference(
    sender: RequestSender,
    question: str,
    session_id: str = "test-session",
    simulator: HumanSimulator = None,
    timeout: float = 60.0,
) -> dict:
    """执行单次推理请求

    Args:
        sender: 请求发送器
        question: 问题文本
        session_id: 会话 ID
        simulator: 人机模拟器（可选）

    Returns:
        推理结果字典
    """
    # 如果提供了模拟器，使用模拟输入
    if simulator:
        full_input = await simulator.simulate_input_batch(question)
    else:
        full_input = question

    response_text = ""
    start_time = time.time()

    async def send_with_timeout():
        nonlocal response_text
        async for chunk in sender.send_chat_request(full_input, session_id, stream=True):
            if chunk["type"] == "chunk":
                content = chunk.get("content", "")
                response_text += content
                # 实时打印（可选）
                print(content, end="", flush=True)

    try:
        await asyncio.wait_for(send_with_timeout(), timeout=timeout)
        latency = time.time() - start_time
        return {
            "question": question,
            "response": response_text,
            "latency": latency,
            "length": len(response_text),
            "timeout": False,
        }
    except asyncio.TimeoutError:
        return {
            "question": question,
            "response": "",
            "latency": timeout,
            "length": 0,
            "timeout": True,
            "error": f"请求超时 ({timeout}s)",
        }
    except Exception as e:
        return {
            "question": question,
            "response": "",
            "latency": 0,
            "length": 0,
            "timeout": False,
            "error": str(e),
        }


async def test_single_node_inference(
    scenario_name: str = "simple_qa",
    verbose: bool = True,
) -> List[TestResult]:
    """单机推理完整测试

    Args:
        scenario_name: 测试场景名称
        verbose: 是否打印详细信息

    Returns:
        测试结果列表
    """
    print("\n" + "=" * 70)
    print("单机推理测试")
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

    # 1. 启动后端（仅主节点）
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=False,
        slave_count=0,
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
            slave_count = status.get("slave_count", 0)

            print(f"[OK] 运行模式: {mode}")
            print(f"[OK] 从节点数: {slave_count}")

            if mode != "single":
                print(f"[WARN] 警告: 预期单机模式，实际为 {mode}")

            if slave_count != 0:
                print(f"[WARN] 警告: 预期 0 个从节点，实际为 {slave_count}")

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
                result_data = await execute_inference(
                    sender=sender,
                    question=question,
                    session_id="test-single-node",
                    simulator=simulator,
                    timeout=config.request_timeout,
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
                if result_data.get("error"):
                    validation.errors.append(result_data["error"])
                    validation.is_valid = False

                if verbose:
                    print(f"\n{validation}")

                # 创建测试结果
                test_result = TestResult(
                    test_name=f"single_node_{scenario_name}_{i}",
                    timestamp=datetime.now().isoformat(),
                    status="PASSED" if validation.is_valid else "FAILED",
                    question=question,
                    response=result_data["response"],
                    latency=result_data["latency"],
                    response_length=result_data["length"],
                    validation=validation,
                    metrics={
                        "timeout": result_data.get("timeout", False),
                        "error": result_data.get("error"),
                    },
                )

                results.append(test_result)

                # 短暂等待，避免过载
                await asyncio.sleep(1)

            # 4. 生成测试报告
            print("\n" + "=" * 70)
            print("测试报告")
            print("=" * 70)

            report = generate_report(results, scenario)
            print_report(report)

            # 保存报告
            report_path = Path(__file__).parent / "results" / f"single_node_{scenario_name}.json"
            save_report(report, report_path)
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


def generate_report(results: List[TestResult], scenario: Scenario) -> dict:
    """生成测试报告

    Args:
        results: 测试结果列表
        scenario: 测试场景

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
        "test_name": f"single_node_{scenario.name}",
        "timestamp": datetime.now().isoformat(),
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


def print_report(report: dict) -> None:
    """打印测试报告

    Args:
        report: 报告字典
    """
    print(f"\n测试名称: {report['test_name']}")
    print(f"执行时间: {report['timestamp']}")

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
        print(f"  [OK] 通过 (成功率 {success_rate:.1%} >= 95%, 平均延迟 {avg_latency}s < 10s)")
    else:
        print(f"  [FAIL] 失败")
        if success_rate < 0.95:
            print(f"    - 成功率 {success_rate:.1%} < 95%")
        if avg_latency >= 10:
            print(f"    - 平均延迟 {avg_latency}s >= 10s")


def save_report(report: dict, path: Path) -> None:
    """保存测试报告到文件

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
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]

    print(f"使用场景: {scenario_name}")

    # 执行测试
    results = await test_single_node_inference(
        scenario_name=scenario_name,
        verbose=True,
    )

    # 返回退出码
    if results:
        passed = sum(1 for r in results if r.status == "PASSED")
        total = len(results)
        success_rate = passed / total

        if success_rate >= 0.95:
            print(f"\n[OK] 测试通过 ({success_rate:.1%})")
            sys.exit(0)
        else:
            print(f"\n[FAIL] 测试失败 ({success_rate:.1%})")
            sys.exit(1)
    else:
        print("\n[FAIL] 测试未执行")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
