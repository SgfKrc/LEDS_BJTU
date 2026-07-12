"""
降级测试

测试系统在从节点离线时的降级能力，确保系统能够自动降级到单机模式。
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
    ValidationResult,
    RequestSender,
    HumanSimulator,
    ResponseValidator,
)
from .scenarios import SCENARIOS, Scenario


def merge_validations(*validations: ValidationResult) -> ValidationResult:
    errors = []
    warnings = []
    for validation in validations:
        errors.extend(validation.errors)
        warnings.extend(validation.warnings)
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_cluster_status(
    status: dict,
    expected_mode: str = None,
    min_online_slaves: int = None,
    max_online_slaves: int = None,
) -> ValidationResult:
    errors = []
    warnings = []
    mode = status.get("mode", "unknown")
    slaves = status.get("slaves", [])
    online_slaves = sum(1 for s in slaves if s.get("state") == "online")

    if expected_mode and mode != expected_mode:
        errors.append(f"集群模式不符: {mode} != {expected_mode}")
    if min_online_slaves is not None and online_slaves < min_online_slaves:
        errors.append(f"在线从节点过少: {online_slaves} < {min_online_slaves}")
    if max_online_slaves is not None and online_slaves > max_online_slaves:
        errors.append(f"在线从节点过多: {online_slaves} > {max_online_slaves}")
    if status.get("error"):
        errors.append(f"集群状态返回错误: {status.get('error')}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


async def execute_inference_with_status(
    sender: RequestSender,
    question: str,
    session_id: str = "test-degradation",
    simulator: HumanSimulator = None,
) -> dict:
    """执行推理请求并记录集群状态

    Args:
        sender: 请求发送器
        question: 问题文本
        session_id: 会话 ID
        simulator: 人机模拟器（可选）

    Returns:
        推理结果字典（包含集群状态）
    """
    # 如果提供了模拟器，使用模拟输入
    if simulator:
        full_input = await simulator.simulate_input_batch(question)
    else:
        full_input = question

    # 获取推理前的集群状态
    status_before = await sender.get_cluster_status()

    # 发送请求
    response_text = ""
    start_time = time.time()

    async for chunk in sender.send_chat_request(full_input, session_id, stream=True):
        if chunk["type"] == "chunk":
            content = chunk.get("content", "")
            response_text += content
            print(content, end="", flush=True)

    latency = time.time() - start_time

    # 获取推理后的集群状态
    status_after = await sender.get_cluster_status()

    return {
        "question": question,
        "response": response_text,
        "latency": latency,
        "length": len(response_text),
        "status_before": status_before,
        "status_after": status_after,
    }


async def test_graceful_degradation(
    scenario_name: str = "simple_qa",
    initial_slave_count: int = 3,
    verbose: bool = True,
) -> List[TestResult]:
    """降级测试

    测试流程:
    1. 启动主节点 + 多个从节点
    2. 执行推理测试（基线）
    3. 停止部分从节点
    4. 执行推理测试（降级后）
    5. 停止所有从节点
    6. 执行推理测试（单机模式）
    7. 验证所有测试都成功

    Args:
        scenario_name: 测试场景名称
        initial_slave_count: 初始从节点数量
        verbose: 是否打印详细信息

    Returns:
        测试结果列表
    """
    print("\n" + "=" * 70)
    print(f"降级测试 (初始从节点数: {initial_slave_count})")
    print("=" * 70)

    # 获取测试场景
    if scenario_name not in SCENARIOS:
        print(f"错误: 场景 '{scenario_name}' 不存在")
        print(f"可用场景: {', '.join(SCENARIOS.keys())}")
        return []

    scenario = SCENARIOS[scenario_name]
    print(f"\n测试场景: {scenario.name}")
    print(f"描述: {scenario.description}")

    # 选择前3个问题进行测试
    test_questions = scenario.questions[:3]
    print(f"问题数: {len(test_questions)}")
    print(f"预期长度: {scenario.expected_length}")

    # 1. 启动后端（主节点 + 从节点）
    orchestrator = TestOrchestrator()
    config = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=initial_slave_count,
    )

    all_results = []

    try:
        await orchestrator.setup(config)

        # 2. 验证初始集群状态
        print("\n" + "-" * 70)
        print("阶段 1: 完整集群测试")
        print("-" * 70)

        async with RequestSender() as sender:
            status = await sender.get_cluster_status()
            mode = status.get("mode", "unknown")
            slave_count = len(status.get("slaves", []))
            online_slaves = sum(
                1 for s in status.get("slaves", [])
                if s.get("state") == "online"
            )

            print(f"[OK] 运行模式: {mode}")
            print(f"[OK] 从节点数: {slave_count} (在线: {online_slaves})")

            if mode != "distributed":
                print(f"[FAIL] 预期分布式模式，实际为 {mode}")
                return []

            # 执行基线测试
            simulator = HumanSimulator(typing_speed=5.0, pause_probability=0.2)

            print(f"\n执行基线推理测试...")
            baseline_question = test_questions[0]
            print(f"问题: {baseline_question}")

            baseline_result = await execute_inference_with_status(
                sender=sender,
                question=baseline_question,
                session_id="test-baseline",
                simulator=simulator,
            )

            print(f"\n\n响应 (长度: {baseline_result['length']}):")
            print(f"  {baseline_result['response'][:200]}...")
            print(f"\n延迟: {baseline_result['latency']:.2f}s")

            # 验证响应
            validation = ResponseValidator.validate_response(
                response=baseline_result["response"],
                question=baseline_question,
                expected_length=scenario.expected_length,
            )
            cluster_validation = validate_cluster_status(
                baseline_result["status_after"],
                expected_mode="distributed",
                min_online_slaves=initial_slave_count,
            )
            validation = merge_validations(validation, cluster_validation)

            if verbose:
                print(f"\n{validation}")

            test_result = TestResult(
                test_name=f"degradation_baseline_{scenario_name}",
                timestamp=datetime.now().isoformat(),
                status="PASSED" if validation.is_valid else "FAILED",
                question=baseline_question,
                response=baseline_result["response"],
                latency=baseline_result["latency"],
                response_length=baseline_result["length"],
                validation=validation,
                metrics={
                    "phase": "baseline",
                    "mode": mode,
                    "slave_count": slave_count,
                    "online_slaves": online_slaves,
                },
            )

            all_results.append(test_result)

            # 3. 停止部分从节点
            print("\n" + "-" * 70)
            print(f"阶段 2: 部分降级测试 (停止 1 个从节点)")
            print("-" * 70)

            print(f"停止从节点 0...")
            await orchestrator.stop_slave(0)

            # 等待集群稳定
            print("等待集群稳定 (5秒)...")
            await asyncio.sleep(5)

            # 检查集群状态
            status = await sender.get_cluster_status()
            mode = status.get("mode", "unknown")
            slave_count = len(status.get("slaves", []))
            online_slaves = sum(
                1 for s in status.get("slaves", [])
                if s.get("state") == "online"
            )

            print(f"[OK] 运行模式: {mode}")
            print(f"[OK] 从节点数: {slave_count} (在线: {online_slaves})")

            # 执行降级后测试
            print(f"\n执行降级后推理测试...")
            partial_question = test_questions[1]
            print(f"问题: {partial_question}")

            partial_result = await execute_inference_with_status(
                sender=sender,
                question=partial_question,
                session_id="test-partial",
                simulator=simulator,
            )

            print(f"\n\n响应 (长度: {partial_result['length']}):")
            print(f"  {partial_result['response'][:200]}...")
            print(f"\n延迟: {partial_result['latency']:.2f}s")

            # 验证响应
            validation = ResponseValidator.validate_response(
                response=partial_result["response"],
                question=partial_question,
                expected_length=scenario.expected_length,
            )
            cluster_validation = validate_cluster_status(
                partial_result["status_after"],
                max_online_slaves=max(0, initial_slave_count - 1),
            )
            validation = merge_validations(validation, cluster_validation)

            if verbose:
                print(f"\n{validation}")

            test_result = TestResult(
                test_name=f"degradation_partial_{scenario_name}",
                timestamp=datetime.now().isoformat(),
                status="PASSED" if validation.is_valid else "FAILED",
                question=partial_question,
                response=partial_result["response"],
                latency=partial_result["latency"],
                response_length=partial_result["length"],
                validation=validation,
                metrics={
                    "phase": "partial_degradation",
                    "mode": mode,
                    "slave_count": slave_count,
                    "online_slaves": online_slaves,
                },
            )

            all_results.append(test_result)

            # 4. 停止所有从节点
            print("\n" + "-" * 70)
            print(f"阶段 3: 完全降级测试 (停止所有从节点)")
            print("-" * 70)

            # 停止剩余从节点（注意：索引已经偏移）
            while orchestrator.backend_manager.slave_processes:
                print("停止从节点 0...")
                await orchestrator.stop_slave(0)

            # 等待集群稳定
            print("等待集群稳定 (5秒)...")
            await asyncio.sleep(5)

            # 检查集群状态
            status = await sender.get_cluster_status()
            mode = status.get("mode", "unknown")
            slave_count = len(status.get("slaves", []))
            online_slaves = sum(
                1 for s in status.get("slaves", [])
                if s.get("state") == "online"
            )

            print(f"[OK] 运行模式: {mode}")
            print(f"[OK] 从节点数: {slave_count} (在线: {online_slaves})")

            # 验证是否降级到单机模式
            if mode != "single":
                print(f"[WARNING] 预期单机模式，实际为 {mode}")

            # 执行单机模式测试
            print(f"\n执行单机模式推理测试...")
            single_question = test_questions[2]
            print(f"问题: {single_question}")

            single_result = await execute_inference_with_status(
                sender=sender,
                question=single_question,
                session_id="test-single",
                simulator=simulator,
            )

            print(f"\n\n响应 (长度: {single_result['length']}):")
            print(f"  {single_result['response'][:200]}...")
            print(f"\n延迟: {single_result['latency']:.2f}s")

            # 验证响应
            validation = ResponseValidator.validate_response(
                response=single_result["response"],
                question=single_question,
                expected_length=scenario.expected_length,
            )
            cluster_validation = validate_cluster_status(
                single_result["status_after"],
                expected_mode="single",
                max_online_slaves=0,
            )
            validation = merge_validations(validation, cluster_validation)

            if verbose:
                print(f"\n{validation}")

            test_result = TestResult(
                test_name=f"degradation_single_{scenario_name}",
                timestamp=datetime.now().isoformat(),
                status="PASSED" if validation.is_valid else "FAILED",
                question=single_question,
                response=single_result["response"],
                latency=single_result["latency"],
                response_length=single_result["length"],
                validation=validation,
                metrics={
                    "phase": "full_degradation",
                    "mode": mode,
                    "slave_count": slave_count,
                    "online_slaves": online_slaves,
                },
            )

            all_results.append(test_result)

            # 5. 生成测试报告
            print("\n" + "=" * 70)
            print("降级测试报告")
            print("=" * 70)

            report = generate_degradation_report(
                all_results, scenario, initial_slave_count
            )
            print_degradation_report(report)

            # 保存报告
            report_path = Path(__file__).parent / "results" / f"degradation_{scenario_name}.json"
            save_degradation_report(report, report_path)
            print(f"\n[OK] 报告已保存: {report_path}")

            return all_results

    except Exception as e:
        print(f"\n[FAIL] 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        # 6. 清理
        await orchestrator.teardown()


def generate_degradation_report(
    results: List[TestResult],
    scenario: Scenario,
    initial_slave_count: int,
) -> dict:
    """生成降级测试报告

    Args:
        results: 测试结果列表
        scenario: 测试场景
        initial_slave_count: 初始从节点数量

    Returns:
        报告字典
    """
    # 按阶段分类
    baseline_results = [r for r in results if r.metrics.get("phase") == "baseline"]
    partial_results = [r for r in results if r.metrics.get("phase") == "partial_degradation"]
    single_results = [r for r in results if r.metrics.get("phase") == "full_degradation"]

    # 统计各阶段
    def calculate_stats(result_list):
        if not result_list:
            return {"count": 0}

        passed = sum(1 for r in result_list if r.status == "PASSED")
        total = len(result_list)
        latencies = [r.latency for r in result_list]

        return {
            "count": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
        }

    baseline_stats = calculate_stats(baseline_results)
    partial_stats = calculate_stats(partial_results)
    single_stats = calculate_stats(single_results)

    # 构建报告
    report = {
        "test_name": f"degradation_{scenario.name}",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "initial_slave_count": initial_slave_count,
        },
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
        },
        "phases": {
            "baseline": {
                "description": "完整集群 (主节点 + 所有从节点)",
                "stats": baseline_stats,
                "results": [r.to_dict() for r in baseline_results],
            },
            "partial_degradation": {
                "description": "部分降级 (停止部分从节点)",
                "stats": partial_stats,
                "results": [r.to_dict() for r in partial_results],
            },
            "full_degradation": {
                "description": "完全降级 (停止所有从节点，单机模式)",
                "stats": single_stats,
                "results": [r.to_dict() for r in single_results],
            },
        },
        "summary": {
            "total_tests": len(results),
            "total_passed": sum(1 for r in results if r.status == "PASSED"),
            "total_failed": sum(1 for r in results if r.status == "FAILED"),
            "overall_success_rate": sum(1 for r in results if r.status == "PASSED") / len(results) if results else 0,
        },
    }

    return report


def print_degradation_report(report: dict) -> None:
    """打印降级测试报告

    Args:
        report: 报告字典
    """
    print(f"\n测试名称: {report['test_name']}")
    print(f"执行时间: {report['timestamp']}")

    print(f"\n配置信息:")
    config = report["config"]
    print(f"  初始从节点数: {config['initial_slave_count']}")

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
        print(f"    平均延迟: {stats['avg_latency']:.2f}s")
        print(f"    延迟范围: {stats['min_latency']:.2f}s - {stats['max_latency']:.2f}s")

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


def save_degradation_report(report: dict, path: Path) -> None:
    """保存降级测试报告到文件

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
    initial_slave_count = 3

    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            initial_slave_count = int(sys.argv[2])
        except ValueError:
            print(f"警告: 无效的从节点数 '{sys.argv[2]}'，使用默认值 3")
            initial_slave_count = 3

    print(f"使用场景: {scenario_name}, 初始从节点数: {initial_slave_count}")

    # 执行测试
    results = await test_graceful_degradation(
        scenario_name=scenario_name,
        initial_slave_count=initial_slave_count,
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
