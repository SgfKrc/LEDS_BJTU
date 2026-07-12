"""
压力测试

测试系统在高并发请求下的稳定性和性能表现。
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List
import statistics

from .framework import (
    TestOrchestrator,
    TestConfig,
    TestResult,
    RequestSender,
    HumanSimulator,
    ResponseValidator,
)
from .scenarios import SCENARIOS, Scenario


async def concurrent_inference_test(
    sender: RequestSender,
    questions: List[str],
    concurrent_users: int = 5,
    requests_per_user: int = 3,
    session_prefix: str = "stress-test",
) -> dict:
    """并发推理测试

    Args:
        sender: 请求发送器
        questions: 问题列表
        concurrent_users: 并发用户数
        requests_per_user: 每个用户的请求数
        session_prefix: 会话ID前缀

    Returns:
        测试结果统计
    """
    print(f"\n开始并发测试: {concurrent_users} 用户, 每用户 {requests_per_user} 请求")

    if not questions:
        return {
            "total_requests": 0,
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0,
            "total_time": 0,
            "throughput": 0,
            "errors": [{"request_id": None, "error": "问题列表为空，无法执行压力测试"}],
        }

    if concurrent_users <= 0 or requests_per_user <= 0:
        return {
            "total_requests": 0,
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0,
            "total_time": 0,
            "throughput": 0,
            "errors": [{"request_id": None, "error": "并发用户数和每用户请求数必须大于 0"}],
        }

    # 生成用户请求
    user_requests = []
    for user_id in range(concurrent_users):
        session_id = f"{session_prefix}-user-{user_id}"
        for req_id in range(requests_per_user):
            # 循环使用问题列表
            question = questions[req_id % len(questions)]
            user_requests.append({
                "session_id": session_id,
                "question": question,
                "user_id": user_id,
                "req_id": req_id,
            })

    print(f"总请求数: {len(user_requests)}")

    # 并发执行所有请求
    tasks = []
    for req in user_requests:
        task = asyncio.create_task(
            execute_single_request(
                sender=sender,
                question=req["question"],
                session_id=req["session_id"],
            )
        )
        tasks.append(task)

    # 等待所有请求完成
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time

    # 统计结果
    success_count = 0
    failure_count = 0
    latencies = []
    response_lengths = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failure_count += 1
            errors.append({
                "request_id": i,
                "error": str(result),
            })
        else:
            if result["status"] == "PASSED":
                success_count += 1
                latencies.append(result["latency"])
                response_lengths.append(result["response_length"])
            else:
                failure_count += 1
                errors.append({
                    "request_id": i,
                    "error": result.get("validation", {}).get("errors", ["Unknown error"]),
                })

    # 计算统计指标
    stats = {
        "total_requests": len(user_requests),
        "concurrent_users": concurrent_users,
        "requests_per_user": requests_per_user,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / len(user_requests) if user_requests else 0,
        "total_time": total_time,
        "throughput": success_count / total_time if total_time > 0 else 0,  # requests/second
    }

    if latencies:
        stats.update({
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            "p99_latency": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
            "stddev_latency": statistics.stdev(latencies) if len(latencies) >= 2 else 0,
        })

    if response_lengths:
        stats.update({
            "avg_response_length": statistics.mean(response_lengths),
            "min_response_length": min(response_lengths),
            "max_response_length": max(response_lengths),
        })

    stats["errors"] = errors

    return stats


async def execute_single_request(
    sender: RequestSender,
    question: str,
    session_id: str,
) -> dict:
    """执行单个推理请求

    Args:
        sender: 请求发送器
        question: 问题文本
        session_id: 会话 ID

    Returns:
        请求结果
    """
    try:
        response_text = ""
        start_time = time.time()

        async for chunk in sender.send_chat_request(question, session_id, stream=True):
            if chunk["type"] == "chunk":
                content = chunk.get("content", "")
                response_text += content

        latency = time.time() - start_time

        # 验证响应
        validation = ResponseValidator.validate_response(
            response=response_text,
            question=question,
            expected_length=(20, 500),  # 压力测试放宽长度要求
        )

        return {
            "status": "PASSED" if validation.is_valid else "FAILED",
            "response": response_text,
            "latency": latency,
            "response_length": len(response_text),
            "validation": validation,
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "latency": 0,
            "response_length": 0,
        }


async def test_stress(
    scenario_name: str = "stress_test",
    concurrent_users: int = 5,
    requests_per_user: int = 3,
    slave_count: int = 2,
    verbose: bool = True,
) -> dict:
    """压力测试完整流程

    Args:
        scenario_name: 测试场景名称
        concurrent_users: 并发用户数
        requests_per_user: 每个用户的请求数
        slave_count: 从节点数量
        verbose: 是否打印详细信息

    Returns:
        测试结果统计
    """
    print("\n" + "=" * 70)
    print(f"压力测试 (并发用户: {concurrent_users}, 每用户请求: {requests_per_user})")
    print("=" * 70)

    # 获取测试场景
    if scenario_name not in SCENARIOS:
        print(f"错误: 场景 '{scenario_name}' 不存在")
        print(f"可用场景: {', '.join(SCENARIOS.keys())}")
        return {}

    scenario = SCENARIOS[scenario_name]
    print(f"\n测试场景: {scenario.name}")
    print(f"描述: {scenario.description}")
    print(f"问题数: {len(scenario.questions)}")

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
                return {}

            # 3. 执行压力测试
            print("\n" + "-" * 70)
            print("执行压力测试")
            print("-" * 70)

            stats = await concurrent_inference_test(
                sender=sender,
                questions=scenario.questions,
                concurrent_users=concurrent_users,
                requests_per_user=requests_per_user,
            )

            # 4. 打印结果
            print("\n" + "=" * 70)
            print("压力测试结果")
            print("=" * 70)

            print_stats(stats)

            # 5. 保存报告
            report = generate_stress_report(stats, scenario, concurrent_users, requests_per_user, slave_count)
            report_path = Path(__file__).parent / "results" / f"stress_{scenario_name}_{concurrent_users}users.json"
            save_stress_report(report, report_path)
            print(f"\n[OK] 报告已保存: {report_path}")

            return stats

    except Exception as e:
        print(f"\n[FAIL] 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return {}

    finally:
        # 6. 清理
        await orchestrator.teardown()


def print_stats(stats: dict) -> None:
    """打印压力测试统计结果

    Args:
        stats: 统计结果字典
    """
    print(f"\n基本信息:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  并发用户数: {stats['concurrent_users']}")
    print(f"  每用户请求数: {stats['requests_per_user']}")

    print(f"\n成功率:")
    print(f"  成功: {stats['success_count']}")
    print(f"  失败: {stats['failure_count']}")
    print(f"  成功率: {stats['success_rate']:.1%}")

    print(f"\n性能指标:")
    print(f"  总耗时: {stats['total_time']:.2f}s")
    print(f"  吞吐量: {stats['throughput']:.2f} req/s")

    if 'avg_latency' in stats:
        print(f"\n延迟统计:")
        print(f"  平均延迟: {stats['avg_latency']:.2f}s")
        print(f"  最小延迟: {stats['min_latency']:.2f}s")
        print(f"  最大延迟: {stats['max_latency']:.2f}s")
        print(f"  中位数延迟: {stats['median_latency']:.2f}s")
        print(f"  P95 延迟: {stats['p95_latency']:.2f}s")
        print(f"  P99 延迟: {stats['p99_latency']:.2f}s")
        print(f"  标准差: {stats['stddev_latency']:.2f}s")

    if 'avg_response_length' in stats:
        print(f"\n响应长度:")
        print(f"  平均长度: {stats['avg_response_length']:.0f} 字符")
        print(f"  最小长度: {stats['min_response_length']} 字符")
        print(f"  最大长度: {stats['max_response_length']} 字符")

    if stats['errors']:
        print(f"\n错误详情 (前 10 个):")
        for error in stats['errors'][:10]:
            print(f"  请求 {error['request_id']}: {error['error']}")

    # 判断整体状态
    success_rate = stats['success_rate']
    throughput = stats['throughput']

    print(f"\n整体状态:")
    if success_rate >= 0.95 and throughput >= 0.5:
        print(f"  [PASS] 通过 (成功率 {success_rate:.1%} >= 95%, 吞吐量 {throughput:.2f} req/s >= 0.5)")
    else:
        print(f"  [FAIL] 失败")
        if success_rate < 0.95:
            print(f"    - 成功率 {success_rate:.1%} < 95%")
        if throughput < 0.5:
            print(f"    - 吞吐量 {throughput:.2f} req/s < 0.5")


def generate_stress_report(
    stats: dict,
    scenario: Scenario,
    concurrent_users: int,
    requests_per_user: int,
    slave_count: int,
) -> dict:
    """生成压力测试报告

    Args:
        stats: 统计结果
        scenario: 测试场景
        concurrent_users: 并发用户数
        requests_per_user: 每用户请求数
        slave_count: 从节点数量

    Returns:
        报告字典
    """
    report = {
        "test_name": f"stress_{scenario.name}_{concurrent_users}users",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "slave_count": slave_count,
        },
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "question_count": len(scenario.questions),
        },
        "stats": stats,
    }

    return report


def save_stress_report(report: dict, path: Path) -> None:
    """保存压力测试报告到文件

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
    scenario_name = "stress_test"
    concurrent_users = 5
    requests_per_user = 3
    slave_count = 2

    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            concurrent_users = int(sys.argv[2])
        except ValueError:
            print(f"警告: 无效的并发用户数 '{sys.argv[2]}'，使用默认值 5")
            concurrent_users = 5

    if len(sys.argv) > 3:
        try:
            requests_per_user = int(sys.argv[3])
        except ValueError:
            print(f"警告: 无效的每用户请求数 '{sys.argv[3]}'，使用默认值 3")
            requests_per_user = 3

    if len(sys.argv) > 4:
        try:
            slave_count = int(sys.argv[4])
        except ValueError:
            print(f"警告: 无效的从节点数 '{sys.argv[4]}'，使用默认值 2")
            slave_count = 2

    print(f"使用场景: {scenario_name}")
    print(f"并发用户: {concurrent_users}, 每用户请求: {requests_per_user}, 从节点: {slave_count}")

    # 执行测试
    stats = await test_stress(
        scenario_name=scenario_name,
        concurrent_users=concurrent_users,
        requests_per_user=requests_per_user,
        slave_count=slave_count,
        verbose=True,
    )

    # 返回退出码
    if stats:
        success_rate = stats['success_rate']
        throughput = stats['throughput']

        if success_rate >= 0.95 and throughput >= 0.5:
            print(f"\n[SUCCESS] 测试通过 (成功率 {success_rate:.1%}, 吞吐量 {throughput:.2f} req/s)")
            sys.exit(0)
        else:
            print(f"\n[FAIL] 测试失败 (成功率 {success_rate:.1%}, 吞吐量 {throughput:.2f} req/s)")
            sys.exit(1)
    else:
        print("\n[FAIL] 测试未执行")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
