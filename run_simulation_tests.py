#!/usr/bin/env python3
"""
仿真测试运行脚本

提供便捷的命令行接口来运行各种仿真测试。
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.simulation.scenarios import list_scenarios, SCENARIOS
from tests.simulation.test_single_node import test_single_node_inference
from tests.simulation.test_distributed import test_distributed_inference
from tests.simulation.test_degradation import test_graceful_degradation
from tests.simulation.test_stress import test_stress
from tests.simulation.test_exceptions import test_network_interruption, test_node_crash


async def run_test(scenario_name: str, test_type: str = "single", slave_count: int = 2, verbose: bool = True, concurrent_users: int = 5, requests_per_user: int = 3, exception_type: str = "network_interruption"):
    """运行指定场景的测试
    
    Args:
        scenario_name: 场景名称
        test_type: 测试类型 ("single", "distributed", "degradation", "stress", "network_interruption", "node_crash")
        slave_count: 从节点数量（仅用于 distributed、degradation、stress 和 exceptions 测试）
        verbose: 是否打印详细信息
        concurrent_users: 并发用户数（仅用于 stress 测试）
        requests_per_user: 每用户请求数（仅用于 stress 测试）
        exception_type: 异常类型（仅用于 exceptions 测试："network_interruption" 或 "node_crash"）
    """
    print(f"\n{'='*70}")
    print(f"运行仿真测试: {scenario_name} ({test_type})")
    print(f"{'='*70}\n")
    
    # 根据测试类型执行不同的测试
    if test_type == "single":
        results = await test_single_node_inference(
            scenario_name=scenario_name,
            verbose=verbose,
        )
    elif test_type == "distributed":
        results = await test_distributed_inference(
            scenario_name=scenario_name,
            slave_count=slave_count,
            verbose=verbose,
        )
    elif test_type == "degradation":
        results = await test_graceful_degradation(
            scenario_name=scenario_name,
            initial_slave_count=slave_count,
            verbose=verbose,
        )
    elif test_type == "stress":
        stats = await test_stress(
            scenario_name=scenario_name,
            concurrent_users=concurrent_users,
            requests_per_user=requests_per_user,
            slave_count=slave_count,
            verbose=verbose,
        )
        # stress test 返回 stats dict 而不是 results list
        if stats:
            success_rate = stats.get("success_rate", 0)
            print(f"\n{'='*70}")
            print(f"压力测试完成")
            print(f"{'='*70}")
            print(f"成功率: {success_rate:.1%}")
            print(f"吞吐量: {stats.get('throughput', 0):.2f} req/s")
            
            if success_rate >= 0.95:
                print(f"✓ 测试通过")
                return 0
            else:
                print(f"✗ 测试失败")
                return 1
        else:
            print(f"\n✗ 测试未执行")
            return 1
    elif test_type == "network_interruption":
        results = await test_network_interruption(
            scenario_name=scenario_name,
            slave_count=slave_count,
            verbose=verbose,
        )
    elif test_type == "node_crash":
        results = await test_node_crash(
            scenario_name=scenario_name,
            slave_count=slave_count,
            verbose=verbose,
        )
    else:
        print(f"错误: 未知的测试类型 '{test_type}'")
        return 1
    
    # 统计结果
    if results:
        passed = sum(1 for r in results if r.status == "PASSED")
        total = len(results)
        success_rate = passed / total
        
        print(f"\n{'='*70}")
        print(f"测试完成")
        print(f"{'='*70}")
        print(f"通过: {passed}/{total} ({success_rate:.1%})")
        
        if success_rate >= 0.95:
            print(f"✓ 测试通过")
            return 0
        else:
            print(f"✗ 测试失败")
            return 1
    else:
        print(f"\n✗ 测试未执行")
        return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="分布式推理仿真测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有可用场景
  python run_simulation_tests.py --list
  
  # 运行单机推理测试
  python run_simulation_tests.py simple_qa
  
  # 运行分布式推理测试（2个从节点）
  python run_simulation_tests.py simple_qa --type distributed --slaves 2
  
  # 运行降级测试（3个从节点）
  python run_simulation_tests.py simple_qa --type degradation --slaves 3
  
  # 运行压力测试（5并发用户，每用户3请求，2个从节点）
  python run_simulation_tests.py stress_test --type stress --users 5 --requests 3 --slaves 2
  
  # 运行网络中断测试
  python run_simulation_tests.py simple_qa --type network_interruption --slaves 3
  
  # 运行节点崩溃测试
  python run_simulation_tests.py simple_qa --type node_crash --slaves 3
  
  # 运行多轮对话测试（静默模式）
  python run_simulation_tests.py multi_turn --quiet
  
可用场景:
  simple_qa              基础知识问答
  multi_turn             多轮对话
  complex_reasoning      复杂推理
  code_generation        代码生成
  technical_explanation  技术解释
  math_problems          数学问题
  long_generation        长文本生成
  stress_test            压力测试

测试类型:
  single                 单机推理（默认）
  distributed            分布式推理（需要指定从节点数）
  degradation            降级测试（模拟从节点离线）
  stress                 压力测试（需要指定并发用户数和请求数）
  network_interruption   网络中断测试（模拟网络故障）
  node_crash             节点崩溃测试（模拟节点崩溃并恢复）
        """
    )
    
    parser.add_argument(
        "scenario",
        nargs="?",
        help="测试场景名称（使用 --list 查看可用场景）"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用场景"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["single", "distributed", "degradation", "stress", "network_interruption", "node_crash"],
        default="single",
        help="测试类型: single (单机), distributed (分布式), degradation (降级), stress (压力), network_interruption (网络中断), node_crash (节点崩溃)"
    )
    
    parser.add_argument(
        "--slaves", "-s",
        type=int,
        default=2,
        help="从节点数量 (默认: 2，仅用于 distributed、degradation、stress 和 exceptions 测试)"
    )
    
    parser.add_argument(
        "--users", "-u",
        type=int,
        default=5,
        help="并发用户数 (默认: 5，仅用于 stress 测试)"
    )
    
    parser.add_argument(
        "--requests", "-r",
        type=int,
        default=3,
        help="每用户请求数 (默认: 3，仅用于 stress 测试)"
    )
    
    parser.add_argument(
        "--exception-type", "-e",
        choices=["network_interruption", "node_crash"],
        default="network_interruption",
        help="异常类型 (默认: network_interruption，仅用于 exceptions 测试)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式（减少输出）"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="运行所有场景"
    )
    
    args = parser.parse_args()
    
    # 列出场景
    if args.list:
        print("\n可用测试场景:")
        print("=" * 70)
        for name in list_scenarios():
            scenario = SCENARIOS[name]
            print(f"  {name:25s} {scenario.description}")
            print(f"  {'':25s} 难度: {scenario.difficulty}, 问题数: {len(scenario.questions)}")
            print()
        return 0
    
    # 运行所有场景
    if args.all:
        print("\n运行所有测试场景")
        print("=" * 70)
        
        all_results = []
        for scenario_name in list_scenarios():
            exit_code = asyncio.run(run_test(
                scenario_name,
                test_type=args.type,
                slave_count=args.slaves,
                concurrent_users=args.users,
                requests_per_user=args.requests,
                exception_type=args.exception_type,
                verbose=not args.quiet
            ))
            all_results.append((scenario_name, exit_code))
            print()
        
        # 汇总
        print("\n" + "=" * 70)
        print("所有场景测试汇总")
        print("=" * 70)
        
        passed = sum(1 for _, code in all_results if code == 0)
        total = len(all_results)
        
        for name, code in all_results:
            status = "✓ 通过" if code == 0 else "✗ 失败"
            print(f"  {status} - {name}")
        
        print(f"\n总计: {passed}/{total} 通过")
        
        return 0 if passed == total else 1
    
    # 运行单个场景
    if not args.scenario:
        parser.print_help()
        return 1
    
    if args.scenario not in SCENARIOS:
        print(f"\n错误: 场景 '{args.scenario}' 不存在")
        print(f"使用 --list 查看可用场景")
        return 1
    
    return asyncio.run(run_test(
        args.scenario,
        test_type=args.type,
        slave_count=args.slaves,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        exception_type=args.exception_type,
        verbose=not args.quiet
    ))


if __name__ == "__main__":
    sys.exit(main())
