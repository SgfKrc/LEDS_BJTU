#!/usr/bin/env python3
"""
仿真测试框架验证脚本

验证框架的基本功能是否正常。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("仿真测试框架验证")
print("=" * 70)

# 测试 1: 导入验证
print("\n[1/5] 验证模块导入...")
try:
    from tests.simulation import (
        TestOrchestrator,
        TestConfig,
        TestResult,
        BackendManager,
        RequestSender,
        HumanSimulator,
        ResponseValidator,
        ValidationResult,
        SCENARIOS,
        Scenario,
    )
    print("  [OK] 所有模块导入成功")
except Exception as e:
    print(f"  [FAIL] 模块导入失败: {e}")
    sys.exit(1)

# 测试 2: 场景验证
print("\n[2/5] 验证测试场景...")
try:
    from tests.simulation.scenarios import list_scenarios
    
    scenarios = list_scenarios()
    print(f"  [OK] 发现 {len(scenarios)} 个场景:")
    for name in scenarios:
        scenario = SCENARIOS[name]
        print(f"    - {name}: {scenario.description} ({len(scenario.questions)} 问题)")
except Exception as e:
    print(f"  [FAIL] 场景验证失败: {e}")
    sys.exit(1)

# 测试 3: 配置验证
print("\n[3/5] 验证配置类...")
try:
    # 创建默认配置
    config1 = TestConfig()
    assert config1.start_master == True
    assert config1.start_slaves == False
    assert config1.slave_count == 0
    
    # 创建自定义配置
    config2 = TestConfig(
        start_master=True,
        start_slaves=True,
        slave_count=3,
    )
    assert config2.slave_count == 3
    
    print("  [OK] 配置类验证成功")
except Exception as e:
    print(f"  [FAIL] 配置类验证失败: {e}")
    sys.exit(1)

# 测试 4: 响应验证器
print("\n[4/5] 验证响应验证器...")
try:
    # 测试有效响应
    result1 = ResponseValidator.validate_response(
        response="机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测和决策，而不需要明确的编程指令。",
        question="什么是机器学习？",
        expected_length=(50, 200),
    )
    assert result1.is_valid == True
    assert len(result1.errors) == 0
    
    # 测试空响应
    result2 = ResponseValidator.validate_response(
        response="",
        question="什么是机器学习？",
        expected_length=(50, 200),
    )
    assert result2.is_valid == False
    assert len(result2.errors) > 0
    
    # 测试过短响应
    result3 = ResponseValidator.validate_response(
        response="是的",
        question="什么是机器学习？",
        expected_length=(50, 200),
    )
    assert result3.is_valid == False
    
    print("  [OK] 响应验证器验证成功")
except Exception as e:
    print(f"  [FAIL] 响应验证器验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 5: 人机模拟器
print("\n[5/5] 验证人机模拟器...")
try:
    import asyncio
    
    simulator = HumanSimulator(
        typing_speed=10.0,  # 快速打字
        pause_probability=0.0,  # 不停顿
    )
    
    # 测试批量输入
    async def test_simulator():
        text = "Hello World"
        result = await simulator.simulate_input_batch(text)
        assert result == text
        return True
    
    result = asyncio.run(test_simulator())
    assert result == True
    
    print("  [OK] 人机模拟器验证成功")
except Exception as e:
    print(f"  [FAIL] 人机模拟器验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n" + "=" * 70)
print("[SUCCESS] 所有验证通过！框架基本功能正常。")
print("=" * 70)

print("\n下一步:")
print("  1. 运行单机推理测试:")
print("     python run_simulation_tests.py simple_qa")
print()
print("  2. 列出所有可用场景:")
print("     python run_simulation_tests.py --list")
print()
print("  3. 运行所有场景测试:")
print("     python run_simulation_tests.py --all")
print()
print("详细信息请参阅: tests/simulation/README.md")
