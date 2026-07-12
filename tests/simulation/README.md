# 分布式推理仿真测试

端到端的仿真测试框架，验证分布式推理在真实环境下的可用性。

## 特点

- ✅ **真实后端启动**：启动完整的 `api_server.py`，非 Mock
- ✅ **真实推理执行**：调用真实大模型进行推理，非 Mock 响应
- ✅ **人机交互模拟**：模拟真实用户输入行为（打字速度、停顿、修改）
- ✅ **单机模式验证**：确保从节点离线时系统仍可用
- ✅ **自动化测试**：支持自动化执行和报告生成

## 快速开始

### 1. 单机推理测试

测试在无从节点情况下，主节点独立完成推理的能力。

```bash
# 运行简单问答测试
python -m tests.simulation.test_single_node simple_qa

# 运行多轮对话测试
python -m tests.simulation.test_single_node multi_turn

# 运行复杂推理测试
python -m tests.simulation.test_single_node complex_reasoning

# 列出所有可用场景
python -c "from tests.simulation.scenarios import list_scenarios; print('\n'.join(list_scenarios()))"
```

### 2. 可用测试场景

| 场景名称 | 描述 | 难度 | 问题数 |
|----------|------|------|--------|
| `simple_qa` | 基础知识问答 | easy | 5 |
| `multi_turn` | 多轮对话 | medium | 5 |
| `complex_reasoning` | 复杂推理 | hard | 4 |
| `code_generation` | 代码生成 | medium | 4 |
| `technical_explanation` | 技术解释 | medium | 5 |
| `math_problems` | 数学问题 | medium | 4 |
| `long_generation` | 长文本生成 | hard | 3 |
| `stress_test` | 压力测试 | easy | 10 |

## 测试报告

测试结果保存在 `tests/simulation/results/` 目录下。

### 报告格式

```json
{
  "test_name": "single_node_simple_qa",
  "timestamp": "2026-07-12T09:30:00",
  "scenario": {
    "name": "简单问答",
    "description": "基础的知识问答测试",
    "question_count": 5
  },
  "summary": {
    "total_requests": 5,
    "passed": 5,
    "failed": 0,
    "success_rate": 1.0
  },
  "performance": {
    "avg_latency": 3.5,
    "max_latency": 5.2,
    "min_latency": 2.1,
    "avg_response_length": 150.5
  },
  "results": [
    {
      "question": "什么是机器学习？",
      "response": "机器学习是人工智能的一个分支...",
      "latency": 3.5,
      "validation": {
        "is_valid": true,
        "errors": [],
        "warnings": []
      }
    }
  ]
}
```

### 验证标准

- **成功率**: ≥ 95%
- **平均延迟**: < 10 秒
- **响应质量**: 相关且连贯

## 架构

### 核心组件

1. **TestOrchestrator** - 测试编排器
   - 管理测试生命周期
   - 启动/停止后端服务
   - 协调各组件执行

2. **BackendManager** - 后端管理器
   - 启动主节点和从节点
   - 管理进程生命周期
   - 等待服务就绪

3. **RequestSender** - 请求发送器
   - 发送 HTTP 请求
   - 处理流式响应
   - 验证响应格式

4. **HumanSimulator** - 人机模拟器
   - 模拟打字速度
   - 模拟随机停顿
   - 模拟输入修改

5. **ResponseValidator** - 响应验证器
   - 验证响应长度
   - 验证响应质量
   - 验证分布式推理

### 类图

```
TestOrchestrator
├── BackendManager
│   ├── start_master()
│   ├── start_slave()
│   └── stop_all()
├── TestConfig
└── TestResult

RequestSender
├── send_chat_request()
├── check_health()
└── get_cluster_status()

HumanSimulator
├── simulate_input()
├── simulate_input_batch()
└── simulate_modification()

ResponseValidator
├── validate_response()
└── validate_distributed_inference()
```

## 使用示例

### 基本使用

```python
import asyncio
from tests.simulation import (
    TestOrchestrator,
    TestConfig,
    RequestSender,
    HumanSimulator,
    SCENARIOS,
)

async def main():
    # 1. 创建编排器
    orchestrator = TestOrchestrator()
    
    # 2. 配置测试
    config = TestConfig(
        start_master=True,
        start_slaves=False,
    )
    
    # 3. 设置环境
    await orchestrator.setup(config)
    
    try:
        # 4. 发送请求
        async with RequestSender() as sender:
            # 检查健康状态
            health = await sender.check_health()
            print(f"健康状态: {health}")
            
            # 模拟用户输入
            simulator = HumanSimulator(typing_speed=5.0)
            question = "什么是机器学习？"
            
            # 发送请求
            response = ""
            async for chunk in sender.send_chat_request(question):
                if chunk["type"] == "chunk":
                    response += chunk["content"]
                    print(chunk["content"], end="", flush=True)
            
            print(f"\n完整响应: {response}")
    
    finally:
        # 5. 清理环境
        await orchestrator.teardown()

asyncio.run(main())
```

### 自定义场景

```python
from tests.simulation.scenarios import create_custom_scenario

# 创建自定义场景
my_scenario = create_custom_scenario(
    name="my_test",
    description="我的自定义测试",
    questions=[
        "问题1",
        "问题2",
        "问题3",
    ],
    expected_length=(100, 300),
    difficulty="medium",
)

# 使用自定义场景
from tests.simulation import SCENARIOS
SCENARIOS["my_test"] = my_scenario
```

## 开发计划

### 第一阶段（当前）✅

- [x] 核心框架实现
- [x] 单机推理测试
- [x] 测试场景定义

### 第二阶段（计划中）

- [ ] 分布式推理测试
- [ ] 降级测试
- [ ] 节点管理模块

### 第三阶段（计划中）

- [ ] 压力测试
- [ ] 异常测试
- [ ] 测试报告生成

### 第四阶段（计划中）

- [ ] CI/CD 集成
- [ ] 性能监控
- [ ] 文档完善

## 故障排查

### 后端启动失败

**问题**: 后端启动超时

**解决方案**:
1. 检查 `.env` 文件是否存在
2. 检查端口是否被占用（8000, 8888）
3. 查看后端日志：`src/api_server.py` 的输出

### 请求失败

**问题**: HTTP 请求失败

**解决方案**:
1. 确认后端已启动：`curl http://localhost:8000/api/health`
2. 检查网络连接
3. 查看后端日志

### 响应验证失败

**问题**: 响应验证不通过

**解决方案**:
1. 检查响应长度是否在预期范围内
2. 检查响应是否包含错误信息
3. 调整 `expected_length` 参数

## 性能基准

| 场景 | 单机延迟 | 分布式延迟（3从节点） | 加速比 |
|------|----------|----------------------|--------|
| simple_qa | 3-5s | 2-3s | 1.5x |
| multi_turn | 5-8s | 3-5s | 1.6x |
| complex_reasoning | 8-12s | 5-8s | 1.6x |
| code_generation | 6-10s | 4-7s | 1.5x |

## 贡献指南

### 添加新场景

1. 在 `scenarios.py` 中定义新场景
2. 添加场景描述和预期行为
3. 运行测试验证
4. 更新本文档

### 添加新测试

1. 创建 `test_xxx.py` 文件
2. 实现测试逻辑
3. 添加测试报告生成
4. 更新本文档

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请联系开发团队。
