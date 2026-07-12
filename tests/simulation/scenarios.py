"""
测试场景定义

定义各种测试场景，包括简单问答、多轮对话、复杂推理等。
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Scenario:
    """测试场景"""

    name: str
    description: str
    questions: List[str]
    expected_length: Tuple[int, int] = (50, 500)
    difficulty: str = "medium"  # "easy", "medium", "hard"

    def __str__(self) -> str:
        return f"{self.name}: {self.description} ({len(self.questions)} 个问题)"


# ============================================================================
# 预定义测试场景
# ============================================================================


SCENARIOS = {
    # ========================================================================
    # 简单问答场景
    # ========================================================================
    "simple_qa": Scenario(
        name="简单问答",
        description="基础的知识问答测试，验证模型的基本回答能力",
        questions=[
            "什么是机器学习？",
            "Python 有哪些优点？",
            "什么是神经网络？",
            "深度学习和机器学习有什么区别？",
            "什么是自然语言处理？",
        ],
        expected_length=(50, 200),
        difficulty="easy",
    ),

    # ========================================================================
    # 多轮对话场景
    # ========================================================================
    "multi_turn": Scenario(
        name="多轮对话",
        description="测试模型在多轮对话中的上下文理解能力",
        questions=[
            "请介绍一下深度学习",
            "它和传统机器学习有什么区别？",
            "能举个实际应用的例子吗？",
            "这些应用面临哪些挑战？",
            "未来发展趋势如何？",
        ],
        expected_length=(100, 300),
        difficulty="medium",
    ),

    # ========================================================================
    # 复杂推理场景
    # ========================================================================
    "complex_reasoning": Scenario(
        name="复杂推理",
        description="测试模型的逻辑推理和分析能力",
        questions=[
            "请分析分布式系统的优缺点",
            "在设计分布式推理系统时需要考虑哪些因素？",
            "如何平衡计算负载和网络通信？",
            "如果某个节点突然离线，系统应该如何处理？",
        ],
        expected_length=(200, 500),
        difficulty="hard",
    ),

    # ========================================================================
    # 代码生成场景
    # ========================================================================
    "code_generation": Scenario(
        name="代码生成",
        description="测试模型的代码生成能力",
        questions=[
            "写一个 Python 函数计算斐波那契数列",
            "用快速排序算法对数组排序",
            "实现一个简单的 HTTP 服务器",
            "写一个函数检查字符串是否是回文",
        ],
        expected_length=(150, 500),
        difficulty="medium",
    ),

    # ========================================================================
    # 技术解释场景
    # ========================================================================
    "technical_explanation": Scenario(
        name="技术解释",
        description="测试模型对技术概念的解釋能力",
        questions=[
            "解释一下什么是 Transformer 架构",
            "注意力机制是如何工作的？",
            "什么是量化？为什么要进行模型量化？",
            "解释一下 KV Cache 的作用",
            "什么是分布式推理？它有什么优势？",
        ],
        expected_length=(150, 400),
        difficulty="medium",
    ),

    # ========================================================================
    # 数学问题场景
    # ========================================================================
    "math_problems": Scenario(
        name="数学问题",
        description="测试模型的数学推理能力",
        questions=[
            "求解方程 2x + 5 = 15",
            "计算 1 到 100 的和",
            "解释什么是梯度下降",
            "什么是损失函数？",
        ],
        expected_length=(100, 300),
        difficulty="medium",
    ),

    # ========================================================================
    # 长文本生成场景
    # ========================================================================
    "long_generation": Scenario(
        name="长文本生成",
        description="测试模型的长文本生成能力",
        questions=[
            "写一篇关于人工智能发展的短文（约 500 字）",
            "描述一个分布式系统的应用场景",
            "写一份关于机器学习的入门指南",
        ],
        expected_length=(400, 800),
        difficulty="hard",
    ),

    # ========================================================================
    # 压力测试场景（短问题，快速回答）
    # ========================================================================
    "stress_test": Scenario(
        name="压力测试",
        description="用于压力测试的简短问题",
        questions=[
            "1+1等于多少？",
            "Python 是什么？",
            "什么是 AI？",
            "Hello?",
            "测试",
            "你好",
            "什么是 ML？",
            "什么是 DL？",
            "什么是 NLP？",
            "什么是 CV？",
        ],
        expected_length=(20, 100),
        difficulty="easy",
    ),
}


# ============================================================================
# 场景辅助函数
# ============================================================================


def get_scenario(name: str) -> Scenario:
    """获取指定名称的场景

    Args:
        name: 场景名称

    Returns:
        场景对象

    Raises:
        KeyError: 场景不存在
    """
    if name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise KeyError(f"场景 '{name}' 不存在。可用场景: {available}")

    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    """列出所有可用场景

    Returns:
        场景名称列表
    """
    return list(SCENARIOS.keys())


def get_scenarios_by_difficulty(difficulty: str) -> List[Scenario]:
    """按难度获取场景

    Args:
        difficulty: 难度级别 ("easy", "medium", "hard")

    Returns:
        场景列表
    """
    return [
        scenario for scenario in SCENARIOS.values()
        if scenario.difficulty == difficulty
    ]


def create_custom_scenario(
    name: str,
    description: str,
    questions: List[str],
    expected_length: Tuple[int, int] = (50, 500),
    difficulty: str = "medium",
) -> Scenario:
    """创建自定义场景

    Args:
        name: 场景名称
        description: 场景描述
        questions: 问题列表
        expected_length: 预期响应长度范围
        difficulty: 难度级别

    Returns:
        场景对象
    """
    return Scenario(
        name=name,
        description=description,
        questions=questions,
        expected_length=expected_length,
        difficulty=difficulty,
    )
