"""
仿真测试框架核心模块

包含测试编排、后端管理、请求发送、人机模拟等核心组件。
"""

import asyncio
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class TestConfig:
    """测试配置"""

    # 后端配置
    start_master: bool = True
    start_slaves: bool = False
    slave_count: int = 0
    model: str = "qwen-1.8b"

    # 端口配置
    master_api_port: int = 8000
    master_tcp_port: int = 8888
    slave_api_port_start: int = 8001

    # 超时配置
    startup_timeout: int = 30
    request_timeout: int = 60

    # 环境配置
    config_path: str = ".env"

    def __post_init__(self):
        """验证配置"""
        if self.start_slaves and self.slave_count <= 0:
            raise ValueError("start_slaves=True 时 slave_count 必须 > 0")


@dataclass
class ValidationResult:
    """验证结果"""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "[OK] 有效" if self.is_valid else "[FAIL] 无效"
        result = f"验证结果: {status}"
        if self.errors:
            result += f"\n  错误: {', '.join(self.errors)}"
        if self.warnings:
            result += f"\n  警告: {', '.join(self.warnings)}"
        return result


@dataclass
class TestResult:
    """测试结果"""

    # 基本信息
    test_name: str
    timestamp: str
    status: str  # "PASSED", "FAILED", "ERROR"

    # 请求信息
    question: str
    response: str

    # 性能指标
    latency: float  # 秒
    response_length: int

    # 验证结果
    validation: ValidationResult

    # 可选：分布式推理指标
    metrics: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "status": self.status,
            "question": self.question,
            "response": self.response,
            "latency": self.latency,
            "response_length": self.response_length,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "metrics": self.metrics,
        }


# ============================================================================
# 后端管理器
# ============================================================================


class BackendManager:
    """后端服务管理器"""

    def __init__(self):
        self.master_process: Optional[subprocess.Popen] = None
        self.slave_processes: List[subprocess.Popen] = []
        self.slave_api_ports: List[int] = []
        self.project_root = Path(__file__).parent.parent.parent

    async def start_master(
        self,
        config_path: str = ".env",
        api_port: int = 8000,
        tcp_port: int = 8888,
        startup_timeout: int = 30,
    ) -> None:
        """启动主节点后端

        Args:
            config_path: 配置文件路径
            api_port: API 服务端口
            tcp_port: TCP 通信端口
        """
        print(f"启动主节点后端 (API:{api_port}, TCP:{tcp_port})...")

        # 设置环境变量
        env = os.environ.copy()
        env["QLH_NODE_ROLE"] = "master"
        env["QLH_SERVER_PORT"] = str(tcp_port)
        env["QLH_API_PORT"] = str(api_port)

        # 构建启动命令
        api_server_path = self.project_root / "src" / "api_server.py"

        # 启动进程
        self.master_process = subprocess.Popen(
            [sys.executable, str(api_server_path)],
            env=env,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # 等待启动完成
        await self._wait_for_startup(
            api_port,
            timeout=startup_timeout,
            process=self.master_process,
            label="主节点",
        )
        print("[OK] 主节点后端启动成功")

    async def start_slave(
        self,
        master_host: str = "localhost",
        master_port: int = 8888,
        slave_api_port: int = 8001,
        startup_timeout: int = 30,
    ) -> None:
        """启动从节点后端

        Args:
            master_host: 主节点主机地址
            master_port: 主节点 TCP 端口
            slave_api_port: 从节点 API 端口
        """
        slave_id = len(self.slave_processes) + 1
        print(f"启动从节点 {slave_id} (API:{slave_api_port})...")

        # 设置环境变量
        env = os.environ.copy()
        env["QLH_NODE_ROLE"] = "slave"
        env["QLH_MASTER_HOST"] = master_host
        env["QLH_MASTER_PORT"] = str(master_port)
        env["QLH_API_PORT"] = str(slave_api_port)

        # 构建启动命令
        api_server_path = self.project_root / "src" / "api_server.py"

        # 启动进程
        process = subprocess.Popen(
            [sys.executable, str(api_server_path)],
            env=env,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.slave_processes.append(process)
        self.slave_api_ports.append(slave_api_port)

        # 等待从节点启动和注册
        await self._wait_for_startup(
            slave_api_port,
            timeout=startup_timeout,
            process=process,
            label=f"从节点 {slave_id}",
        )
        print(f"[OK] 从节点 {slave_id} 启动成功")

    async def _wait_for_startup(
        self,
        port: int,
        timeout: int = 30,
        process: Optional[subprocess.Popen] = None,
        label: str = "后端",
    ) -> None:
        """等待后端启动完成

        Args:
            port: API 端口
            timeout: 超时时间（秒）
        """
        start_time = time.time()
        health_url = f"http://localhost:{port}/api/health"

        while time.time() - start_time < timeout:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"{label}进程提前退出 (pid={process.pid}, code={process.returncode})"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            await asyncio.sleep(1)

        raise TimeoutError(f"后端启动超时（{timeout}秒，端口 {port}）")

    @staticmethod
    def _is_process_alive(process: Optional[subprocess.Popen]) -> bool:
        return process is not None and process.poll() is None

    def check_process_health(self) -> Dict:
        """检查已管理后端进程是否仍在运行。"""
        master_alive = self._is_process_alive(self.master_process)
        slaves = []
        for idx, process in enumerate(self.slave_processes):
            slaves.append({
                "index": idx,
                "pid": process.pid,
                "api_port": self.slave_api_ports[idx] if idx < len(self.slave_api_ports) else None,
                "alive": self._is_process_alive(process),
                "returncode": process.poll(),
            })
        return {
            "master": {
                "pid": self.master_process.pid if self.master_process else None,
                "alive": master_alive,
                "returncode": self.master_process.poll() if self.master_process else None,
            },
            "slaves": slaves,
            "all_alive": master_alive and all(item["alive"] for item in slaves),
        }

    @staticmethod
    def _terminate_process(process: subprocess.Popen, label: str) -> None:
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{label} 无法在 kill 后退出 (pid={process.pid})") from exc

    async def stop_all(self) -> None:
        """停止所有后端进程"""
        print("停止所有后端进程...")

        # 停止主节点
        if self.master_process:
            self._terminate_process(self.master_process, "主节点")
            self.master_process = None

        # 停止所有从节点
        for i, process in enumerate(self.slave_processes, 1):
            self._terminate_process(process, f"从节点 {i}")

        self.slave_processes.clear()
        self.slave_api_ports.clear()
        print("[OK] 所有后端进程已停止")

    async def stop_slave(self, slave_index: int) -> int:
        """停止指定从节点并返回其 API 端口。"""
        if slave_index < 0 or slave_index >= len(self.slave_processes):
            raise IndexError(f"从节点索引 {slave_index} 超出范围")

        process = self.slave_processes[slave_index]
        port = self.slave_api_ports[slave_index]
        print(f"停止从节点 {slave_index + 1} (PID: {process.pid}, API:{port})...")

        self._terminate_process(process, f"从节点 {slave_index + 1}")
        self.slave_processes.pop(slave_index)
        self.slave_api_ports.pop(slave_index)

        print(f"[OK] 从节点 {slave_index + 1} 已停止")
        return port

    def get_master_pid(self) -> Optional[int]:
        """获取主节点进程 ID"""
        return self.master_process.pid if self.master_process else None

    def get_slave_pids(self) -> List[int]:
        """获取所有从节点进程 ID"""
        return [p.pid for p in self.slave_processes]


# ============================================================================
# 请求发送器
# ============================================================================


class RequestSender:
    """HTTP 请求发送器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.aclose()
            self.session = None

    async def send_chat_request(
        self,
        message: str,
        session_id: str = "test-session",
        stream: bool = True,
    ) -> AsyncGenerator[Dict, None]:
        """发送聊天请求并处理响应

        Args:
            message: 用户消息
            session_id: 会话 ID
            stream: 是否使用流式响应

        Yields:
            响应片段（流式）或完整响应
        """
        if not self.session:
            raise RuntimeError("请使用 async with 语句初始化 RequestSender")

        endpoint = f"{self.base_url}/api/chat/stream" if stream else f"{self.base_url}/api/chat"

        payload = {
            "message": message,
            "session_id": session_id,
        }

        start_time = time.time()

        if stream:
            # 流式响应
            streamed_text = ""
            async with self.session.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])

                        # 检查是否有错误
                        if data.get("error"):
                            raise RuntimeError(f"推理错误: {data['error']}")

                        if data.get("done"):
                            final_content = data.get("response") or data.get("content") or ""
                            if final_content and not streamed_text:
                                yield {
                                    "type": "chunk",
                                    "content": final_content,
                                    "latency": time.time() - start_time,
                                    "metrics": data.get("metrics", {}),
                                }
                            continue

                        content = data.get("token") or data.get("content") or ""
                        if not content:
                            continue
                        streamed_text += content

                        yield {
                            "type": "chunk",
                            "content": content,
                            "latency": time.time() - start_time,
                        }
        else:
            # 非流式响应
            response = await self.session.post(endpoint, json=payload)
            response.raise_for_status()

            data = response.json()

            yield {
                "type": "complete",
                "content": data.get("content", ""),
                "latency": time.time() - start_time,
                "tokens": data.get("tokens_generated", 0),
            }

    async def check_health(self) -> Dict:
        """检查后端健康状态

        Returns:
            健康状态字典
        """
        if not self.session:
            raise RuntimeError("请使用 async with 语句初始化 RequestSender")

        response = await self.session.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()

    async def get_cluster_status(self) -> Dict:
        """获取集群状态

        Returns:
            集群状态字典
        """
        if not self.session:
            raise RuntimeError("请使用 async with 语句初始化 RequestSender")

        response = await self.session.get(f"{self.base_url}/api/cluster/status")
        response.raise_for_status()
        return response.json()

    async def get_layer_config(self) -> Dict:
        """获取分层配置

        Returns:
            分层配置字典
        """
        if not self.session:
            raise RuntimeError("请使用 async with 语句初始化 RequestSender")

        response = await self.session.get(f"{self.base_url}/api/cluster/layers")
        response.raise_for_status()
        return response.json()


# ============================================================================
# 人机模拟器
# ============================================================================


class HumanSimulator:
    """人机交互模拟器"""

    def __init__(
        self,
        typing_speed: float = 5.0,
        pause_probability: float = 0.3,
        pause_duration: Tuple[float, float] = (0.5, 2.0),
    ):
        """
        Args:
            typing_speed: 打字速度（字符/秒）
            pause_probability: 停顿概率（0-1）
            pause_duration: 停顿持续时间范围（秒）
        """
        self.typing_speed = typing_speed
        self.pause_probability = pause_probability
        self.pause_duration = pause_duration

    async def simulate_input(self, text: str) -> AsyncGenerator[str, None]:
        """模拟逐字符输入，包含随机停顿

        Args:
            text: 要输入的文本

        Yields:
            每次输入的字符
        """
        for i, char in enumerate(text):
            # 模拟打字延迟
            delay = 1.0 / self.typing_speed

            # 随机停顿（模拟思考）
            if random.random() < self.pause_probability:
                pause_time = random.uniform(*self.pause_duration)
                delay += pause_time

            await asyncio.sleep(delay)
            yield char

    async def simulate_input_batch(self, text: str) -> str:
        """模拟输入并返回完整文本（不使用生成器）

        Args:
            text: 要输入的文本

        Returns:
            完整文本
        """
        full_input = ""
        async for char in self.simulate_input(text):
            full_input += char

        return full_input

    async def simulate_modification(
        self,
        original: str,
        modified: str,
    ) -> AsyncGenerator[Dict, None]:
        """模拟用户修改输入（删除 + 重新输入）

        Args:
            original: 原始文本
            modified: 修改后的文本

        Yields:
            操作字典 {"action": "delete"} 或 {"action": "input", "char": char}
        """
        # 计算公共前缀
        common_length = 0
        for i, (c1, c2) in enumerate(zip(original, modified)):
            if c1 == c2:
                common_length += 1
            else:
                break

        # 删除多余字符
        delete_count = len(original) - common_length
        for _ in range(delete_count):
            yield {"action": "delete"}
            await asyncio.sleep(0.1)

        # 输入新字符
        new_part = modified[common_length:]
        for char in new_part:
            yield {"action": "input", "char": char}
            await asyncio.sleep(1.0 / self.typing_speed)


# ============================================================================
# 响应验证器
# ============================================================================


class ResponseValidator:
    """响应验证器"""

    @staticmethod
    def validate_response(
        response: str,
        question: str,
        expected_length: Optional[Tuple[int, int]] = (50, 500),
    ) -> ValidationResult:
        """验证响应质量

        Args:
            response: 响应文本
            question: 问题文本
            expected_length: 预期长度范围

        Returns:
            验证结果
        """
        errors = []
        warnings = []

        # 1. 空响应检查
        if not response or response.strip() == "":
            errors.append("响应为空")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 2. 长度验证
        if expected_length:
            if len(response) < expected_length[0]:
                errors.append(f"响应过短: {len(response)} < {expected_length[0]}")
            elif len(response) > expected_length[1]:
                warnings.append(f"响应过长: {len(response)} > {expected_length[1]}")

        # 3. 错误信息检查
        error_keywords = ["error", "错误", "exception", "异常", "failed", "失败"]
        response_lower = response.lower()

        for keyword in error_keywords:
            if keyword in response_lower:
                warnings.append(f"响应包含错误关键词: {keyword}")
                break

        # 4. 无法回答检查
        unable_keywords = ["无法回答", "不知道", "不清楚", "无法确定", "I don't know", "I'm not sure"]
        for keyword in unable_keywords:
            if keyword in response_lower:
                warnings.append(f"模型表示无法回答: {keyword}")
                break

        # 5. 相关性验证（简单关键词匹配）
        question_keywords = ResponseValidator._extract_keywords(question)
        if question_keywords:
            keyword_match_count = sum(
                1 for kw in question_keywords if kw in response_lower
            )

            if keyword_match_count == 0:
                warnings.append("响应与问题关键词无匹配")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def validate_distributed_inference(
        metrics: Dict,
        expected_nodes: int,
    ) -> ValidationResult:
        """验证分布式推理执行

        Args:
            metrics: 推理指标
            expected_nodes: 预期参与节点数

        Returns:
            验证结果
        """
        errors = []
        warnings = []

        # 1. 节点参与数
        nodes_involved = metrics.get("nodes_involved", 0)
        if nodes_involved != expected_nodes:
            errors.append(
                f"参与节点数不符: {nodes_involved} != {expected_nodes}"
            )

        # 2. 层前向传播
        layer_forward_count = metrics.get("layer_forward_count", 0)
        if layer_forward_count == 0:
            errors.append("未执行层前向传播")

        # 3. 网络流量
        network_traffic = metrics.get("network_traffic", 0)
        if network_traffic == 0:
            warnings.append("未检测到网络流量")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """提取文本关键词（简单实现）

        Args:
            text: 输入文本

        Returns:
            关键词列表
        """
        # 简单的关键词提取：分词后过滤停用词
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "什么", "怎么", "为什么", "哪", "谁", "多少", "几",
        }

        # 简单分词（实际项目中应使用 jieba 等分词库）
        words = text.replace("？", "").replace("！", "").replace("。", "").split()

        # 过滤停用词和短词
        keywords = [
            word for word in words
            if word not in stopwords and len(word) > 1
        ]

        return keywords[:5]  # 最多返回5个关键词


# ============================================================================
# 测试编排器
# ============================================================================


class TestOrchestrator:
    """测试编排器"""

    def __init__(self):
        self.backend_manager = BackendManager()
        self.config: Optional[TestConfig] = None

    async def setup(self, config: TestConfig) -> None:
        """设置测试环境

        Args:
            config: 测试配置
        """
        print("=" * 60)
        print("开始设置测试环境")
        print("=" * 60)

        self.config = config

        # 启动主节点
        if config.start_master:
            await self.backend_manager.start_master(
                config_path=config.config_path,
                api_port=config.master_api_port,
                tcp_port=config.master_tcp_port,
                startup_timeout=config.startup_timeout,
            )

        # 启动从节点
        if config.start_slaves:
            for i in range(config.slave_count):
                slave_api_port = config.slave_api_port_start + i
                await self.backend_manager.start_slave(
                    master_host="localhost",
                    master_port=config.master_tcp_port,
                    slave_api_port=slave_api_port,
                    startup_timeout=config.startup_timeout,
                )

        process_health = self.backend_manager.check_process_health()
        if config.start_master and not process_health["master"]["alive"]:
            raise RuntimeError(f"主节点进程健康检查失败: {process_health['master']}")
        dead_slaves = [
            slave for slave in process_health["slaves"]
            if not slave["alive"]
        ]
        if dead_slaves:
            raise RuntimeError(f"从节点进程健康检查失败: {dead_slaves}")

        print("=" * 60)
        print("[OK] 测试环境设置完成")
        print("=" * 60)

    async def teardown(self) -> None:
        """清理测试环境"""
        print("\n" + "=" * 60)
        print("开始清理测试环境")
        print("=" * 60)

        await self.backend_manager.stop_all()

        print("=" * 60)
        print("[OK] 测试环境清理完成")
        print("=" * 60)

    async def stop_slave(self, slave_index: int) -> int:
        """停止指定从节点

        Args:
            slave_index: 从节点索引（从0开始）

        Returns:
            被停止从节点的 API 端口
        """
        return await self.backend_manager.stop_slave(slave_index)

    async def restart_slave(self, slave_index: int, slave_api_port: Optional[int] = None) -> None:
        """重启指定从节点

        Args:
            slave_index: 从节点索引（从0开始）
            slave_api_port: 从节点 API 端口；未提供时按初始配置和索引计算
        """
        if not self.config:
            raise RuntimeError("测试配置未设置")

        print(f"重启从节点 {slave_index + 1}...")

        # 启动新的从节点
        target_port = slave_api_port or (self.config.slave_api_port_start + slave_index)
        await self.backend_manager.start_slave(
            master_host="localhost",
            master_port=self.config.master_tcp_port,
            slave_api_port=target_port,
            startup_timeout=self.config.startup_timeout,
        )

        print(f"[OK] 从节点 {slave_index + 1} 已重启")
