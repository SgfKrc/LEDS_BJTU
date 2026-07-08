"""
全局配置文件 — 统一参数，一处修改全局生效
============================================
集中所有硬编码参数，便于调试、切换模式。

量化模式对照表（已验证，Qwen-1.8B-Chat + RTX GPU + PyTorch 2.12，统一50 token基准）:
    fp16:            显存 3.47 GB | 推理 53.2 tok/s | 精度无损
    fp16 + compile:  显存 3.47 GB | 推理 55.1 tok/s | 算子融合 +3.6%
    int8:            显存 2.30 GB | 推理  9.8 tok/s | 精度微损 (compile 不兼容)
    int4:            显存 1.75 GB | 推理 28.7 tok/s | 精度略降 (compile 不兼容，慢13%)

算子融合兼容性:
    - FP16: compile 有效 (+3.6%)，合并 LayerNorm→Linear→GELU 等连续算子
    - INT4/INT8: compile 无效 (-13%)，bitsandbytes CUDA kernel 绕过原生算子
    - 推荐: INT4 不加 compile（边缘设备优先省显存，1.75 GB vs 3.47 GB）
"""

import os
import sys


def _get_app_root() -> str:
    """
    获取应用根目录（模型、日志等文件的基准路径）。

    PyInstaller 打包: exe 所在目录（models/ 与 exe 同级）
    开发模式:        src/../ 即项目根目录
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 开发模式：此文件在 src/ 下 → 项目根 = src/../
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_APP_ROOT = _get_app_root()


# ============================================================
# 1. 网络配置
# ============================================================
SERVER_IP = "192.168.x.x"       # 主节点监听IP（部署时改为实际IP）
SERVER_PORT = 8888              # 主节点监听端口
HEARTBEAT_INTERVAL = 3          # 心跳包间隔（秒）
RECONNECT_MAX_RETRIES = 5       # 断线重连最大尝试次数
RECONNECT_DELAY = 2             # 重连间隔（秒）

# ============================================================
# 2. 模型配置（绝对路径，兼容 PyInstaller 打包 + 开发模式）
# ============================================================
MODEL_NAME = "Qwen/Qwen-1.8B-Chat"          # HuggingFace 模型标识
MODEL_PATH = os.path.join(_APP_ROOT, "models", "qwen-1_8b-chat")       # Safetensors 格式
GGUF_MODEL_PATH = os.path.join(_APP_ROOT, "models", "Qwen-1_8B-Chat.Q4_K_M.gguf")  # GGUF 格式
QUANT_TYPE = "int4"                          # 量化精度: "fp16" | "int8" | "int4"
USE_COMPILE = False                          # 算子融合（仅FP16有效，INT4下自动跳过）
DEVICE = "cuda"                              # 推理设备: "cuda" | "cpu"

# --- 推理引擎选择 ---
# "auto": 自动选择 — CUDA 可用 → PyTorch + bitsandbytes, 否则 → llama.cpp + GGUF
# "pytorch": 强制 PyTorch + Transformers（需要 CUDA 或大内存 CPU）
# "llama_cpp": 强制 llama.cpp + GGUF（推荐 CPU / 集显设备）
INFERENCE_ENGINE = "llama_cpp"
# GGUF 量化文件推荐 (RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf on HuggingFace):
#   Q4_K_M (1.16 GB) — 推荐，速度/质量最佳平衡
#   Q5_K_M (1.31 GB) — 更高质量
#   Q8_0   (1.82 GB) — 近无损

TRUST_REMOTE_CODE = True                     # Qwen 模型需要自定义代码

# --- 多模型实验支持 (P3) ---
ACTIVE_MODEL_ID = "qwen-1_8b"            # 当前活跃的模型 ID
EXPERIMENTAL_MODELS_ENABLED = False      # 运行时由 CUDA 检测设置（torch.cuda.is_available()）

# ============================================================
# 3. 分页KV缓存配置
# ============================================================
PAGE_SIZE = 128                  # 单页容纳 Token 数量
MAX_PAGE_NUM = 256               # 最大内存页数
MAX_SEQ_LEN = 4096               # 最大序列长度（Qwen-1.8B 原生 8K，laptop 档用 4K）

# ============================================================
# 4. 模型分层配置（动态分配，运行时由 scheduler 根据设备画像计算）
#    Qwen-1.8B-Chat 共 24 层 Transformer
# ============================================================
TOTAL_MODEL_LAYERS = 24                  # Qwen-1.8B-Chat Transformer 层总数
LAYER_STRATEGY = "dynamic"              # 分层策略: "dynamic" 动态 | "manual" 手动覆盖
DISTRIBUTED_INFERENCE_ENABLED = True    # 分布式推理开关（主节点默认开启，从节点默认关闭）

# 回退分层配置（当没有从节点注册时使用）
# 注意：分层现在由 compute_layer_assignment() 动态计算，此配置仅供文档参考。
# 实际部署中不依赖硬编码的 client1/client2 槽位。
DEFAULT_LAYER_CONFIG = {}

# 显存约束：单层 Transformer 最低显存 + 量化修正系数
# Qwen-1.8B 每层 ~70MB (FP16)，Embedding + LM Head 各 ~580MB
MIN_VRAM_PER_LAYER_MB = 70               # 单层最低显存（FP16 基准）
EMBEDDING_VRAM_MB = 580                  # Token Embedding 层
LM_HEAD_VRAM_MB = 580                    # LM Head 输出层
SAFE_VRAM_MARGIN = 1.1                   # 10% 安全余量
LAYER_VRAM_FACTOR = {                    # 量化精度修正系数
    "fp16": 1.0,
    "int8": 0.55,
    "int4": 0.35,
}

# 图算法智能编排阈值：节点数超过此值（>5）时自动启用最大带宽生成树 + DFS，
# 替代纯算力权重分配；节点数 ≤ 阈值时回退到简单排序（权重比例分配）
GRAPH_ORCHESTRATOR_THRESHOLD = 5         # 节点数 > 5 启用图算法，≤ 5 使用简单排序

# 流水线推理超时与并发控制
PIPELINE_TIMEOUT = 120                   # 流水线单步超时（秒），含网络传输 + 前向计算
PIPELINE_MAX_CONCURRENT = 1              # 最大并发流水线任务数（当前仅支持 1，串行执行）
PIPELINE_STEP_TIMEOUT = 30               # 单个节点前向传播超时（秒）
PIPELINE_QUEUE_MAX_SIZE = 100            # 请求队列最大容量（超出返回 503）
PIPELINE_QUEUE_RESULT_TTL = 300          # 已完成任务结果保留时间（秒），超时清理
PIPELINE_QUEUE_POLL_INTERVAL = 0.5       # 排队请求轮询间隔（秒）

# ============================================================
# 4.5 推理调度策略（MLFQ 三级反馈队列）
# ============================================================
PIPELINE_SCHEDULING_STRATEGY = "mlfq"     # 调度策略: "fifo" | "mlfq"

# 三级队列阈值（按 max_new_tokens 分级）
PIPELINE_Q0_MAX_TOKENS = 128              # Q0 交互级上限 (≤128)
PIPELINE_Q1_MAX_TOKENS = 512              # Q1 普通级上限 (≤512，>Q1→Q2 批量级)

# 老化提升参数（防饥饿）
PIPELINE_AGING_Q1_TO_Q0_SECONDS = 60      # Q1 等待 > 60s → 提升到 Q0
PIPELINE_AGING_Q2_TO_Q1_SECONDS = 120     # Q2 等待 > 120s → 提升到 Q1
PIPELINE_AGING_MAX_WAIT_SECONDS = 300     # 绝对上限: 等待 > 300s → 强制置顶 Q0

# 协同抢占控制（二期实施，参数预留）
PIPELINE_PREEMPT_ENABLED = True           # 是否启用协同抢占（仅 Q0 可抢占 Q1/Q2）
PIPELINE_PREEMPT_MIN_INTERVAL = 10.0      # 两次抢占最小间隔（防抖动）
PIPELINE_PREEMPT_MIN_TOKENS = 16          # 至少生成 N token 后才接受抢占
PIPELINE_PREEMPT_MAX_OVERHEAD_MS = 500    # checkpoint+restore 超过此值 → 禁用抢占

# ============================================================
# 5. 节点身份配置
# ============================================================
NODE_ROLE = "master"             # 节点角色: "master" 主节点 | "client" 从节点
NODE_ID = "master"               # 节点唯一标识（master 固定为 "master"，client 按 hostname 自动生成）
MAX_NODES = 3                    # 最大节点数上限（主节点可动态调整，仅限已注册节点，不含空位）
                                  # 从节点通过 TCP 注册后自动加入列表，不再预创建空槽位

# 从节点连接主节点配置（仅 NODE_ROLE="client" 时生效）
# 主节点启动后自动检测 Tailscale/ZeroTier 组网 IP 并写入共享数据库，
# 从节点通过 discover_master() 自动发现即可，通常无需手动配置。
# 仅当数据库不可用时才回退到此配置值。
CLIENT_MASTER_HOST = "100.90.76.108"  # 主节点 IP 地址（Tailscale/ZeroTier IP）
CLIENT_MASTER_PORT = 8888            # 主节点监听端口

# SMTP 邮件告警配置（详见 src/email_notifier.py）
MASTER_DOWN_EMAIL_TIMEOUT = 180      # 主节点宕机超过此秒数（3分钟）后发送邮件告警（0=禁用）

# P3: 主节点转让审查配置（详见 src/review.py）
REVIEW_TIMEOUT_HOURS = 48            # 审查工单超时时间（小时）
REVIEW_APPROVE_THRESHOLD = 2         # 通过阈值: score >= +2
REVIEW_REJECT_THRESHOLD = -2         # 阻止阈值: score <= -2

# ============================================================
# 6. 集群安全
# ============================================================
# 集群共享密钥 — 所有节点必须使用相同密钥才能加入集群
# 通过环境变量 QLH_CLUSTER_SECRET 设置，不设置时使用默认值（仅限开发环境）
# 生产部署时务必修改为随机字符串（建议 32+ 字符）
CLUSTER_SECRET = os.environ.get(
    "QLH_CLUSTER_SECRET",
    "qlh-default-cluster-secret-change-me-in-production"
)
# HMAC 签名时间窗口（秒）— 防重放攻击
AUTH_TIMESTAMP_WINDOW = 300  # ±5 分钟

# ============================================================
# 7. 运行模式
# ============================================================
RUN_MODE = "distributed"         # "single" 单机 | "distributed" 分布式
LOG_LEVEL = "INFO"               # 日志级别: DEBUG | INFO | WARNING | ERROR
LOG_DIR = os.path.join(_APP_ROOT, "logs")  # 日志文件目录（绝对路径）

# ============================================================
# 6. 可视化配置（Streamlit 已移除，统一使用 FastAPI + React 前端）
# ============================================================
# 前端静态文件由 FastAPI 在端口 8000 直接提供服务（生产模式）
# 开发模式：Vite dev server 在 5173 端口，通过 proxy 转发 /api 到 8000


# ============================================================
# 7. 自适应配置（由 device_profiler 运行时生成）
# ============================================================

def auto_config(profile: dict = None) -> dict:
    """
    根据设备画像生成自适应推理配置。

    传入 device_profiler.DeviceProfiler.to_dict() 返回的设备画像 dict，
    或传入 None 使用保守默认值（laptop 档）。

    返回的 dict 字段与上方硬编码常量一一对应，可直接用于覆盖。

    Example:
        from device_profiler import DeviceProfiler
        p = DeviceProfiler()
        cfg = auto_config(p.to_dict())
        QUANT_TYPE = cfg["quant_type"]
    """
    if profile is None:
        # 保守默认：laptop 档
        return {
            "quant_type": "int4",
            "page_size": 128,
            "max_page_num": 256,
            "max_seq_len": 4096,
            "max_new_tokens": 1024,
            "use_compile": False,
            "device": "cuda",
            "description": "默认配置 (laptop 档)，未检测到设备画像",
        }

    tier = profile.get("tier", "laptop")
    gpu = profile.get("gpu", {})
    ram = profile.get("ram", {})
    platform_info = profile.get("platform", {})

    has_cuda = gpu.get("cuda_available", False) if gpu else False
    vram_gb = gpu.get("vram_total_gb", 0) if gpu else 0
    ram_gb = ram.get("total_gb", 8) if ram else 8
    is_arm = (platform_info.get("machine", "") in ("aarch64", "armv7l", "arm64")
              if platform_info else False)

    # 按档位返回配置
    if tier == "workstation":
        return {
            "quant_type": "fp16",
            "page_size": 128,
            "max_page_num": 512,
            "max_seq_len": 8192,      # 原生 8K 上下文
            "max_new_tokens": 2048,
            "use_compile": True,
            "device": "cuda" if has_cuda else "cpu",
            "description": "桌面工作站 — FP16 原版 + compile 融合",
        }
    elif tier == "laptop":
        return {
            "quant_type": "int4",
            "page_size": 128,
            "max_page_num": 256,
            "max_seq_len": 4096,      # 8GB VRAM 可容纳约 7K token KV 缓存
            "max_new_tokens": 1024,
            "use_compile": False,
            "device": "cuda" if has_cuda else "cpu",
            "description": "游戏本 / 独显本 — INT4 量化",
        }
    elif tier == "ultrabook":
        return {
            "quant_type": "int4",
            "page_size": 64,
            "max_page_num": 128,
            "max_seq_len": 2048,
            "max_new_tokens": 512,
            "use_compile": False,
            "device": "cuda" if has_cuda else "cpu",
            "description": "轻薄本 / 集显本 — INT4 + 缩减 KV 缓存",
        }
    elif tier == "edge":
        return {
            "quant_type": "int4",
            "page_size": 64,
            "max_page_num": 64,
            "max_seq_len": 1024,
            "max_new_tokens": 256,
            "use_compile": False,
            "device": "cpu",
            "description": "边缘设备 — CPU-only + 最小 KV 缓存",
        }
    elif tier == "mobile":
        return {
            "quant_type": "int4",
            "page_size": 32,
            "max_page_num": 32,
            "max_seq_len": 512,
            "max_new_tokens": 128,
            "use_compile": False,
            "device": "cpu",
            "description": "移动设备 — 极限压缩（建议导出 ONNX/GGUF）",
        }
    else:
        # 兜底
        return {
            "quant_type": "int4",
            "page_size": 128,
            "max_page_num": 256,
            "max_seq_len": 4096,
            "max_new_tokens": 1024,
            "use_compile": False,
            "device": "cuda" if has_cuda else "cpu",
            "description": "未知设备 — 保守配置",
        }
