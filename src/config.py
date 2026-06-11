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

# ============================================================
# 1. 网络配置
# ============================================================
SERVER_IP = "192.168.x.x"       # 主节点监听IP（部署时改为实际IP）
SERVER_PORT = 8888              # 主节点监听端口
HEARTBEAT_INTERVAL = 3          # 心跳包间隔（秒）
RECONNECT_MAX_RETRIES = 5       # 断线重连最大尝试次数
RECONNECT_DELAY = 2             # 重连间隔（秒）

# ============================================================
# 2. 模型配置
# ============================================================
MODEL_NAME = "Qwen/Qwen-1.8B-Chat"          # HuggingFace 模型标识
MODEL_PATH = "./models/qwen-1_8b-chat"       # 本地模型存放路径（已验证）
QUANT_TYPE = "int4"                          # 量化精度: "fp16" | "int8" | "int4"
USE_COMPILE = False                          # 算子融合（仅FP16有效，INT4下自动跳过）
DEVICE = "cuda"                              # 推理设备: "cuda" | "cpu"
TRUST_REMOTE_CODE = True                     # Qwen 模型需要自定义代码

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
DEFAULT_LAYER_CONFIG = {
    "master":  (0, 8),                   # 主节点: Embedding + Layer 0-7
    "client1": (8, 16),                  # 从节点1: Layer 8-15
    "client2": (16, 24),                 # 从节点2: Layer 16-23 + LM Head
}

# ============================================================
# 5. 节点身份配置
# ============================================================
NODE_ROLE = "master"             # 节点角色: "master" 主节点 | "client" 从节点
NODE_ID = "master"               # 节点唯一标识（master 固定为 "master"，client 按 hostname 自动生成）
MAX_NODES = 3                    # 最大节点数上限（主节点可动态调整，仅限已注册节点，不含空位）
                                  # 从节点通过 TCP 注册后自动加入列表，不再预创建空槽位

# 从节点连接主节点配置（仅 NODE_ROLE="client" 时生效）
CLIENT_MASTER_HOST = "192.168.x.x"   # 主节点 IP 地址（部署时改为实际 IP）
CLIENT_MASTER_PORT = 8888            # 主节点监听端口

# SMTP 邮件告警配置（详见 src/email_notifier.py）
MASTER_DOWN_EMAIL_TIMEOUT = 180      # 主节点宕机超过此秒数（3分钟）后发送邮件告警（0=禁用）

# ============================================================
# 6. 运行模式
# ============================================================
RUN_MODE = "distributed"         # "single" 单机 | "distributed" 分布式
LOG_LEVEL = "INFO"               # 日志级别: DEBUG | INFO | WARNING | ERROR
LOG_DIR = "./logs"               # 日志文件目录

# ============================================================
# 6. 可视化配置
# ============================================================
WEB_HOST = "0.0.0.0"            # Streamlit 绑定地址
WEB_PORT = 8501                 # Streamlit 端口


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
