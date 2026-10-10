"""tests/conftest.py — 测试会话级准备

目前只做一件事：**尽早**安装 transformers 5.x 兼容 shim。

为什么需要在这里（而不是只靠 `src/model_module.py`）：
`transformers_stream_generator`（Qwen-1.8B remote code 的 `chat_stream` 依赖）在
transformers 5.x 下无法导入，而 transformers 5.x 的 `dynamic_module_utils.check_imports`
会在加载**任何** remote code 时把它列为待导入依赖，**非 "No module named" 的 ImportError 会直接抛**。

`src/model_module.py` 里已经装了一次（生产入口够用），但测试里有些用例**不经过**
`model_module` 就自己去 `from_pretrained(...)`（例如直接构造 remote code 模型的用例），
在那种执行顺序下 shim 尚未生效 ⇒ 会偶发 `ImportError: cannot import name
'DisjunctiveConstraint'`。会话最初装一次即可消除这种顺序依赖。

注意：**不要把它放到 `src/config.py`** —— 入口模块会导入 config，而
`test_entry_module_no_heavy_imports` 明确要求入口顶层不得拉入 transformers。
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--qlh-order-seed",
        action="store",
        type=int,
        default=None,
        help="deterministically shuffle collected tests with this seed",
    )


def pytest_collection_modifyitems(config, items) -> None:
    seed = config.getoption("--qlh-order-seed")
    if seed is not None:
        random.Random(seed).shuffle(items)


def pytest_report_header(config) -> str:
    seed = config.getoption("--qlh-order-seed")
    return f"qlh test order seed: {seed}" if seed is not None else ""

try:  # 尽量早地装上；失败也不应让整个测试会话崩掉
    from transformers5_compat import install as _install_transformers5_compat

    _install_transformers5_compat()
except Exception:  # noqa: BLE001 - 环境缺 transformers 时静默跳过（相关用例会自行跳过）
    pass


def pytest_runtest_setup(item) -> None:  # noqa: ARG001 - 需要 item 签名
    """每个用例开始前，确保 transformers 的 5.x 兼容符号在位。

    为什么需要：有测试会**动全局的 transformers** —— 例如
    `tests/test_island_engine.py` 为在无 transformers 的容器里导入 `model_module`
    而把 `sys.modules["transformers"]` 换成桩；也可能有用例删/替换模块属性。
    这类改动若未彻底复原，会让后续用例（如
    `test_inference_service_protocol::test_build_app_client_role_gates_chat`，它经
    `inference_svc_main → model_module` 真的去加载 remote code）看到
    `cannot import name 'DisjunctiveConstraint'` —— 看起来像 transformers 版本问题，
    实为**测试间污染**。

    ⚠️ 判定条件必须基于「**shim 符号是否在位**」，而不是「模块是否存在/是不是桩」：
    实测被污染时 `transformers` 仍是真模块（有 `__version__`），只是少了那几个符号。

    正常情况下这是廉价 no-op。
    """
    import sys as _sys

    module = _sys.modules.get("transformers")
    if module is not None and hasattr(module, "DisjunctiveConstraint"):
        return  # 符号在位 ⇒ 无需干预
    # 情况一：模块被换成桩 ⇒ 清掉相关缓存，重新导入真模块
    if module is not None and not hasattr(module, "__version__"):
        _sys.modules.pop("transformers", None)
        _sys.modules.pop("model_module", None)
    try:
        from transformers5_compat import install as _install

        _install()  # 情况二：真模块但符号被删 ⇒ 直接补回
    except Exception:  # noqa: BLE001 - 兜底失败不应中断用例
        pass


#: 运行期会被生产代码/用例直接改写的全局 config 名（详见下方 fixture 说明）。
_CONFIG_BASELINE_NAMES = ("INFERENCE_ENGINE", "QUANT_TYPE", "USE_COMPILE")
#: 首次用例开始前记录的基线值（= 会话起点状态），每个用例前后恢复。
_CONFIG_BASELINE: dict = {}


@pytest.fixture(autouse=True)
def _reset_process_wide_chat_state():
    """每个用例前后重置**进程级**聊天状态、request-id ContextVar 与被改写的全局 config。

    为什么必须在 conftest 兜底（而不是靠各用例自觉）：

    1. **request-id 是 ContextVar** —— 有用例直接 `.set()` 一个固定 id
       （如 `test_api_logging.py` 设 `"abc123def456"`）后**不会自动复原**
       （monkeypatch 不管 ContextVar）。后续用例经
       `api_server._commit_chat_context_turn` 会拿到同一个 operation_id。
    2. **`api_server._chat_context` 是模块级单例** —— 其
       `_committed_operations`（operation_id → turn 的幂等表）跨用例累积：
       同一个 operation_id 第二次带**不同**内容就会抛
       `ConversationContextConflict` → API 409
       `operation_id has conflicting conversation turn`。
    3. **生产代码会在运行时改写全局 `config`** —— 例如
       `api_server._auto_load_default_model()` 直接写
       `cfg.INFERENCE_ENGINE = resolution.engine` /
       `cfg.QUANT_TYPE = resolution.quant_type`（票 2 引入）。它是进程级副作用、
       不随用例复原，于是任何触发过自动加载的用例都会改变后续用例看到的
       "当前引擎"，使 `ModelManager.select_engine()` 类断言随机失败
       （实测：`test_model_module.py::TestSelectEngine::test_manual_llama_cpp_override`
       全量红 / 单跑绿）。

    实测症状：全量下上述两类假红合计 25 例，单跑全绿。
    """
    # 首次调用时记录全局 config 基线（= 会话起点状态），之后每用例恢复。
    if not _CONFIG_BASELINE:
        try:
            import config as _cfg
            for _name in _CONFIG_BASELINE_NAMES:
                if hasattr(_cfg, _name):
                    _CONFIG_BASELINE[_name] = getattr(_cfg, _name)
        except Exception:  # noqa: BLE001
            pass

    def _reset() -> None:
        try:
            import api_server
        except Exception:  # noqa: BLE001 - 环境缺依赖时不该让用例中断
            pass
        else:
            try:
                service = getattr(api_server, "_chat_context", None)
                if service is not None:
                    with service._lock:
                        # 只清"幂等/跨请求累积"的部分，保留 service 的身份与配置。
                        service._committed_operations.clear()
                        service._revisions.clear()
                        service.histories.clear()
                        service._history_generations.clear()
                        service.active_session_id = None
            except Exception:  # noqa: BLE001
                pass
            try:
                ctx = getattr(api_server, "_request_id_ctx", None)
                if ctx is not None:
                    ctx.set("")
            except Exception:  # noqa: BLE001
                pass
        try:
            import config as _cfg
            for _name, _value in _CONFIG_BASELINE.items():
                setattr(_cfg, _name, _value)
        except Exception:  # noqa: BLE001
            pass

    _reset()
    yield
    _reset()
