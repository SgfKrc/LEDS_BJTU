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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
