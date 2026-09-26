"""TUI 端到端 flow —— **F3 档：双机物理 + 分布式**（默认不跑，需显式开关 + 集群已就绪）。

与 F2（`tests/test_tui_e2e_flow_real.py`）的区别
-----------------------------------------------
| | F2（`..._real.py`） | **F3（本文件）** |
|---|---|---|
| 拓扑 | 单机 loopback | **双机物理**：本机 = master，Surface = client |
| 模型 | 真权重 | 真权重（master 侧加载完整模型） |
| 分布式 | 不涉及 | **TUI 里真操作**：`t`（分布式开关）→ 确认 → `/route required` → 发消息 |
| 判据 | 回答非空 | 回答非空 **且** `metrics.distributed_used is True` |
| 开关 | `QLH_RUN_REAL_MODEL_SMOKE=1` | `QLH_RUN_DUAL_HOST_TUI=1`（**另需双机集群已就绪**） |

纪律（与 F1/F2 一致）
---------------------
* 缺开关 / 缺集群 / 缺工件 / 缺依赖 ⇒ **一律 skip，不用 `xfail`、不静默 `pass`**；
* **不虚构证据**：所有断言都基于真实回答文本与真实 `/api/*` 返回；
* **不改产品代码路径**：全程只用公共 API（`tui_api` / `Pilot` / `App.run_test`）；
* **本档不代为拉起 Surface** —— 双机集群必须由人工先建好（见下方前置）。

前置（测试只检查、不代做）
------------------------
1. 本机（master）已起 `api_server`（端口同 `QLH_TUI_E2E_PORT`，默认 8000）并**已加载完整模型**；
2. Surface 已 `POST /api/cluster/connect` 入集群，`/api/cluster/nodes` 里 **≥2 个节点 online**；
3. 远端 worker 的模型摘要与 master 一致（否则远端 Stage 会被拒 —— 那是 #28 的领域，
   **不属本档断言范围**，本档只判「TUI 操作 → 真分布式执行」这条链是否通）。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src"), str(ROOT / "tests")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from tui_e2e_flow import wait_for  # noqa: E402

# F3 uses a real backend/model and a physical peer.  Keep it out of the
# always-on unit channel; the smoke channel is already serial and opt-in.
pytestmark = [pytest.mark.real_model, pytest.mark.slow]

MODEL_ID = (os.environ.get("QLH_TUI_E2E_MODEL") or "qwen2.5-0.5b").strip()
ENGINE = (os.environ.get("QLH_TUI_E2E_ENGINE") or "pytorch").strip()
HOST = "127.0.0.1"
PORT = int(os.environ.get("QLH_TUI_E2E_PORT") or "8000")
REPLY_TIMEOUT = float(os.environ.get("QLH_TUI_E2E_TIMEOUT") or "180")
PROMPT = (os.environ.get("QLH_TUI_E2E_PROMPT")
          or "用一句话说明分布式推理的意义。").strip()
#: F3 要求**至少两个节点 online**（本机 master + 至少一个远端 client）。
MIN_ONLINE_NODES = int(os.environ.get("QLH_TUI_E2E_MIN_NODES") or "2")


def _enabled() -> bool:
    return (os.environ.get("QLH_RUN_DUAL_HOST_TUI") or "").strip() == "1"


def _run(coro):
    return asyncio.run(coro)


def _online_node_count(api) -> int:
    """`/api/cluster/nodes` 里 online 节点数（取不到 ⇒ 0）。

    ⚠️ 2026-09-26 实测修正：节点状态字段是 **`state`**（值 `"online"`），**不是 `status`**
    —— `status` 恒为空，按它统计会永远得 0、让 F3 无条件 skip。顶层另有 `online_count`
    可直接用，优先取它。
    """
    try:
        payload = api.get("/cluster/nodes")
    except Exception:  # noqa: BLE001 - 集群不可达 ⇒ 计 0，由 skip 理由说明
        return 0
    if not isinstance(payload, dict):
        return 0
    count = payload.get("online_count")
    if isinstance(count, int):
        return count
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return 0
    return sum(
        1 for node in nodes
        if isinstance(node, dict) and str(node.get("state", "")).lower() == "online"
    )


def _skip_reason(api=None) -> str:
    if not _enabled():
        return "设置 QLH_RUN_DUAL_HOST_TUI=1 才运行双机物理 TUI E2E（F3 档）"
    try:
        import textual  # noqa: F401
    except ImportError:
        return "需要 textual（TUI 档）"
    if api is None:
        return ""
    try:
        status = api.get("/status")
    except Exception as exc:  # noqa: BLE001
        return f"本机 master 后端不可达（F3 需先起 api_server）: {exc}"
    if not isinstance(status, dict) or status.get("model_loaded") is not True:
        return "本机 master 未加载完整模型（F3 前置：先加载模型）"
    online = _online_node_count(api)
    if online < MIN_ONLINE_NODES:
        return (f"双机集群未就绪：online 节点 {online} < {MIN_ONLINE_NODES}"
                f"（F3 前置：Surface 需已 /api/cluster/connect 入集群）")
    # ⚠️ **#29（已登记）**：TUI 只传 `routing_preference`、**不传 `execution_mode`**
    #    （`src/tui_api.py:439` 一带），而后者默认 `auto` ⇒ 请求会被**静默导向层流水线**，
    #    既到不了任务图、也不会产生 `distributed_used` ⇒ F3 的判据无从满足。
    #    在 #29 修好（或 TUI 侧提供 `execution_mode` 传递）之前本档只能 skip；
    #    确认已修复后用 `QLH_TUI_E2E_EXPECT_DISTRIBUTED=1` 显式开启（届时必须真拿到判据）。
    if (os.environ.get("QLH_TUI_E2E_EXPECT_DISTRIBUTED") or "").strip() != "1":
        return ("已知问题 #29：TUI 未传递 execution_mode（默认 auto ⇒ 静默走层流水线），"
                "无法请求任务图分布式、拿不到 distributed_used。"
                "确认修复后设 QLH_TUI_E2E_EXPECT_DISTRIBUTED=1 再跑本档")
    return ""


def _assistant_body(text: str) -> str:
    """剥掉角色标签，取 assistant 回答**正文**（同 F2：整体长度会把空回答误判成通过）。"""
    parts = re.split(r"assistant", text)
    return parts[-1].strip() if len(parts) > 1 else ""


def test_dual_host_tui_distributed_end_to_end():
    """★ F3 主判据：**TUI 真操作** ⇒ 真分布式执行 ⇒ `distributed_used=true`。

    操作序列（全部走 TUI，不绕过 UI 直调 `/api/chat`）：
    1. 验证集群已就绪（≥2 节点 online）—— 不就绪直接 skip（不虚构）；
    2. 起 TUI，等 `#chat-pane` 挂载；
    3. **按 `t`** 触发分布式开关（`MainScreen.action_cluster_toggle`）+ **确认**；
    4. 聊天层 **`/route required`** ⇒ `routing_preference=distributed_required`；
    5. 在输入框发送一条消息，等回答；
    6. 断言：回答非空 **且** 最近一个 workflow 的 `distributed_used is True`。
    """
    from tui_api import ApiClient
    from tui_textual import KoakumaApp, MainScreen

    control = ApiClient(host=HOST, port=PORT, timeout=120.0)
    reason = _skip_reason(control)
    if reason:
        pytest.skip(reason)

    async def _main():
        from textual.widgets import Input, Static

        app = KoakumaApp(ApiClient(host=HOST, port=PORT, timeout=120.0), interval=30)
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await wait_for(pilot, lambda: isinstance(app.screen, MainScreen))
            # `ContentSwitcher` 只挂载当前屏 ⇒ 必须等 `#chat-pane`，不能直接 query_one
            # （F2 档同款竞态，见 `TUI端到端flow测试与答辩演示复用计划 §3.4`）。
            assert await wait_for(
                pilot, lambda: bool(app.screen.query("#chat-pane"))), "聊天屏 #chat-pane 未挂载"

            # --- 1) TUI 里真操作：点侧栏切到 cluster 页 → 按 `t` → 确认 ---
            screen = app.screen
            # ⚠️ `action_cluster_toggle` 有前置：`PAGES[self.page_index][0] != "cluster"`
            #    时**直接 return**（静默什么都不做）⇒ 必须先真点侧栏切页。
            landed = await pilot.click("#nav ListItem#nav-cluster")
            assert landed, "点击侧栏 cluster 项未命中（`Pilot.click` 自带落点断言）"
            await pilot.pause()
            # ⚠️ 另一个前置：`cluster_aux` 未加载时 `current is None` ⇒ 同样直接 return。
            #    真后端在 ⇒ `load_cluster_aux()` 会去拉 `/cluster/config/*`。
            screen.load_cluster_aux()
            assert await wait_for(
                pilot, lambda: bool((screen.cluster_aux or {}).get("distributed"))), (
                f"cluster_aux 未加载到分布式配置，无法切换: {screen.cluster_aux!r}")

            # 真按 `t`（`Binding("t", "cluster_toggle")`）⇒ 弹确认框 ⇒ 按 `y` 确认
            await pilot.press("t")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

            # --- 2) 路由偏好：/route required ⇒ distributed_required ---
            pane = screen.query_one("#chat-pane")
            assert await wait_for(pilot, lambda: bool(pane.query(Input))), "聊天输入框未挂载"
            box = pane.query_one(Input)
            box.value = "/route required"
            box.focus()
            await pilot.press("enter")
            assert await wait_for(
                pilot, lambda: getattr(app, "routing_preference", None)
                == "distributed_required"), (
                f"/route required 未生效: routing_preference="
                f"{getattr(app, 'routing_preference', None)!r}")

            # --- 3) 发一条真消息，等回答 ---
            box.value = PROMPT
            box.focus()
            await pilot.press("enter")

            def _answer() -> str:
                return _assistant_body(str(pane.query_one("#chat-log", Static).render()))

            got = await wait_for(pilot, lambda: len(_answer()) >= 2, timeout=REPLY_TIMEOUT)
            if not got:
                raise AssertionError(
                    f"真模型在 {REPLY_TIMEOUT:.0f}s 内未给出非空回答；已渲染文本（前 400 字）: "
                    f"{str(pane.query_one('#chat-log', Static).render())[:400]!r}")

        # --- 4) 后端视角的硬判据：这次请求真的走了分布式 ---
        workflows = control.get("/workflows")
        items = workflows.get("workflows") if isinstance(workflows, dict) else None
        assert items, f"/workflows 无记录，无法判定分布式: {workflows!r}"
        latest = items[0]
        workflow_id = str(latest.get("workflow_id") or "")
        detail = control.get(f"/workflows/{workflow_id}") if workflow_id else {}
        metrics = detail.get("metrics") if isinstance(detail, dict) else None

        assert latest.get("distributed_used") is True or (
            isinstance(metrics, dict) and metrics.get("distributed_used") is True
        ), (f"本次 TUI 请求未走分布式: distributed_used="
            f"{latest.get('distributed_used')!r} / metrics="
            f"{(metrics or {}).get('distributed_used')!r}; workflow={workflow_id!r}")
        assert not (latest.get("fallback") or (metrics or {}).get("fallback")), (
            f"不应发生回退: fallback={latest.get('fallback')!r} / "
            f"{(metrics or {}).get('fallback')!r}")

        # 远端 Stage 必须真的由 `remote_*` provider 承担（否则只是"本地假装分布式"）
        stages = detail.get("stages") if isinstance(detail, dict) else None
        if isinstance(stages, list) and stages:
            remote = [
                stage for stage in stages
                if str(stage.get("provider", "")).startswith("remote_")
            ]
            assert remote, (
                "没有任何 Stage 由远端 provider 承担（provider 一览: "
                f"{[stage.get('provider') for stage in stages]!r}）")

    _run(_main())
