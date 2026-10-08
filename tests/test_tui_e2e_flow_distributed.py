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
import time
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


def _workflow_items(api) -> list:
    """`/api/workflows` 里的 workflow 记录列表（取不到 ⇒ 空）。"""
    try:
        payload = api.get("/workflows")
    except Exception:  # noqa: BLE001 - 后端不可达 ⇒ 空，由断言暴露
        return []
    items = payload.get("workflows") if isinstance(payload, dict) else None
    return [item for item in (items or []) if isinstance(item, dict)]


def _workflow_ids(api) -> set:
    """当前全部 workflow id —— 用于「操作前后差集」定位**本次新增**的那一条。

    ⚠️ 不能取 `items[0]`：`/workflows` 的首条恒为旧记录（实测三次运行都取到同一条
    `wf_21e78e…`，而本次请求其实已产生新 workflow ⇒ 判据永远失败）。
    """
    return {
        str(item.get("workflow_id"))
        for item in _workflow_items(api)
        if item.get("workflow_id")
    }


def _distributed_enabled(api) -> bool:
    """`/api/cluster/config` 里分布式推理开关的当前值（取不到 ⇒ False）。"""
    try:
        payload = api.get("/cluster/config")
    except Exception:  # noqa: BLE001 - 后端不可达 ⇒ 视为未开启，由断言暴露
        return False
    if not isinstance(payload, dict):
        return False
    switch = payload.get("distributed_inference")
    return bool(isinstance(switch, dict) and switch.get("enabled") is True)


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
    # ★ 2026-10-08：**不得用 `model_loaded` 当"就绪"判据**。
    #   Route-A 层段模式下 master 按层段参与，`model_loaded` 会被**合法地**置 False
    #   （语义见本文件 Route-A 用例的注记与 `routes_models.get_current_model`；
    #   `/status.model_loaded` 与 `/models/current.loaded` 同源，都读 `model_host.model_loaded`）。
    #   实测后果：先跑本文件的 Route-A 用例、再跑本用例时，跑前 `loaded=True`、跑完 `False`
    #   ⇒ 本用例被**误 skip**（理由写"未加载完整模型"，指向了错误的修复方向）。
    #   改判**可路由性**：capacity 能给出可用计划（v3 语义）即视为就绪，与"master 是整模
    #   还是层段"无关；只有 capacity 也求不出来时才退回 `model_loaded` 兜底。
    capacity_ready, capacity_reason = _pipeline_capacity_ready(api)
    if not capacity_ready and (
        not isinstance(status, dict) or status.get("model_loaded") is not True
    ):
        return ("F3 前置未满足：capacity 求不出可用计划"
                f"（{capacity_reason or 'unknown'}），且本机 master 未加载完整模型")
    # ★ 2026-10-07：层段流水线（Route-A / 任务图）要求 **PyTorch** 引擎 —— llama.cpp 只能
    #   整模本地推理（`llama.cpp engine does not support layer-split pipeline`）。实测引擎
    #   不对时请求会静默落到 `local_llama_cpp`：既不产生 workflow，也没有 `distributed_used`
    #   ⇒ 档会给出**误导性失败**（"未走分布式"），而不是说明"环境没就绪"。这里补前置检查。
    try:
        current = api.get("/models/current")
    except Exception:  # noqa: BLE001 - 取不到就当作未知，交给下游断言
        current = {}
    engine = str((current or {}).get("engine") or "").strip().lower()
    if engine and engine != "pytorch":
        return (f"本机 master 引擎为 {engine!r}（层段流水线需要 pytorch；llama.cpp 只能整模"
                f"本地推理，F3 会静默落到 local_llama_cpp）")
    online = _online_node_count(api)
    if online < MIN_ONLINE_NODES:
        return (f"双机集群未就绪：online 节点 {online} < {MIN_ONLINE_NODES}"
                f"（F3 前置：Surface 需已 /api/cluster/connect 入集群）")
    # ★ 2026-10-07：#29 的**发起侧**缺口已闭合 —— TUI 现在通过 `/mode task_graph` 显式传递
    #   `execution_mode`（`tui_shared.build_interactive_request` → `tui_api.iter_chat_payloads`
    #   → `tui_textual` 的 `/mode` 命令）。此前 TUI 只传 `routing_preference`，请求被静默导向
    #   层流水线 ⇒ 到不了任务图、拿不到 `distributed_used`，本档只能靠
    #   `QLH_TUI_E2E_EXPECT_DISTRIBUTED=1` 手工放行。现在默认就跑，断言必须真拿到判据。
    return ""


def test_dual_host_tui_routed_pipeline_streams_text():
    """★ Route-A（stage_offer_v3 层段链）的**产物级**验收：TUI 真操作 + 普通分布式流水线。

    与主判据（任务图 + `distributed_used`）互补：这里**不切任务图**，只走 `/route required`
    ⇒ 普通流水线 ⇒ 远端层段参与 ⇒ **回答非空**。普通流水线**不产生 workflow**，所以主判据
    的取数方式在这里天然不适用（这正是"未出现新 workflow"的由来）。

    这条正是「原生思考抑制判据」缺陷的产物级回归：修复前该判据把**模板给模型的指令**当成
    「模型已进入思考」，而 `qwen3-5-2b` 模板在 `enable_thinking` 非 true 时注入的是**已闭合**
    的思考块 ⇒ 生成段永远等不到 `</think>` ⇒ 正文被整段吞掉（流式 0 个 token 事件、非流式
    「流水线返回空响应」）。
    """
    from tui_api import ApiClient
    from tui_textual import KoakumaApp, MainScreen

    control = ApiClient(host=HOST, port=PORT, timeout=120.0)
    if not _enabled():
        pytest.skip("设置 QLH_RUN_DUAL_HOST_TUI=1 才运行双机物理 TUI E2E（F3 档）")
    try:
        import textual  # noqa: F401
    except ImportError:
        pytest.skip("需要 textual（TUI 档）")
    # ★ 与主判据（任务图）**不同**：Route-A 不要求 master 处于"完整模型"状态 —— 恰恰相反，
    #   Route-A 要求 master **按层段参与**（跑起来后 `/status.model_loaded` 会变成 False，
    #   实测因此被 `_skip_reason` 误 skip）。这里的前置只判 capacity 能否给出可用计划
    #   （v3 语义，见 `_pipeline_capacity_ready`）。
    capacity_ready, capacity_reason = _pipeline_capacity_ready(control)
    if not capacity_ready:
        pytest.skip(
            f"Route-A 前提未满足：capacity 求解失败（{capacity_reason or 'unknown'}）；"
            f"需先让三机层段链就绪（master 参与层段 + worker 段覆盖完整）再跑本用例")
    # ★ 2026-10-08：**不做"实时探测请求"**。曾用一个 `POST /chat`（max_new_tokens=1）探真实
    #   可路由性，但它本身就是一次真实的分布式请求：会让链路进入重配置/能力刷新窗口，紧随其后的
    #   验收请求被拒（实测：探测通过后 0 秒 / 5 秒，验收请求都报
    #   `advertised layer_ranges cannot cover …`，而同一时刻单发请求成功）。前置只保留
    #   「`/cluster/status` 就绪 + capacity 可解」这两项**只读**检查；真跑不通时用例的失败信息
    #   已经足够定位。

    async def _main() -> str:
        from textual.widgets import Input, Static

        app = KoakumaApp(ApiClient(host=HOST, port=PORT, timeout=120.0), interval=30)
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await wait_for(pilot, lambda: isinstance(app.screen, MainScreen))
            assert await wait_for(
                pilot, lambda: bool(app.screen.query("#chat-pane"))), "聊天屏 #chat-pane 未挂载"
            screen = app.screen

            # 1) 分布式开关：与主判据同样的「确保开启」语义（`t` 是反转开关，不能盲目按）
            landed = await pilot.click("#nav ListItem#nav-cluster")
            assert landed, "点击侧栏 cluster 项未命中（`Pilot.click` 自带落点断言）"
            await pilot.pause()
            screen.load_cluster_aux()
            assert await wait_for(
                pilot, lambda: bool((screen.cluster_aux or {}).get("distributed"))), (
                f"cluster_aux 未加载到分布式配置: {screen.cluster_aux!r}")
            if not _distributed_enabled(control):
                screen.action_cluster_toggle()
                await pilot.pause()
                await pilot.press("y")
            assert await wait_for(
                pilot, lambda: _distributed_enabled(control)), (
                "分布式开关未开启（/api/cluster/config 复核失败）")

            # 2) 路由偏好：/route required（**不**切任务图 ⇒ 走普通流水线）
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

            # 3) 发一条真消息，等非空回答
            box.value = PROMPT
            box.focus()
            await pilot.press("enter")

            def _answer() -> str:
                return _assistant_body(str(pane.query_one("#chat-log", Static).render()))

            got = await wait_for(pilot, lambda: len(_answer()) >= 2, timeout=REPLY_TIMEOUT)
            if not got:
                raise AssertionError(
                    f"真模型在 {REPLY_TIMEOUT:.0f}s 内未给出非空回答（Route-A 产物级判据）；"
                    f"已渲染文本（前 400 字）: "
                    f"{str(pane.query_one('#chat-log', Static).render())[:400]!r}")
            return _answer()

    answer = _run(_main())
    assert len(answer) >= 2, f"回答过短: {answer!r}"
    for marker in ("流水线返回空响应", "当前没有可用的分布式路径", "distributed_required"):
        assert marker not in answer, f"回答里出现失败文案 {marker!r}: {answer[:200]!r}"


def _pipeline_capacity_ready(api) -> tuple:
    """Route-A 前提：capacity 求解能给出可用计划（**v3 语义**）。

    ⚠️ **不能**用 `/cluster/status` 的 `layer_ready` —— 那是 **legacy 层配置 ACK** 的概念；
    v3 层段 worker（`stage_offer_v3`）不走 legacy ACK，该字段恒为 `False/not_configured`
    ⇒ 拿它做前置会让本档永远 skip（实测踩到：capacity 明明已经 `admitted=True`、
    `assignments=[('master',0,16),(Surface,16,20),(Y700,20,24)]`，档却 skip 了）。

    这里读 `/cluster/pipeline-capacity` **并叠加** `/status.pipeline.available`：

    ⚠️ 只用前者不够 —— 它是**上一次求解的缓存**，会给出"上次成功"的陈旧结论（实测：档因此
    不 skip、直接跑，然后在请求入口被拒 `pipeline workers not ready`，1.7 秒失败）。
    真机上 Y700 每 ~37 秒主动断开重连一次（`tcp_comm: 客户端 android-21af7c52 已断开`，客户端
    发起），落在断开窗口里的请求必然失败 ⇒ 前置必须是**实时**的。
    """
    try:
        # ⚠️ 必须是 `/cluster/status`（`ApiClient.get` 会自动补 `/api` 前缀）—— 非 cluster 的
        #    `/status` 里没有 `pipeline` 字段，误读会恒定判 `pipeline_unavailable`（实测踩到：
        #    档一直 skip，而同一时刻 `/cluster/status` 报 `available=True reason=ready`）。
        status = api.get("/cluster/status")
    except Exception:  # noqa: BLE001
        return False, "status 不可达"
    pipeline = status.get("pipeline") if isinstance(status, dict) else None
    if not isinstance(pipeline, dict) or pipeline.get("available") is not True:
        return False, str((pipeline or {}).get("readiness_reason_code") or "pipeline_unavailable")
    try:
        payload = api.get("/cluster/pipeline-capacity")
    except Exception:  # noqa: BLE001 - 取不到 ⇒ 未就绪
        return False, "capacity 不可达"
    if not isinstance(payload, dict):
        return False, "capacity 返回异常"
    if payload.get("admitted") is True:
        return True, ""
    reason_code = str(payload.get("reason_code") or "")
    if reason_code in {
        "pipeline_layer_range_coverage_insufficient",
        "pipeline_distributed_capacity_insufficient",
        "pipeline_cluster_capacity_insufficient",
    }:
        return False, reason_code
    return True, ""


def _assistant_body(text: str) -> str:
    """剥掉角色标签，取 assistant 回答**正文**（同 F2：整体长度会把空回答误判成通过）。"""
    parts = re.split(r"assistant", text)
    return parts[-1].strip() if len(parts) > 1 else ""


def test_dual_host_tui_distributed_end_to_end():
    """★ F3 主判据（**任务图档**，默认不跑）：**TUI 真操作** ⇒ 任务图执行 ⇒ `distributed_used=true`。

    ⚠️ 2026-10-08 更正（用户裁定）：**任务图需要整模加载，优先级低于层段（Route-A）路径，
    产品正常情况下不放开它**。因此本档默认 skip —— 它失败只说明"没走任务图"，**不代表
    分布式不可用**（层段路径才是默认路径）。层段侧的主判据见
    `test_dual_host_tui_routed_pipeline_streams_text`。

    实测反例（当作证据留档）：TUI 内 `/mode task_graph` 生效（`app.execution_mode
    == "task_graph"` 断言通过），但同一请求的后端日志走的是 Route-A
    （`Route-A 分配原始: raw=[('master',0,20,20,None),('android-21af7c52',20,24,4,
    'stage_offer_v3')]` + 逐 step handoff）⇒ 不产生 workflow。这与"任务图优先级低"
    一致，**不是缺陷**。

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

    if (os.environ.get("QLH_TUI_E2E_EXPECT_TASK_GRAPH") or "").strip() != "1":
        pytest.skip(
            "任务图需整模加载、优先级低于层段路径（产品正常不放开）⇒ 本档默认不跑；"
            "要显式验证任务图路径请设 QLH_TUI_E2E_EXPECT_TASK_GRAPH=1。"
            "分布式主判据请走层段档 test_dual_host_tui_routed_pipeline_streams_text")

    control = ApiClient(host=HOST, port=PORT, timeout=120.0)
    reason = _skip_reason(control)
    if reason:
        pytest.skip(reason)
    # ★ 2026-10-07：记录操作**前**已有的 workflow，稍后用差集定位本次新增项。
    before_ids = _workflow_ids(control)

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

            # 真按 `t`（`Binding("t", "cluster_toggle")`）⇒ 弹确认框 ⇒ 按 `y` 确认。
            # ★ 2026-10-07 两处修正：
            #   ① `t` 是屏幕级 binding，键盘焦点若被别的控件吃掉就**静默无效**（实测：按了
            #      `t`、`y` 后端没收到任何 PUT，开关仍 false ⇒ 请求被 `distributed_required`
            #      路由门拒成 `outcome=refused`）⇒ 改走屏幕动作触发同一条 UI 路径。
            #   ② 它是**反转**开关 ⇒ 档不能盲目按（第二次运行会把已开启的开关关掉）⇒
            #      先读后端当前值，仅在关闭时才 toggle。两次都**复核后端状态**，失败立刻暴露。
            if not _distributed_enabled(control):
                screen.action_cluster_toggle()
                await pilot.pause()
                await pilot.press("y")
            assert await wait_for(
                pilot, lambda: _distributed_enabled(control)), (
                "分布式开关未开启（/api/cluster/config 复核失败）")

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

            # --- 2b) 执行模式：/mode task_graph ⇒ 显式任务图（否则拿不到 distributed_used）---
            box.value = "/mode task_graph"
            box.focus()
            await pilot.press("enter")
            assert await wait_for(
                pilot, lambda: getattr(app, "execution_mode", None) == "task_graph"), (
                f"/mode task_graph 未生效: execution_mode="
                f"{getattr(app, 'execution_mode', None)!r}")

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
        # ★ 2026-10-07：不取 `items[0]`（该端点首条**恒为旧记录**，实测三次运行都取到同一条
        #   `wf_21e78e…`，而本次请求其实已产生新 workflow ⇒ 判据永远失败）。改为用
        #   「操作前后 id 差集」定位本次新增项，并轮询等它落库（任务图是异步的）。
        def _new_workflow():
            for item in _workflow_items(control):
                candidate = str(item.get("workflow_id") or "")
                if candidate and candidate not in before_ids:
                    return item
            return None

        deadline = time.time() + REPLY_TIMEOUT
        latest = None
        while latest is None and time.time() < deadline:
            latest = _new_workflow()
            if latest is None:
                time.sleep(1.0)
        assert latest is not None, (
            f"{REPLY_TIMEOUT:.0f}s 内未出现新 workflow（操作前 {len(before_ids)} 条）"
            f"⇒ 本次 TUI 请求没有落到任务图")
        workflow_id = str(latest.get("workflow_id") or "")
        detail = control.get(f"/workflows/{workflow_id}") if workflow_id else {}
        metrics = detail.get("metrics") if isinstance(detail, dict) else None

        assert latest.get("distributed_used") is True or (
            isinstance(metrics, dict) and metrics.get("distributed_used") is True
        ), (f"本次 TUI 请求未走分布式: distributed_used="
            f"{latest.get('distributed_used')!r} / metrics="
            f"{(metrics or {}).get('distributed_used')!r}; workflow={workflow_id!r}; "
            f"操作前 {len(before_ids)} 条 / 现在 {len(_workflow_items(control))} 条")
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
