"""T7 TUI 实测走查脚本（自动化驱动 tui_admin.py --plain 模式）

验收对应 docs/TUI适配实施计划.md T7：
  - 主节点角色：7 屏全走查（总览/节点/分布式/队列/画像/日志/设置），
    动作面全执行（连接、注册、注销、删除、转让、备用主节点、容量、
    发现、转让日志、邮件测试、重置身份、分层切换/覆盖/重置、队列
    策略/暂停/恢复/清空/取消、GPU 切换、自动配置、远程日志、连通测试）
  - 从节点角色：Dashboard + Nodes（/cluster/master-health 分支）+ 远程日志
  - 验收：tui_admin.py 零改动、7 屏 × 2 角色全部通过、无"内部错误"屏

用法：
  python scripts/tui_walkthrough.py --host 127.0.0.1 --port 8100 --mode master
  python scripts/tui_walkthrough.py --host 127.0.0.1 --port 8101 --mode client
"""
import argparse
import os
import subprocess
import sys
import tempfile

# ------------------------------------------------------------
# 输入序列：每行一个输入（空字符串 = 回车/默认值）
# ------------------------------------------------------------

MASTER_INPUTS = [
    "1",           # 屏 1 系统状态总览
    "",            #   返回
    "2",           # 屏 2 节点管理
    "v",           #   自动发现主节点
    "l",           #   转让日志
    "c",           #   连接主节点
    "",            #     主节点 Tailscale IP（默认 100.64.0.1）
    "",            #     主节点端口（默认 8888）
    "y",           #     本机为主节点，确认切换为从节点加入（act_connect 的 confirm）
    "a",           #   注册节点
    "test-node2",  #     node_id
    "",            #     hostname（空）
    "",            #     address（空）
    "",            #     network（默认 unknown）
    "",            #     node_type（默认 pc）
    "d",           #   注销节点
    "test-node2",
    "y",           #     确认
    "x",           #   删除节点记录
    "delete-ok",   #     （桩契约：delete-ok 节点删除成功，其余 404 模拟不存在）
    "y",
    "t",           #   转让主节点
    "test-client",
    "y",
    "s",           #   设置备用主节点
    "test-client",
    "y",
    "u",           #   移除备用主节点
    "y",
    "m",           #   最大节点数
    "16",
    "e",           #   邮件测试
    "y",
    "z",           #   重置身份
    "reset",
    "",            #   返回
    "3",           # 屏 3 分布式与分层
    "t",           #   切换分布式推理开关
    "y",
    "o",           #   手动覆盖分层
    "test-master", #     段 1 节点 ID
    "0",           #     起始层
    "12",          #     结束层
    "",            #     结束输入
    "y",           #     确认覆盖
    "R",           #   重置分层为自动
    "y",
    "",            #   返回
    "4",           # 屏 4 请求队列
    "s",           #   切换调度策略
    "fifo",
    "p",           #   暂停队列
    "u",           #   恢复队列
    "C",           #   清空队列
    "y",
    "k",           #   取消任务
    "task-1",
    "y",
    "",            #   返回
    "5",           # 屏 5 设备画像
    "g",           #   切换 GPU
    "0",
    "y",
    "A",           #   自动配置
    "y",
    "",            #   返回
    "6",           # 屏 6 日志查看
    "2",           #   后端最近日志（远程）
    "4",           #   日志统计
    "3",           #   日志文件列表
    "",            #   返回
    "7",           # 屏 7 设置
    "T",           #   测试后端连通
    "",            #   返回
    "q",           # 退出
]

CLIENT_INPUTS = [
    "1",           # 屏 1 系统状态总览（从节点视角）
    "",
    "2",           # 屏 2 节点管理（非主节点 → master-health 分支）
    "",
    "6",           # 屏 6 日志查看（远程日志）
    "2",
    "",
    "q",
]

# 屏标题（按出现顺序）
SCREEN_TITLES = [
    "==== 系统状态总览 ====",
    "==== 节点管理 ====",
    "==== 分布式与分层 ====",
    "==== 请求队列(MLFQ) ====",
    "==== 设备画像 ====",
    "==== 日志查看 ====",
    "==== 设置 ====",
]


def run_tui(host: str, port: int, mode: str, log_token: str = "") -> str:
    """启动 tui_admin.py --plain 并喂入输入序列，返回完整 stdout。"""
    inputs = MASTER_INPUTS if mode == "master" else CLIENT_INPUTS
    feed = "\n".join(inputs) + "\n"
    cmd = [
        sys.executable, "src/tui_admin.py",
        "--plain", "--host", host, "--port", str(port),
        "--no-color", "--interval", "30",
    ]
    if log_token:
        cmd += ["--log-token", log_token]
    proc = subprocess.run(
        cmd,
        input=feed.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=".",
        timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # 强制 TUI 输出 UTF-8
    )
    return proc.stdout.decode("utf-8", errors="replace")


def check(output: str, mode: str) -> list:
    """返回失败项列表（空 = 通过）。"""
    import re
    fails = []
    # 1. 无"内部错误"屏 / 操作失败（排除日志内容行：桩日志含 ERROR 级别记录，
    #    渲染为 "[错误] 2026-08-02 ... ERROR ..."，属合法日志内容而非错误屏）
    err_lines = [
        ln for ln in output.splitlines()
        if (ln.startswith("[错误]") or "操作失败" in ln)
        and not re.match(r"^\[错误\] \d{4}-\d{2}-\d{2} ", ln)
    ]
    if err_lines:
        fails.append("出现错误行: %s" % err_lines[:5])
    # 2. 屏标题齐全（client 模式只需从节点视角的三个屏：Dashboard/Nodes/Logs，
    #    其余屏不在从节点走查范围内——T7 定义见 docs/TUI适配实施计划.md）
    titles = SCREEN_TITLES if mode == "master" else SCREEN_TITLES[:2] + [SCREEN_TITLES[5]]
    for title in titles:
        if title not in output:
            fails.append("缺少屏幕: %s" % title)
    # 3. 动作结果行（»）覆盖检查
    if mode == "master":
        expected = [
            "发现主节点",        # v 自动发现
            "转让日志",          # l 转让日志
            "连接结果",          # c 连接
            "注册结果",          # a 注册
            "已注销",            # d 注销
            "删除结果",          # x 删除
            "转让结果",          # t 转让
            "指定结果",          # s 备用设置
            "移除结果",          # u 移除备用
            "容量更新",          # m 最大节点数
            "邮件测试",          # e 邮件测试
            "重置结果",          # z 重置身份
            "分布式推理已",      # t 切换
            "分层覆盖",          # o 覆盖分层
            "分层已重置",        # R 重置分层
            "策略已切换",        # s 队列策略
            "队列已暂停",        # p 暂停
            "队列已恢复",        # u 恢复
            "已清空",            # C 清空
            "取消",              # k 取消任务
            "已切换到 GPU",      # g 选 GPU
            "自动配置完成",      # A 自动配置
            "连接正常",          # T 连通测试
        ]
        for key in expected:
            if key not in output:
                fails.append("缺少动作结果: %s" % key)
    else:
        # 从节点：Dashboard + Nodes（master-health 分支）+ 远程日志
        for key in ["系统状态总览", "节点管理", "日志查看",
                    "主节点健康",          # master-health 渲染（桩: 在线 100.64.0.1:8888）
                    "节点注册成功"]:       # 远程日志内容（legacy_control SAMPLE_LOGS）
            if key not in output:
                fails.append("缺少从节点内容: %s" % key)
    return fails


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="T7 TUI 实测走查")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--mode", choices=["master", "client"], default="master")
    parser.add_argument("--log-token", default="")
    args = parser.parse_args()

    print("== T7 走查: mode=%s gateway=http://%s:%d ==" % (args.mode, args.host, args.port))
    output = run_tui(args.host, args.port, args.mode, args.log_token)
    with open(os.path.join(tempfile.gettempdir(), "t7_walkthrough_%s.log" % args.mode),
              "w", encoding="utf-8") as f:
        f.write(output)

    # 打印关键输出（标题行 + » 结果行 + 错误行）
    for ln in output.splitlines():
        if (ln.startswith("====") or ln.startswith("»")
                or ln.startswith("[错误]") or "操作失败" in ln
                or ln.startswith("健康:") or ln.startswith("后端:")
                or ln.startswith("主节点") or "模型已加载" in ln
                or ln.startswith("  运行模式") or ln.startswith("  节点角色")):
            print(ln)

    fails = check(output, args.mode)
    if fails:
        print("\n[T7 %s] FAIL:" % args.mode)
        for f in fails:
            print("  - %s" % f)
        return 1
    print("\n[T7 %s] PASS: 屏幕齐全、无错误屏、动作结果全部出现" % args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
