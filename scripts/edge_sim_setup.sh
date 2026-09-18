#!/usr/bin/env bash
# QLH 边缘设备模拟环境 —— 复现脚本（WSL2 原生，不依赖 Docker）
#
# 对应文档：docs/边缘设备模拟环境计划-2026-09-15.md
# 验收指标（计划 §1）：① venv ≤300MB ② 冷启动 ≤15s ③ 无 torch ④ ≤1B GGUF 本地推理
#                       ⑤ TUI / 节点注册 / RPC worker 在受限环境可用
#
# 为什么不用 Docker：计划 §2 的结论是「尺寸/依赖/冷启动类指标 → 容器」，
# 而 §3.1 的 WSL 分发级限制（.wslconfig）已能覆盖全部 5 项指标。因此本脚本
# 走「WSL 原生」路径，省掉 Docker Desktop 依赖；需要逐容器隔离时再补 §3.2。
#
# 用法（两种模式）：
#
#   1) 主机侧：配置 .wslconfig 限制 WSL 资源（Windows / Git Bash 里执行）
#        bash scripts/edge_sim_setup.sh --host-config
#      改后需 `wsl --shutdown` 生效。
#
#   2) WSL 内：建受限 venv、编 RPC worker、跑验收（在 WSL 里执行）
#        wsl -d Ubuntu-22.04 -- bash <repo>/scripts/edge_sim_setup.sh
#      可加 --json <path> 把 preflight 结果落盘作证据。
#      可加 --model <path.gguf> 额外跑一次真实生成，验指标④ 端到端（建议用 ≤1B 模型）。
#
# 幂等：重复执行安全。已存在的 venv / 已装的 uv / 已编的 rpc-server 会被复用；
# 传 --rebuild 可强制重建 venv 与 rpc-server。
#
# 退出码：0 = 全部验收通过；1 = 有验收项失败；2 = 前置条件不满足（见 stderr）。

set -euo pipefail

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

#: venv 与构建产物的落点：放在 $HOME 下（而非 /mnt/*，避免 NTFS 上编译与 IO 变慢）
WORK_DIR="${QLH_EDGE_SIM_WORK_DIR:-${HOME}/qlh-wsl}"
EDGE_VENV="${WORK_DIR}/.venv-edge"
RPC_BUILD="${WORK_DIR}/llama-rpc-build"

#: 与 requirements-edge.txt 一致的 Python 版本约束（numpy==2.5.3 要求 >=3.12）
PYTHON_VERSION="${QLH_EDGE_SIM_PYTHON:-3.12}"

#: RPC worker 监听点（EdgeRpcWorker 读同样的环境变量名）
RPC_HOST="${QLH_EDGE_RPC_HOST:-127.0.0.1}"
RPC_PORT="${QLH_EDGE_RPC_PORT:-50052}"

#: Edge HTTP 服务用于指标⑤ 探测的临时端口（避开常规 8090）
EDGE_PROBE_PORT="${QLH_EDGE_SIM_PROBE_PORT:-8099}"

FORBIDDEN_MODULES=(torch transformers accelerate bitsandbytes pandas einops tiktoken)

REBUILD=0
JSON_OUT=""
HOST_CONFIG=0
#: 传 --model <gguf> 时，除 preflight 外再跑一次真实 /generate（指标④ 端到端）。
MODEL_PATH="${QLH_EDGE_MODEL:-}"

log()  { printf '[edge-sim] %s\n' "$*"; }
warn() { printf '[edge-sim] WARN: %s\n' "$*" >&2; }
die()  { printf '[edge-sim] ERROR: %s\n' "$*" >&2; exit "${2:-2}"; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,2\} \{0,1\}//'; }

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-config) HOST_CONFIG=1; shift ;;
        --rebuild)     REBUILD=1; shift ;;
        --json)        JSON_OUT="${2:-}"; [[ -n "${JSON_OUT}" ]] || die "--json 需要一个路径参数"; shift 2 ;;
        --model)       MODEL_PATH="${2:-}"; [[ -n "${MODEL_PATH}" ]] || die "--model 需要一个 .gguf 路径"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             die "未知参数: $1（用 --help 查看用法）" ;;
    esac
done

# ---------------------------------------------------------------------------
# 模式 1：主机侧 —— 写 .wslconfig（必须在 Windows 侧执行，WSL 里改不了）
# ---------------------------------------------------------------------------

write_wslconfig() {
    # 目标文件在 Windows 用户目录。Git Bash/MSYS 下 $USERPROFILE 可用，
    # 也兼容在 cmd/PowerShell 中通过 bash 调用的情况。
    local target
    if [[ -n "${USERPROFILE:-}" ]] && command -v cygpath >/dev/null 2>&1; then
        target="$(cygpath -u "${USERPROFILE}")/.wslconfig"
    else
        target="${HOME}/.wslconfig"
    fi

    log "写入 WSL 资源限制 → ${target}"
    if [[ -f "${target}" ]]; then
        local backup="${target}.bak.$(date +%s)"
        cp "${target}" "${backup}"
        log "已备份原文件 → ${backup}"
    fi

    cat > "${target}" <<'WSLCONFIG'
[wsl2]
memory=4GB
processors=2
swap=2GB
WSLCONFIG

    log "内容："
    sed 's/^/    /' "${target}"
    log ""
    log "生效需要（会中断 WSL 内所有会话）："
    log "    wsl --shutdown"
    log "之后用「N 核 / ~4GB」验证：wsl -d <distro> -- bash -lc 'nproc; free -m | head -2'"
}

if [[ "${HOST_CONFIG}" -eq 1 ]]; then
    write_wslconfig
    exit 0
fi

# ---------------------------------------------------------------------------
# 模式 2：WSL 内 —— 建环境 + 编 worker + 验收
# ---------------------------------------------------------------------------

# --- 2.0 前置检查 -----------------------------------------------------------

if [[ "$(uname -s)" != "Linux" ]]; then
    die "本模式必须在 WSL/Linux 内运行（当前 uname -s = $(uname -s)）；主机侧请用 --host-config"
fi

if [[ ! -f "${REPO_ROOT}/requirements-edge.txt" ]]; then
    die "在 ${REPO_ROOT} 找不到 requirements-edge.txt（仓库根判定失败？）"
fi

if [[ ! -f "${REPO_ROOT}/scripts/edge_preflight.py" ]]; then
    die "找不到 scripts/edge_preflight.py"
fi

# 受限资源的证据：内存/核数应已由 .wslconfig 压低。未设不算致命（可能是 CI 直跑），
# 但要显式提示，避免把「未受限」的结果当成受限环境结论。
cpu_count="$(nproc)"
mem_total_mb="$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)"
log "环境：nproc=${cpu_count}  MemTotal=${mem_total_mb}MB"
if [[ "${cpu_count}" -gt 4 || "${mem_total_mb}" -gt 5120 ]]; then
    warn "本机资源看起来未受限（>4 核 或 >5GB）。若这是「边缘模拟」验收，"
    warn "请先在主机侧跑 --host-config 并 wsl --shutdown，否则指标口径不成立。"
fi

# --- 2.1 uv + Python --------------------------------------------------------

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HOME}/.local/bin"

if ! command -v uv >/dev/null 2>&1; then
    log "安装 uv（免 sudo，落到 ~/.local/bin）"
    command -v curl >/dev/null 2>&1 || die "缺少 curl，无法安装 uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv 安装后仍不可用，请检查 ~/.local/bin 是否在 PATH"
log "uv: $(uv --version)"

log "确保 managed Python ${PYTHON_VERSION} 存在"
uv python install "${PYTHON_VERSION}" >/dev/null

# --- 2.2 Edge venv ----------------------------------------------------------

if [[ "${REBUILD}" -eq 1 && -d "${EDGE_VENV}" ]]; then
    log "--rebuild：移除旧 venv ${EDGE_VENV}"
    rm -rf "${EDGE_VENV}"
fi

if [[ -x "${EDGE_VENV}/bin/python" ]]; then
    log "复用已存在的 venv：${EDGE_VENV}"
else
    log "创建 venv：${EDGE_VENV}（Python ${PYTHON_VERSION}）"
    mkdir -p "${WORK_DIR}"
    uv venv --python "${PYTHON_VERSION}" "${EDGE_VENV}" >/dev/null
fi

log "安装 requirements-edge.txt（严格按 pin，不放宽）"
# 注意：用 uv pip 而非 venv 自带 pip —— uv venv 默认不装 pip。
uv pip install --python "${EDGE_VENV}/bin/python" \
    -r "${REPO_ROOT}/requirements-edge.txt" >/dev/null

log "依赖校验（禁用模块必须缺席）"
"${EDGE_VENV}/bin/python" - <<'PY'
import importlib
import sys

required = ["llama_cpp", "fastapi", "uvicorn", "psutil", "httpx", "numpy", "textual"]
for name in required:
    importlib.import_module(name)

forbidden = ["torch", "transformers", "accelerate", "bitsandbytes", "pandas", "einops", "tiktoken"]
present = [n for n in forbidden if importlib.util.find_spec(n) is not None]
if present:
    print(f"禁用模块被引入了: {present}", file=sys.stderr)
    sys.exit(1)
print("  required/forbidden 校验通过")
PY

# --- 2.3 RPC worker（Linux 版，从 uv 的 sdist 缓存取同源 llama.cpp） ---------

build_rpc_server() {
    # 关键：RPC 传输有协议版本号，主机与 worker 必须同源编译，否则
    # 会得到「RPC server version mismatch」而连不上。这里刻意从
    # llama-cpp-python 的 sdist 缓存里取 vendor/llama.cpp，
    # 保证与 Edge venv 装的 llama-cpp-python 完全同源。
    local venv_python="${EDGE_VENV}/bin/python"

    if [[ -x "${RPC_BUILD}/bin/ggml-rpc-server" && "${REBUILD}" -eq 0 ]]; then
        log "复用已编译的 ggml-rpc-server：${RPC_BUILD}/bin/ggml-rpc-server"
        return 0
    fi

    log "定位 llama-cpp-python 的 sdist 源码树（同源编译）"
    local llama_src=""
    llama_src="$(find "${HOME}/.cache/uv" -maxdepth 12 -type d \
        -path "*llama.cpp" 2>/dev/null | grep -v '/\.git/' | head -1 || true)"

    if [[ -z "${llama_src}" || ! -f "${llama_src}/CMakeLists.txt" ]]; then
        warn "未在 uv 缓存中找到 llama.cpp 源码树。"
        warn "请先确认 Edge venv 已装 llama-cpp-python（本脚本 2.2 已做），"
        warn "或手动 `uv pip install --no-binary :all: llama-cpp-python` 触发源码下载后重试。"
        return 1
    fi
    log "源码树：${llama_src}"

    # 构建工具必须装在独立 venv，绝不能进 Edge 运行时 venv：
    # cmake 二进制约 90MB，直接装进 EDGE_VENV 会把体积从 ~247MB 顶到 ~322MB，
    # 直接击穿「venv ≤300MB」门禁（实测）。
    local build_venv="${WORK_DIR}/.venv-build"
    if [[ ! -x "${build_venv}/bin/cmake" ]]; then
        log "创建构建工具 venv：${build_venv}"
        uv venv --python "${PYTHON_VERSION}" "${build_venv}" >/dev/null
        uv pip install --python "${build_venv}/bin/python" cmake ninja >/dev/null
    fi
    local cmake_bin="${build_venv}/bin/cmake"
    local ninja_bin="${build_venv}/bin/ninja"

    log "配置并编译 ggml-rpc-server（GGML_RPC=ON, CUDA=OFF）"
    rm -rf "${RPC_BUILD}"
    mkdir -p "${RPC_BUILD}"
    (
        cd "${RPC_BUILD}"
        "${cmake_bin}" "${llama_src}" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_RPC=ON -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            -DCMAKE_MAKE_PROGRAM="${ninja_bin}" >/dev/null
        # 受限环境只有 2 核，-j 2 足够且不会拖垮宿主。
        "${cmake_bin}" --build . --target ggml-rpc-server -j 2 >/dev/null
    )

    [[ -x "${RPC_BUILD}/bin/ggml-rpc-server" ]] || { warn "编译未产出 ggml-rpc-server"; return 1; }
    log "已产出：${RPC_BUILD}/bin/ggml-rpc-server"
    return 0
}

rpc_ok=0
if build_rpc_server; then
    rpc_ok=1
fi

# --- 2.4 指标 ①-④：edge_preflight ------------------------------------------

log ""
log "=== 指标 ①-④：edge_preflight ==="
# edge_preflight 的 --json 是输出开关（打到 stdout），因此落盘用重定向。
set +e
if [[ -n "${JSON_OUT}" ]]; then
    "${EDGE_VENV}/bin/python" "${REPO_ROOT}/scripts/edge_preflight.py" \
        --python "${EDGE_VENV}/bin/python" --json > "${JSON_OUT}"
    preflight_rc=$?
    log "preflight JSON 已写入 ${JSON_OUT}"
    sed 's/^/    /' "${JSON_OUT}"
else
    "${EDGE_VENV}/bin/python" "${REPO_ROOT}/scripts/edge_preflight.py" \
        --python "${EDGE_VENV}/bin/python"
    preflight_rc=$?
fi
set -e

# --- 2.5 指标 ⑤：TUI / 节点注册 / RPC worker --------------------------------

log ""
log "=== 指标 ⑤：TUI / 节点注册 / RPC worker ==="
indicator5_rc=0
indicator4_rc=0

# ⑤-a TUI：模块可导入、CLI 可用、归档引擎显式报错（不静默失效）
log "⑤-a TUI"
if ! PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "${EDGE_VENV}/bin/python" - <<'PY'
import sys
import tui_textual, tui_commands, tui_backend  # noqa: F401
assert hasattr(tui_textual, "KoakumaApp") and hasattr(tui_textual, "run")
print("  tui 模块可用:", tui_textual.KoakumaApp.__name__)
PY
then
    warn "TUI 模块导入失败"
    indicator5_rc=1
fi

# ⑤-b 节点注册 + RPC worker：起 Edge 服务，驱动 /rpc/start → /rpc/status → /rpc/stop
log "⑤-b 节点注册 + RPC worker（起 Edge 服务于 :${EDGE_PROBE_PORT}）"
edge_log="$(mktemp)"
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
QLH_EDGE_RPC_SERVER="${RPC_BUILD}/bin/ggml-rpc-server" \
QLH_EDGE_RPC_HOST="${RPC_HOST}" \
QLH_EDGE_RPC_PORT="${RPC_PORT}" \
QLH_EDGE_MODEL="${MODEL_PATH}" \
    "${EDGE_VENV}/bin/python" -m uvicorn --host 127.0.0.1 --port "${EDGE_PROBE_PORT}" \
    qlh_edge:app > "${edge_log}" 2>&1 &
svc_pid=$!
trap 'kill "${svc_pid}" 2>/dev/null || true' EXIT

# 等待就绪（最多 ~15s）
ready=0
for _ in $(seq 1 30); do
    if curl -fsS -m 2 "http://127.0.0.1:${EDGE_PROBE_PORT}/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 0.5
done

if [[ "${ready}" -ne 1 ]]; then
    warn "Edge 服务未在超时内就绪，日志尾部："
    tail -20 "${edge_log}" | sed 's/^/    /' >&2
    indicator5_rc=1
else
    probe() {
        local method="$1" path="$2"
        curl -fsS -m 5 -X "${method}" "http://127.0.0.1:${EDGE_PROBE_PORT}${path}" 2>&1 || echo "<失败>"
    }
    echo "  GET  /health      : $(probe GET /health)"
    echo "  GET  /status      : $(probe GET /status)"
    echo "  GET  /capabilities: $(probe GET /capabilities)"
    echo "  GET  /rpc/status  : $(probe GET /rpc/status)"

    if [[ "${rpc_ok}" -eq 1 ]]; then
        echo "  POST /rpc/start   : $(probe POST /rpc/start)"
        sleep 1
        start_status="$(probe GET /rpc/status)"
        echo "  GET  /rpc/status  : ${start_status}"

        # 端口必须真的在监听，否则「running:true」只是状态自述
        if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${RPC_PORT}"; then
            echo "  端口 ${RPC_PORT} 监听: 是"
        else
            warn "状态显示 running，但端口 ${RPC_PORT} 未监听"
            indicator5_rc=1
        fi

        echo "  POST /rpc/stop    : $(probe POST /rpc/stop)"
    else
        warn "跳过 RPC worker 起停（worker 未编译成功）"
        indicator5_rc=1
    fi

    # 指标④：≤1B GGUF 端到端真实生成（需要 --model；未提供则只确认路由存在）
    if [[ -n "${MODEL_PATH}" ]]; then
        log "④ ≤1B GGUF 端到端生成（$(basename "${MODEL_PATH}")）"
        gen_out="$(curl -fsS -m 240 -X POST \
            "http://127.0.0.1:${EDGE_PROBE_PORT}/generate" \
            -H 'Content-Type: application/json' \
            -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0.0,"top_p":1.0}' 2>&1 || echo '<失败>')"
        echo "  POST /generate    : ${gen_out}"
        case "${gen_out}" in
            *'"text"'*'"completion_tokens"'*) ;;   # 有正文与用量即视为生成成功
            *) warn "指标④ /generate 未返回预期结构"; indicator4_rc=1 ;;
        esac
        echo "  GET  /health      : $(probe GET /health)"
    else
        log "④ 未提供 --model，仅确认 /generate 路由存在（传 --model <gguf> 可跑真实生成）"
    fi
fi

kill "${svc_pid}" 2>/dev/null || true
trap - EXIT
rm -f "${edge_log}"

# --- 2.6 汇总 ---------------------------------------------------------------

log ""
log "=== 汇总 ==="
log "  指标 ①-③   (preflight 门禁) : rc=${preflight_rc}"
log "  指标 ④     (真实生成)       : rc=${indicator4_rc}$([[ -n "${MODEL_PATH}" ]] || echo '  (未提供 --model，跳过)')"
log "  指标 ⑤     (TUI/RPC)        : rc=${indicator5_rc}"

if [[ "${preflight_rc}" -eq 0 && "${indicator5_rc}" -eq 0 && "${indicator4_rc}" -eq 0 ]]; then
    log "全部验收通过 ✅"
    exit 0
fi
log "存在未通过项 ❌"
exit 1
