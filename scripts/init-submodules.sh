#!/usr/bin/env bash
# ============================================================
#  QLH submodule 初始化（含嵌套），显式把代理传给**每一层** git 子进程
#
#  背景：主仓 `.git/config` 里的 `http.proxy` **不会**自动传给嵌套 submodule
#  自己的 fetch。因此全新 clone 后（或 CI 上）执行 `git submodule update --init
#  --recursive` 时，内层
#      android/app/src/main/cpp/llama.cpp
#  会因无代理直连失败：
#      fatal: Unable to find current revision in submodule path
#  本机之所以能过，是因为对象已在本地（见 local_docs/CORE-LLAMA-ANDROID-01-p0-*）。
#
#  做法：同时导出 `http(s)_proxy`（libcurl 读）与 `GIT_CONFIG_*`（git 2.31+，
#  会被 git 子进程继承 → 嵌套 submodule 的 fetch 也走代理）。
#
#  用法：
#    ./scripts/init-submodules.sh                      # 代理取 QLH_HTTP_PROXY / http_proxy / 默认 7897
#    QLH_HTTP_PROXY=http://127.0.0.1:7897 ./scripts/init-submodules.sh
#    ./scripts/init-submodules.sh --no-proxy           # 直连（本地已有对象时也够用）
#    ./scripts/init-submodules.sh --check              # 只打印代理与当前 submodule 状态
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_PROXY="http://127.0.0.1:7897"
USE_PROXY="${QLH_HTTP_PROXY:-${http_proxy:-${HTTPS_PROXY:-$DEFAULT_PROXY}}}"
CHECK_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --no-proxy) USE_PROXY="" ;;
        --check)    CHECK_ONLY=1 ;;
        -h|--help)  sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          echo "[init-submodules] 未知参数：$arg" >&2; exit 2 ;;
    esac
done

if [[ -n "$USE_PROXY" ]]; then
    export http_proxy="$USE_PROXY" https_proxy="$USE_PROXY"
    export HTTP_PROXY="$USE_PROXY" HTTPS_PROXY="$USE_PROXY"
    # GIT_CONFIG_* 是环境变量，会被 git 子进程继承 → 嵌套 submodule 的 fetch 也走代理
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0="$USE_PROXY"
    echo "[init-submodules] 代理：$USE_PROXY（已导出 http(s)_proxy 与 GIT_CONFIG_*）"
else
    echo "[init-submodules] 直连模式（--no-proxy）"
fi

echo "[init-submodules] 当前 submodule 状态："
git submodule status --recursive || true

if [[ "$CHECK_ONLY" == 1 ]]; then
    exit 0
fi

echo "[init-submodules] sync + update（可能需要几分钟）…"
git submodule sync --recursive
git submodule update --init --recursive

echo "[init-submodules] 完成；复核状态："
git submodule status --recursive
