#!/usr/bin/env bash
# 在**设备侧**按需裁层，并把工件投递到模型目录（Termux 环境运行）。
#
# 为什么要设备侧裁：Android 能执行哪一段层由它手上那份 GGUF 决定 —— llama.cpp 的
# 层段元数据（`block_count` / `first_local_layer_maps_to`）是**加载期**读的，运行时
# 改不了。所以"按调度分配跑任意区间"的前提是设备自己产出对应工件。
#
# 本脚本把这一步固化：拿区间 → 用整模裁层 → 产出 GGUF + manifest 到模型目录；
# App 下一次扫描模型目录即把该区间作为 `layer_ranges` 上报，闭环完成。
#
# 用法（Termux）：
#   bash qlh_device_cut_range.sh --model <整模.gguf> --start 16          # 尾段 [16, 总层数)
#   bash qlh_device_cut_range.sh --model <整模.gguf> --start 4 --end 16  # 中段 [4, 16)
#   bash qlh_device_cut_range.sh --model <整模.gguf> --start 0 --end 8   # 头段 [0, 8)
#   # 免手填区间：从主节点读本节点当前的分配
#   bash qlh_device_cut_range.sh --model <整模.gguf> \
#        --from-master http://<master>:8000 --node-id <node_id>
#
# 前提：Termux 已装 python3 与 gguf（`python3 -m pip install gguf`）；
#      `cut_layers.py` 与本脚本同目录，或用 `--cut-layers` 指定。
set -euo pipefail

MODEL=""
START=""
END=""
MODELS_DIR="/sdcard/Download/QLH/models"
CUT_LAYERS="$(dirname "$0")/cut_layers.py"
MASTER=""
NODE_ID=""
DRY_RUN=0

usage() {
  cat >&2 <<'USAGE'
qlh_device_cut_range.sh —— 设备侧按需裁层
  --model <整模.gguf>        必填，源整模
  --start <K>                区间起点（0 = 头段，须配 --end）
  --end <E>                  区间终点，缺省 = 保留到末尾
  --models-dir <目录>        模型目录，默认 /sdcard/Download/QLH/models
  --cut-layers <路径>        cut_layers.py 路径，默认与本脚本同目录
  --from-master <URL>        从主节点 /api/cluster/layers 读分配（配 --node-id）
  --node-id <id>             本节点 id
  --dry-run                  只打印计划，不写文件
USAGE
  exit "${1:-1}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --start) START="${2:-}"; shift 2 ;;
    --end) END="${2:-}"; shift 2 ;;
    --models-dir) MODELS_DIR="${2:-}"; shift 2 ;;
    --cut-layers) CUT_LAYERS="${2:-}"; shift 2 ;;
    --from-master) MASTER="${2:-}"; shift 2 ;;
    --node-id) NODE_ID="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "未知参数: $1" >&2; usage 1 ;;
  esac
done

[ -n "$MODEL" ] || { echo "缺少 --model（源整模 GGUF）" >&2; usage 2; }
[ -f "$MODEL" ] || { echo "源整模不存在: $MODEL" >&2; exit 2; }
[ -f "$CUT_LAYERS" ] || { echo "找不到 cut_layers.py: $CUT_LAYERS" >&2; exit 2; }

# 可选：从主节点读本节点当前的分配，免手填区间。
if [ -n "$MASTER" ] && [ -n "$NODE_ID" ]; then
  RANGE="$(python3 - "$MASTER" "$NODE_ID" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/api/cluster/layers"
node = sys.argv[2]
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
except Exception as exc:  # noqa: BLE001 - 只做提示，失败即退出
    print(f"取主节点分配失败: {exc}", file=sys.stderr)
    raise SystemExit(3)
for item in data.get("assignments", []):
    if str(item.get("node_id")) == node:
        print(f"{int(item.get('start_layer', 0))} {int(item.get('end_layer', 0))}")
        break
else:
    print(f"主节点当前没有给 {node} 的分配", file=sys.stderr)
    raise SystemExit(3)
PY
)"
  START="${RANGE%% *}"
  END="${RANGE##* }"
  echo "[cut] 从主节点读到分配: [${START},${END})"
fi

[ -n "$START" ] || { echo "缺少 --start（或 --from-master + --node-id）" >&2; usage 2; }

STEM="$(basename "${MODEL%.*}")"
SUFFIX="${START}${END:+-$END}"
OUT="$MODELS_DIR/${STEM}-cut-${SUFFIX}.gguf"
MANIFEST="$MODELS_DIR/${STEM}-cut-${SUFFIX}.manifest.json"

if [ "$START" -eq 0 ]; then
  [ -n "$END" ] || { echo "头段（--start 0）必须给 --end" >&2; exit 2; }
  ARGS=(--keep-head "$END")
else
  ARGS=(--k "$START")
  [ -n "$END" ] && ARGS+=(--end "$END")
fi
[ "$DRY_RUN" -eq 1 ] && ARGS+=(--dry-run)

mkdir -p "$MODELS_DIR"
echo "[cut] 源   = $MODEL"
echo "[cut] 区间 = [${START},${END:-总层数})"
echo "[cut] 产物 = $OUT"
python3 "$CUT_LAYERS" --src "$MODEL" --dst "$OUT" --manifest "$MANIFEST" "${ARGS[@]}"

if [ "$DRY_RUN" -eq 0 ]; then
  echo
  echo "[next] 工件与 manifest 已就位。App 下次扫描模型目录时会把该区间作为"
  echo "       layer_ranges 上报给主节点（可重启 App 触发重扫）。"
fi
