#!/bin/bash
# ================================================================
# QLH 边缘推理系统 — Linux .deb 打包脚本
# ================================================================
# 用法:
#   集显版: ./build-deb.sh cpu
#   独显版: ./build-deb.sh cuda
#   仅检查: ./build-deb.sh cpu --preflight-only
#
# 前置条件:
#   1. Ubuntu 22.04+ / Debian 12+
#   2. python3, python3-venv, python3-pip 已安装
#   3. Node.js 18+ (前端构建)
#   4. dpkg-deb 可用
#
# 输出:
#   packaging/linux/qlh-edge-inference-cpu_0.1.8.1_amd64.deb
#   packaging/linux/qlh-edge-inference-cuda_0.1.8.1_amd64.deb
# ================================================================

set -euo pipefail

VARIANT="${1:-cpu}"
PREFLIGHT_ONLY="${2:-}"
VERSION="0.1.8.1"
ARCH="amd64"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PACKAGING_DIR="$PROJECT_ROOT/packaging"
SRC_DIR="$PROJECT_ROOT/src"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
BUILD_DIR="/tmp/qlh-deb-build"
MODEL_TOOL_ROOT="$PROJECT_ROOT/build/model-tools/llama-quantize"
MODEL_TOOL_PACKAGE="$MODEL_TOOL_ROOT/packages/linux-x86_64"

if [ "$VARIANT" != "cpu" ] && [ "$VARIANT" != "cuda" ]; then
    echo "错误: 变体只能是 cpu 或 cuda。"
    exit 1
fi
if [ -n "$PREFLIGHT_ONLY" ] && [ "$PREFLIGHT_ONLY" != "--preflight-only" ]; then
    echo "错误: 未知参数 $PREFLIGHT_ONLY"
    exit 1
fi

# 发布门只接受 Linux 原生工具；WSL 映射到 /mnt/c 的 Windows 命令不可用于 .deb。
missing_tools=()
for tool in python3 dpkg-deb node npm git cmake c++ make; do
    tool_path="$(command -v "$tool" 2>/dev/null || true)"
    if [ -z "$tool_path" ]; then
        missing_tools+=("$tool: missing")
    elif [[ "$tool_path" == /mnt/* ]]; then
        missing_tools+=("$tool: Windows interop path $tool_path")
    fi
done
if [ "${#missing_tools[@]}" -gt 0 ]; then
    echo "错误: Linux .deb 发布工具链不完整:"
    printf '  - %s\n' "${missing_tools[@]}"
    echo "  Ubuntu/Debian 基础依赖: sudo apt install build-essential cmake git nodejs npm python3-venv dpkg-dev"
    exit 1
fi

if ! python3 -c 'import venv' >/dev/null 2>&1; then
    echo "错误: 当前 Python 缺少 venv 模块。请安装 python3-venv。"
    exit 1
fi
NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || true)"
if ! [[ "$NODE_MAJOR" =~ ^[0-9]+$ ]] || [ "$NODE_MAJOR" -lt 18 ]; then
    echo "错误: 发布构建需要 Linux 原生 Node.js 18+，当前主版本: ${NODE_MAJOR:-unknown}"
    exit 1
fi
if [ -z "${QLH_SIGNING_KEY:-}" ] || [ ! -f "$QLH_SIGNING_KEY" ] || [ ! -r "$QLH_SIGNING_KEY" ]; then
    echo "错误: QLH_SIGNING_KEY 必须指向可读的发布私钥文件。"
    exit 1
fi

echo "================================================================"
echo "  QLH 边缘推理系统 — .deb 打包"
echo "  版本: $VERSION"
echo "  变体: $VARIANT"
echo "  输出: $SCRIPT_DIR"
echo "================================================================"
echo ""

echo "[preflight] Linux 原生发布工具链、Node.js ${NODE_MAJOR} 和签名 key 可用。"
if [ "$PREFLIGHT_ONLY" = "--preflight-only" ]; then
    exit 0
fi

echo "[toolchain] 构建并校验固定 revision 的 llama-quantize..."
python3 "$PROJECT_ROOT/scripts/build_llama_quantize.py" \
    --output-root "$MODEL_TOOL_ROOT" \
    --json
if [ ! -x "$MODEL_TOOL_PACKAGE/llama-quantize" ] || [ ! -f "$MODEL_TOOL_PACKAGE/manifest.json" ]; then
    echo "错误: Linux llama-quantize 受管包缺失或不可执行。"
    exit 1
fi

# ---- 清理旧构建 ----
[ -n "$BUILD_DIR" ] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ---- 1. 构建前端 ----
echo "[1/6] 构建前端..."
cd "$FRONTEND_DIR"
# 发布构建必须遵循已提交的 lockfile，不能在不同 npm 版本间重写它。
npm ci --silent
npx vite build
cd "$PROJECT_ROOT"

# ---- 2. 创建目录结构 ----
echo "[2/6] 创建安装目录结构..."
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/bin"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/src"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/frontend/dist"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/models"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/logs"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/model-tools/llama-quantize/linux-x86_64"
mkdir -p "$BUILD_DIR/usr/share/applications"
mkdir -p "$BUILD_DIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$BUILD_DIR/lib/systemd/system"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/local/bin"
mkdir -p "$BUILD_DIR/usr/sbin"

# ---- 3. 复制源码和前端 ----
echo "[3/6] 复制应用文件..."
# 用 tar 管道排除 __pycache__ / *.pyc（避免带入 Windows 侧的 cpython-312 缓存）
( cd "$SRC_DIR" && tar cf - --exclude='__pycache__' --exclude='*.pyc' . ) | \
    ( cd "$BUILD_DIR/opt/qlh-edge-inference/src/" && tar xf - )
cp -r "$FRONTEND_DIR/dist"/* "$BUILD_DIR/opt/qlh-edge-inference/frontend/dist/"
cp "$SCRIPT_DIR/launcher.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-app"
chmod 755 "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-app"
cp "$PACKAGING_DIR/qlh_launcher.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-launcher"
cp "$PACKAGING_DIR/diagnose.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/diagnose.py"
cp "$PACKAGING_DIR/repair.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/repair.py"
cp "$PACKAGING_DIR/data_retention.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/data_retention.py"
cp "$PACKAGING_DIR/update_core.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/update_core.py"
cp "$PACKAGING_DIR/updater.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/updater.py"
cp "$PACKAGING_DIR/version_store.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/version_store.py"
cp "$PACKAGING_DIR/launcher_slots.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/launcher_slots.py"
cp "$PACKAGING_DIR/install_manifest.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/install_manifest.py"
cp -a "$MODEL_TOOL_PACKAGE/." "$BUILD_DIR/opt/qlh-edge-inference/model-tools/llama-quantize/linux-x86_64/"
chmod 755 "$BUILD_DIR/opt/qlh-edge-inference/model-tools/llama-quantize/linux-x86_64/llama-quantize"
# UP-N2 可信发布：验签器与内置信任集必须随 Launcher 分发
cp "$PACKAGING_DIR/signing.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/signing.py"
cp -r "$PACKAGING_DIR/pubkeys" "$BUILD_DIR/opt/qlh-edge-inference/bin/pubkeys"
chmod 755 "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-launcher"
# Desktop and bjtu invoke qlh-launcher through its shebang; force the package venv.
sed -i '1c#!/opt/qlh-edge-inference/venv/bin/python3' "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-launcher"
cp "$SCRIPT_DIR/bjtu" "$BUILD_DIR/opt/qlh-edge-inference/bin/bjtu"
chmod 755 "$BUILD_DIR/opt/qlh-edge-inference/bin/bjtu"
cp "$SCRIPT_DIR/qlh-env-register" "$BUILD_DIR/usr/sbin/qlh-env-register"
chmod 755 "$BUILD_DIR/usr/sbin/qlh-env-register"
printf '%s\n' "$VERSION" > "$BUILD_DIR/opt/qlh-edge-inference/version.txt"
# 将旧 launcher.py 复制为应用包装器引用的模块；新 qlh-launcher 仅负责 bootstrap
cp "$PACKAGING_DIR/launcher.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/__launcher_main__.py"

# 复制 requirements 文件（供 postinst 重建 venv 参考）
cp "$PACKAGING_DIR/requirements-cpu.txt" "$BUILD_DIR/opt/qlh-edge-inference/"

# 复制桌面和服务文件
cp "$SCRIPT_DIR/qlh-edge-inference.desktop" "$BUILD_DIR/usr/share/applications/"
cp "$SCRIPT_DIR/qlh-edge-inference.service" "$BUILD_DIR/lib/systemd/system/"

# 图标（如果有 PNG 版本则复制，否则创建占位符）
if [ -f "$SCRIPT_DIR/qlh.png" ]; then
    cp "$SCRIPT_DIR/qlh.png" "$BUILD_DIR/usr/share/icons/hicolor/256x256/apps/"
else
    echo "  [注意] 未找到 qlh.png 图标文件，跳过图标安装。"
    echo "    请从 leds.ico 转换为 PNG 并放到 packaging/linux/qlh.png"
fi

# ---- 4. 创建虚拟环境并安装依赖 ----
echo "[4/6] 安装 Python 依赖..."
VENV_DIR="$BUILD_DIR/opt/qlh-edge-inference/venv"
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3。请安装 Python 3 和 python3-venv。"
    exit 1
fi

# --without-pip 只能绕开 ensurepip；若 Python 没有 venv 模块，无法在用户态创建 venv。
if ! python3 -c 'import venv' &> /dev/null; then
    echo "错误: 当前 Python 缺少 venv 模块。"
    echo "  有 sudo: sudo apt install python3-venv"
    echo "  无 sudo: 请使用自带 venv 模块的用户态 Python 发行版后重试。"
    exit 1
fi

if ! python3 -m venv --copies "$VENV_DIR" 2>/dev/null; then
    # 仅兼容 ensurepip 不可用的 Python；venv 模块已在上方确认存在。
    echo "  [兼容] ensurepip 不可用，改用 --without-pip + get-pip.py 引导 venv"
    rm -rf "$VENV_DIR"
    python3 -m venv --copies --without-pip "$VENV_DIR"
    GETPIP_FILE="$BUILD_DIR/get-pip.py"
    if ! command -v curl &> /dev/null; then
        echo "错误: ensurepip 不可用且未找到 curl，无法下载 get-pip.py。"
        echo "  请安装 curl，或使用包含 ensurepip 的 Python。"
        exit 1
    fi
    curl -fsSL --retry 3 https://bootstrap.pypa.io/get-pip.py -o "$GETPIP_FILE"
    "$VENV_DIR/bin/python" "$GETPIP_FILE" -q
fi
VENV_PIP="$VENV_DIR/bin/pip"

if [ "$VARIANT" == "cuda" ]; then
    echo "  安装 CUDA 版 PyTorch..."
    "$VENV_PIP" install torch  # 默认 CUDA 12.x
else
    echo "  安装 CPU-only PyTorch..."
    "$VENV_PIP" install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "  安装共享依赖..."
"$VENV_PIP" install -r "$PACKAGING_DIR/requirements-cpu.txt"

# UP-N6.0：扫描最终应用树，签名后立即以发布信任集复验并原子落盘。
"$VENV_DIR/bin/python" "$PACKAGING_DIR/install_manifest.py" build \
    --root "$BUILD_DIR/opt/qlh-edge-inference" \
    --app-id qlh-edge-inference \
    --version "$VERSION" \
    --platform linux \
    --variant "$VARIANT" \
    --package-kind application \
    --key "$QLH_SIGNING_KEY" \
    --trusted-keys-dir "$PACKAGING_DIR/pubkeys"

# ---- 5. 打包 DEBIAN 控制文件 ----
echo "[5/6] 创建包元数据..."
if [ "$VARIANT" == "cuda" ]; then
    PKG_NAME="qlh-edge-inference-cuda"
    CONTROL_FILE="$SCRIPT_DIR/control-cuda"
else
    PKG_NAME="qlh-edge-inference-cpu"
    CONTROL_FILE="$SCRIPT_DIR/control-cpu"
fi

# 更新 control 文件中的版本号
sed "s/^Version:.*/Version: $VERSION/" "$CONTROL_FILE" > "$BUILD_DIR/DEBIAN/control"
cp "$SCRIPT_DIR/postinst" "$BUILD_DIR/DEBIAN/"
cp "$SCRIPT_DIR/prerm" "$BUILD_DIR/DEBIAN/"
cp "$SCRIPT_DIR/postrm" "$BUILD_DIR/DEBIAN/"
chmod 755 "$BUILD_DIR/DEBIAN/postinst" "$BUILD_DIR/DEBIAN/prerm" "$BUILD_DIR/DEBIAN/postrm"

# ---- 6. 构建 .deb ----
echo "[6/6] 构建 .deb 包..."
DEB_FILE="${PKG_NAME}_${VERSION}_${ARCH}.deb"
dpkg-deb --build "$BUILD_DIR" "$SCRIPT_DIR/$DEB_FILE"

# 清理
rm -rf "$BUILD_DIR"

echo ""
echo "================================================================"
echo "  ✅ 打包完成！"
echo "  $SCRIPT_DIR/$DEB_FILE"
echo "================================================================"
ls -lh "$SCRIPT_DIR/$DEB_FILE"
