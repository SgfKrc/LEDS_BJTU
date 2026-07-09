#!/bin/bash
# ================================================================
# QLH 边缘推理系统 — Linux .deb 打包脚本
# ================================================================
# 用法:
#   集显版: ./build-deb.sh cpu
#   独显版: ./build-deb.sh cuda
#
# 前置条件:
#   1. Ubuntu 22.04+ / Debian 12+
#   2. python3, python3-venv, python3-pip 已安装
#   3. Node.js 18+ (前端构建)
#   4. dpkg-deb 可用
#
# 输出:
#   packaging/linux/qlh-edge-inference-cpu_0.1.7_amd64.deb
#   packaging/linux/qlh-edge-inference-cuda_0.1.7_amd64.deb
# ================================================================

set -euo pipefail

VARIANT="${1:-cpu}"
VERSION="0.1.7"
ARCH="amd64"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PACKAGING_DIR="$PROJECT_ROOT/packaging"
SRC_DIR="$PROJECT_ROOT/src"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
BUILD_DIR="/tmp/qlh-deb-build"

# 前置检查
if ! command -v dpkg-deb &> /dev/null; then
    echo "错误: dpkg-deb 未找到。请安装 dpkg-dev 包。"
    echo "  sudo apt install dpkg-dev"
    exit 1
fi

echo "================================================================"
echo "  QLH 边缘推理系统 — .deb 打包"
echo "  版本: $VERSION"
echo "  变体: $VARIANT"
echo "  输出: $SCRIPT_DIR"
echo "================================================================"
echo ""

# ---- 清理旧构建 ----
[ -n "$BUILD_DIR" ] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# ---- 1. 构建前端 ----
echo "[1/6] 构建前端..."
cd "$FRONTEND_DIR"
npm install --silent
npx vite build
cd "$PROJECT_ROOT"

# ---- 2. 创建目录结构 ----
echo "[2/6] 创建安装目录结构..."
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/bin"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/src"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/frontend/dist"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/models"
mkdir -p "$BUILD_DIR/opt/qlh-edge-inference/logs"
mkdir -p "$BUILD_DIR/usr/share/applications"
mkdir -p "$BUILD_DIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$BUILD_DIR/lib/systemd/system"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/local/bin"

# ---- 3. 复制源码和前端 ----
echo "[3/6] 复制应用文件..."
cp -r "$SRC_DIR"/* "$BUILD_DIR/opt/qlh-edge-inference/src/"
cp -r "$FRONTEND_DIR/dist"/* "$BUILD_DIR/opt/qlh-edge-inference/frontend/dist/"
cp "$SCRIPT_DIR/launcher.py" "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-launcher"
chmod 755 "$BUILD_DIR/opt/qlh-edge-inference/bin/qlh-launcher"
# 将 launcher.py 复制为启动器包装器引用的模块
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
python3 -m venv --copies "$BUILD_DIR/opt/qlh-edge-inference/venv"
VENV_PIP="$BUILD_DIR/opt/qlh-edge-inference/venv/bin/pip"

if [ "$VARIANT" == "cuda" ]; then
    echo "  安装 CUDA 版 PyTorch..."
    "$VENV_PIP" install torch  # 默认 CUDA 12.x
else
    echo "  安装 CPU-only PyTorch..."
    "$VENV_PIP" install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "  安装共享依赖..."
"$VENV_PIP" install -r "$PACKAGING_DIR/requirements-cpu.txt"

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
