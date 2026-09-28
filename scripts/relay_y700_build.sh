#!/data/data/com.termux/files/usr/bin/bash
# relay_y700_build.sh —— 在 Android/Termux **设备上**编译 keep-head 组件（★ 把三机验收踩过的坑固化）
#
# 为什么需要它（`docs/边缘最小发行包审计清单-2026-09-26.md` 的「三机验收结果」）：
# 2026-09-28 的三机验收里，Y700 跑不了 hybrid，根因是**设备上的 llama.cpp 是 b8054（9-21 构建）**，
# 与我们当前源码的 shim **ABI 不匹配** ⇒ `GGML_ASSERT(lid < model.hparams.n_layer())` 崩溃。
# 判据很硬：**新 shim 配 qwen2 也崩** ⇒ 是 ABI 问题、不是 hybrid 特有。
# 修法是在设备上**原生编译当前 llama.cpp**（Termux 的 clang + cmake 都现成，**不需要 NDK**），
# 但过程中有三个坑，本文档把它们固化：
#
#   坑 1：新版 `libllama.so` **带 SONAME**（`libllama.so.0`）⇒ 不建符号链接会
#         `dlopen failed: library "libllama.so.0" not found`。
#   坑 2：`cp` 复制**符号链接**只会得到一个十几字节的文件 ⇒ 必须 `cp -L`（解引用）。
#   坑 3：服务在 Termux 里**必须** `setsid` + `termux-wake-lock`，否则会被 Android 杀掉
#         （现象很像"模型算错"：连接被中止）。
#
# 用法（设备侧）::
#
#     bash relay_y700_build.sh help
#     bash relay_y700_build.sh shim                 # 只重编 shim（假定 llama.cpp 已就绪）
#     bash relay_y700_build.sh llamacpp <tar.gz>    # 编译 llama.cpp（源码包由主机侧打包传入）
#     bash relay_y700_build.sh install <build-dir>  # 装新库到 ./bin（含 SONAME 链接 + cp -L）
#
# ⚠️ 本脚本刻意**只做构建与安装**，不碰服务启停 —— 启停见 `relay_mid_service.py` 的调用方式
#    （`setsid` + `termux-wake-lock`，见文末注释）。
set -uo pipefail

KEEPHEAD_DIR="${KEEPHEAD_DIR:-$HOME/qlh-keephead}"
BIN_DIR="$KEEPHEAD_DIR/bin"
INC_DIR="$KEEPHEAD_DIR/inc"
SHIM_SRC="$KEEPHEAD_DIR/qlh_keep_head.c"

# 与 `.venv-test` 无关：设备侧只用系统 clang。
CLANG="${CLANG:-clang}"

die() { echo "FAIL: $*" >&2; exit 1; }
note() { echo "[build] $*"; }

require_tools() {
    local missing=()
    for t in "$CLANG" make; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        note "缺工具：${missing[*]} ⇒ 先装： pkg install -y clang make cmake"
        return 1
    fi
    return 0
}

# --- 只重编 shim ---------------------------------------------------------------
cmd_shim() {
    require_tools || exit 1
    [ -f "$SHIM_SRC" ] || die "找不到 shim 源码：$SHIM_SRC（由主机侧 scp 传入）"
    [ -d "$INC_DIR" ] || die "找不到 headers 目录：$INC_DIR（需要 llama.h + ggml*.h + gguf.h）"
    mkdir -p "$BIN_DIR"
    note "编译 shim（-O2，动态链接 $BIN_DIR/libllama.so）"
    # shellcheck disable=SC2086
    "$CLANG" -shared -fPIC -O2 -I "$INC_DIR" \
        -o "$BIN_DIR/libqlh_keep_head.so.new" "$SHIM_SRC" \
        -L "$BIN_DIR" -lllama -Wl,-rpath,"$BIN_DIR" \
        || die "编译失败（见上）"
    note "产物：$BIN_DIR/libqlh_keep_head.so.new（$(stat -c%s "$BIN_DIR/libqlh_keep_head.so.new") 字节）"
    note "自检导出符号："
    nm -D "$BIN_DIR/libqlh_keep_head.so.new" 2>/dev/null | grep qlh_kh | awk '{print "    " $3}'
    note "★ 确认含 qlh_kh_last_error（R-R5 加的可观测性；旧 shim 没有它）"
    note "启用： cp $BIN_DIR/libqlh_keep_head.so.new $BIN_DIR/libqlh_keep_head.so"
}

# --- 编译 llama.cpp ------------------------------------------------------------
cmd_llamacpp() {
    local tarball="${1:-}"
    [ -n "$tarball" ] || die "用法：build llamacpp <llamacpp-src.tar.gz>"
    [ -f "$tarball" ] || die "找不到源码包：$tarball"
    require_tools || exit 1
    command -v cmake >/dev/null 2>&1 || die "缺 cmake ⇒ pkg install -y cmake"

    local src="$HOME/llamacpp-src" build="$HOME/build-llama"
    note "解压 $tarball"
    rm -rf "$src" "$build"; mkdir -p "$src"
    tar xzf "$tarball" -C "$src" || die "解压失败"

    note "cmake 配置（BUILD_SHARED_LIBS=ON；⚠️ 必须带 models/ 源码目录，否则 tools/mtmd 报 models/models.h 缺失）"
    mkdir -p "$build" && cd "$build" || die "无法进入 $build"
    cmake "$src" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DGGML_OPENMP=ON -G "Unix Makefiles" || die "cmake 配置失败"
    note "编译（-j$(nproc)，约 10–30 分钟）"
    make -j"$(nproc)" llama || die "make 失败"
    note "产物：$build/bin/"
    ls -la "$build"/bin/*.so 2>/dev/null | awk '{print "    " $5, $9}'
    note "下一步： bash $0 install $build"
}

# --- 安装到 ./bin（含 SONAME 链接 + cp -L）-------------------------------------
cmd_install() {
    local build="${1:-$HOME/build-llama}"
    [ -d "$build/bin" ] || die "找不到 $build/bin"
    mkdir -p "$BIN_DIR"

    note "备份旧库（只备份一次，后缀 .bak-<日期>）"
    local stamp; stamp="$(date +%Y%m%d)"
    for f in libllama.so libggml.so libggml-base.so libggml-cpu.so; do
        if [ -f "$BIN_DIR/$f" ] && [ ! -f "$BIN_DIR/$f.bak-$stamp" ]; then
            cp -L "$BIN_DIR/$f" "$BIN_DIR/$f.bak-$stamp" && note "  backed up $f → $f.bak-$stamp"
        fi
    done

    note "安装新库（★ 坑 2：用 cp -L 解引用符号链接，否则只会得到一个十几字节的文件）"
    for f in "$build"/bin/libllama.so "$build"/bin/libggml*.so; do
        [ -e "$f" ] || continue
        local b; b="$(basename "$f")"
        cp -L --remove-destination "$f" "$BIN_DIR/$b" && note "  $b → $(stat -c%s "$BIN_DIR/$b") 字节"
    done

    note "★ 坑 1：建 SONAME 符号链接（新版带版本号，否则 dlopen 失败）"
    for pair in "libllama.so:libllama.so.0" "libggml.so:libggml.so.0" \
                "libggml-base.so:libggml-base.so.0" "libggml-cpu.so:libggml-cpu.so.0"; do
        local src="${pair%%:*}"; local dst="${pair##*:}"
        [ -f "$BIN_DIR/$src" ] && ln -sf "$src" "$BIN_DIR/$dst" && note "  $dst -> $src"
    done

    note "校验实际 SONAME（应与上面建的链接一致）："
    for f in "$BIN_DIR"/libllama.so "$BIN_DIR"/libggml.so; do
        [ -f "$f" ] || continue
        printf "    %s SONAME=" "$(basename "$f")"
        readelf -d "$f" 2>/dev/null | grep SONAME | sed 's/.*\[\(.*\)\]/\1/'
    done
    note "自检（起不起得来取决于 libllama.so 能否被 dlopen）："
    LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}" python3 - <<'PY' 2>&1 | tail -3 || true
import sys
sys.path.insert(0, __import__("os").environ.get("KEEPHEAD_DIR", "."))
try:
    import relay_segment_info  # 零依赖，先确认它能 import
    print("  relay_segment_info OK")
except Exception as exc:
    print("  relay_segment_info FAIL:", exc)
PY
}

case "${1:-help}" in
    shim)     cmd_shim ;;
    llamacpp) shift; cmd_llamacpp "$@" ;;
    install)  shift; cmd_install "$@" ;;
    help|*)
        cat <<'EOF'
relay_y700_build.sh —— Android/Termux 设备上的 keep-head 组件构建（固化三机验收踩过的坑）

  shim                   只重编 shim（llama.cpp 已就绪时用）
  llamacpp <tar.gz>      编译 llama.cpp（源码包由主机侧 tar 打包传入）
  install <build-dir>    把新库装到 ~/qlh-keephead/bin（含 SONAME 链接 + cp -L）
  help

环境变量：KEEPHEAD_DIR（默认 ~/qlh-keephead）、CLANG（默认 clang）

★ 起服务的正确方式（坑 3：Android 会杀后台）：
    cd ~/qlh-keephead
    export LD_LIBRARY_PATH="$PWD/bin:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$PWD:$PWD/src"
    termux-wake-lock
    setsid python3 relay_mid_service.py --role middle --listen 127.0.0.1:50180 \
        --keep-head-shim "$PWD/bin/libqlh_keep_head.so" \
        --model /sdcard/Download/QLH/models/<段工件>.gguf \
        --mode nextn --threads 4 --n-seq-max 1 --n-batch 2048 \
        --ready-file "$PWD/mid.ready" --max-connections 8 \
        < /dev/null > "$PWD/mid-nohup.log" 2>&1 &
  ⚠️ --n-seq-max 1（单序列；用 8 会把 ctx 均分成 512，长 decode 会 rc=1）
EOF
        ;;
esac
