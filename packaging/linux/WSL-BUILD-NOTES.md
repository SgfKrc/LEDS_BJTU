# WSL 环境下 .deb 打包踩坑记录

> 适用场景：Windows 开发机 + WSL2 (Ubuntu 22.04) 执行 `packaging/linux/build-deb.sh cpu|cuda`。
> 以下坑均为实际构建中踩过并解决的，含无 sudo 权限的绕行方案。

## 0. 推荐做法（有 sudo 权限）

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-tk dpkg-dev \
     build-essential cmake zenity
```

装齐后 `build-deb.sh` 原样可用（`llama-cpp-python` 会从源码编译，需要
`build-essential` + `cmake`）。

---

## 1. WSL2 的 /tmp 是 tmpfs，会话结束即清空

- **现象**：下载好的 wheel / get-pip.py / venv，在下一次 `wsl.exe` 调用时全部消失；初次打包因此失败（pip 回退到 index 重新拉 sdist）。
- **原因**：WSL2 的 `/tmp` 是 tmpfs（内存盘）。最后一个 wsl 进程退出后 VM 停止，tmpfs 内容清空。
- **解决**：构建中间产物放 `~/`（ext4 持久盘）；下载 + 构建放同一次 wsl 会话内完成。

## 2. 缺 python3-pip / python3.10-venv（ensurepip 不可用）

- **现象**：
  - `python3 -m venv` 报 `The virtual environment was not created successfully because ensurepip is not available`
  - `python3 -m pip` 报 `No module named pip`
- **原因**：Ubuntu 22.04 精简安装可能不装 `python3-pip` 或 `python3-venv`。两者不是一回事：
  `python3-pip` 缺失时可由 `get-pip.py` 补齐；`python3-venv` 缺失时，Python 连创建
  虚拟环境所需的 `venv` 模块都没有，不能靠 `get-pip.py` 修复。
- **解决**：
  - 有 sudo：`sudo apt install python3-pip python3.10-venv`
  - 无 sudo 且 `python3 -c 'import venv'` 成功：`build-deb.sh` 会自动使用
    `venv --without-pip`，再从 `https://bootstrap.pypa.io/get-pip.py` 引导 pip（需要 `curl`）。
  - 无 sudo 且 `venv` 模块也缺失：需要改用一个自带 `venv` 的用户态 Python 发行版，或请管理员
    安装 `python3-venv`；脚本会在安装依赖前给出明确错误，避免留下半成品环境。

## 3. llama-cpp-python 在 PyPI 只有 sdist，无编译工具必失败

- **现象**：`pip install` 拉 71 MB sdist 后开始编译，报
  `Could not find the compiler specified in the environment variable CC: x86_64-linux-gnu-gcc`。
- **原因**：`llama-cpp-python` 官方只把预编译 wheel 发布在 GitHub Releases，PyPI 上仅 sdist。
- **解决**（无编译工具时，直接用官方 manylinux wheel）：

  ```bash
  mkdir -p ~/qlh-build/wheels
  cd ~/qlh-build/wheels
  curl -sL -o llama_cpp_python-0.3.34-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
    "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34/llama_cpp_python-0.3.34-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
  export PIP_FIND_LINKS="$HOME/qlh-build/wheels"
  ./build-deb.sh cpu
  ```

  - wheel 版本需满足 `requirements-cpu.txt` 的 `>=0.3.0`；0.3.x 的 wheel 是
    `py3-none`（ctypes 加载 `.so`），cp310 可用。
- **注意**：`PIP_FIND_LINKS` 指向**不存在**的路径时 pip 会静默忽略并回退到 index
  的 sdist（首次踩坑：路径在 /tmp 被清空后，pip 直接重新拉 sdist 编译）。

## 4. WSL 里不能执行 Windows 的 node.exe

- **现象**：即使 `/mnt/c/Program Files/nodejs/` 在 PATH 里，`node --version` 仍报
  `command not found` 或 `cannot execute binary file: Exec format error`。
- **原因**：该 WSL 发行版未启用 binfmt 互操作，`.exe` 无法直接执行（`npm` 这类
  shell shim 能跑，但最终调 node 会失败）。
- **解决**：装原生 Linux Node（无需 root）：

  ```bash
  curl -sL -o /tmp/node.tar.xz https://nodejs.org/dist/v20.19.4/node-v20.19.4-linux-x64.tar.xz
  tar -xf /tmp/node.tar.xz -C ~/
  mv ~/node-v20.19.4-linux-x64 ~/node
  export PATH="$HOME/node/bin:$PATH"
  ```

## 5. WSL 与 Windows 的 npm 版本不同

- **背景**：WSL 侧 npm（10.x）与 Windows 侧 npm（11.x）对 lockfile 的规范化可能不同。
- **现行处理**：`build-deb.sh` 使用 `npm ci` 而不是 `npm install`，只按已提交的
  `frontend/package-lock.json` 安装依赖，不应重写锁文件。
- **异常处理**：若构建后锁文件仍出现改动，先确认改动来源和内容；它不是打包流程应有的副作用，
  不应在未检查的情况下直接丢弃。

  ```bash
  git diff -- frontend/package-lock.json
  ```

## 6. Windows 侧 pyc 残留会被打进 deb

- **现象**：deb 里出现 `src/**/__pycache__/*.cpython-312.pyc`。
- **原因**：Windows 侧运行 Python 留下的缓存被 `cp -r "$SRC_DIR"/*` 全量复制。
- **解决**：`build-deb.sh` 已改为 tar 管道复制并排除 `__pycache__` / `*.pyc`。

## 7. 其他细节

- **图标**：deb 需要 PNG。`packaging/linux/qlh.png` 由 `leds.ico` 转换（ico 内最大
  帧 128×128，放大到 256）：

  ```bash
  python -c "from PIL import Image; Image.open('packaging/leds.ico').convert('RGBA').resize((256,256), Image.LANCZOS).save('packaging/linux/qlh.png')"
  ```

- **磁盘/内存**：torch CPU wheel 约 192 MB、llama-cpp-python wheel 23 MB；构建时
  建议预留 5 GB+ 磁盘（deb 解包后 venv 完整环境约 1.5 GB+）。
- **产物忽略**：`packaging/linux/*.deb` 已加入 `.gitignore`。
- **安装验证需要 root**：`sudo dpkg -i qlh-edge-inference-cpu_0.1.8.1_amd64.deb`，
  随后 `sudo apt-get install -f` 补齐依赖。
