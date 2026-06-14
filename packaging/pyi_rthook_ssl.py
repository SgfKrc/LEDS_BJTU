"""
PyInstaller runtime hook — 确保 _ssl 在 psycopg2 之前加载。

psycopg2-binary 捆绑了自己的 OpenSSL DLL（libssl-3-x64-{hash}.dll），
这些 DLL 通过 os.add_dll_directory 注册后可能干扰 Python _ssl 模块的
OpenSSL 加载顺序，导致 "内存位置访问无效" 错误。

此 hook 在 Python 脚本执行之前运行，通过 ctypes 显式预加载 Python 的
OpenSSL DLL，确保 _ssl.pyd 导入时依赖 DLL 已在内存中。
"""
import sys
import os
import ctypes

_ifrozen = getattr(sys, 'frozen', False)
if _ifrozen:
    _internal_dir = os.path.join(os.path.dirname(sys.executable), '_internal')
    if os.path.isdir(_internal_dir):
        # 将 _internal 加入 DLL 搜索路径（优先级最高）
        try:
            os.add_dll_directory(_internal_dir)
        except Exception:
            pass

        # 按依赖顺序显式加载：libcrypto 先，libssl 后
        for _dll_name in ('libcrypto-3.dll', 'libssl-3.dll'):
            _dll_path = os.path.join(_internal_dir, _dll_name)
            if os.path.isfile(_dll_path):
                try:
                    ctypes.CDLL(_dll_path)
                except Exception:
                    pass

# 再通过 Python import 加载 ssl 模块（_ssl.pyd 会复用已加载的 DLL）
try:
    import ssl  # noqa: F401
except Exception:
    pass
