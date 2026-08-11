"""T9.6 主应用包聊天页的静态发布契约。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = PROJECT_ROOT / "packaging"


def test_primary_package_declares_and_builds_chat_companion_before_signing():
    requirements = (PACKAGING_DIR / "requirements-cpu.txt").read_text(encoding="utf-8")
    spec = (PACKAGING_DIR / "qlh-tui-chat.spec").read_text(encoding="utf-8")

    assert "textual==8.2.8" in requirements
    assert "from PyInstaller.utils.hooks import collect_submodules" not in spec
    assert '"textual.drivers.windows_driver"' in spec
    assert '"IPython"' in spec
    assert '"matplotlib"' in spec
    assert '"PyQt5"' in spec
    assert 'name="QLH-TUI-Chat"' in spec
    assert "console=True" in spec

    for variant, output_dir in (
        ("cpu", "dist\\QLH-Edge-Inference"),
        ("cuda", "dist\\QLH-Edge-Inference-CUDA"),
    ):
        build_path = PACKAGING_DIR / f"build-{variant}.bat"
        build_bytes = build_path.read_bytes()
        build = build_bytes.decode("utf-8")
        chat_build = build.index("packaging\\qlh-tui-chat.spec")
        release_copy = build.index(f'copy /y "bjtu.bat" "{output_dir}\\bjtu.bat"')
        manifest_build = build.index("packaging\\install_manifest.py build")
        expected_exe = f'{output_dir}\\QLH-TUI-Chat\\QLH-TUI-Chat.exe'

        assert build_bytes.count(b"\n") == build_bytes.count(b"\r\n")
        assert chat_build < release_copy < manifest_build
        assert f'--distpath "{output_dir}"' in build
        assert expected_exe in build
        assert f'copy /y "bjtu.bat" "{output_dir}\\bjtu.bat"' in build
        assert f'copy /y "README.md" "{output_dir}\\docs\\README.md"' in build
        assert f'copy /y "packaging\\scripts\\convert_to_gguf.py" "{output_dir}\\tools\\convert_to_gguf.py"' in build
        assert "--include" not in build
        assert "QLH_NONINTERACTIVE" in build

        if variant == "cuda":
            assert "torch.version.cuda" in build
            assert "torch.cuda.is_available()" not in build


def test_packaged_bjtu_uses_only_the_signed_chat_payload():
    windows_path = PROJECT_ROOT / "bjtu.bat"
    windows_bytes = windows_path.read_bytes()
    windows = windows_bytes.decode("utf-8")
    linux = (PACKAGING_DIR / "linux" / "bjtu").read_text(encoding="utf-8")
    deb_build = (PACKAGING_DIR / "linux" / "build-deb.sh").read_text(encoding="utf-8")

    assert 'if not exist "%~dp0QLH-TUI-Chat\\QLH-TUI-Chat.exe"' in windows
    assert '"%~dp0QLH-TUI-Chat\\QLH-TUI-Chat.exe" %2 %3 %4 %5 %6' in windows
    assert "QLH_CHAT_EXE" not in windows
    assert windows_bytes.count(b"\n") == windows_bytes.count(b"\r\n")
    assert "APP_DIR=\"/opt/qlh-edge-inference\"" in linux
    assert 'exec "$APP_DIR/venv/bin/python" "$APP_DIR/src/tui_chat.py" "$@"' in linux
    assert '"$VENV_PIP" install -r "$PACKAGING_DIR/requirements-cpu.txt"' in deb_build
    assert "--preflight-only" in deb_build
    assert '[[ "$tool_path" == /mnt/* ]]' in deb_build
    assert 'NODE_MAJOR' in deb_build
