import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_DIR = PROJECT_ROOT / "packaging"
LINUX_DIR = PACKAGING_DIR / "linux"
HELPER = LINUX_DIR / "qlh-env-register"


def _bash_family(bash_path: str) -> str:
    """判定 bash 解释器家族：'msys'（Git Bash/MSYS2）| 'wsl' | 'unknown'。

    Windows 上 shutil.which('bash') 可能解析到 Git Bash（MINGW64_NT）或
    WSL 启动器（Linux），两者路径语义不同：MSYS 认 /g/...，WSL 认 /mnt/g/...。
    """
    if os.name != "nt":
        return "posix"
    try:
        result = subprocess.run(
            [bash_path, "-c", "uname -s"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        name = (result.stdout or "").strip().lower()
    except Exception:
        return "unknown"
    if name.startswith(("mingw", "msys")):
        return "msys"
    if name == "linux":
        return "wsl"
    return "unknown"


def _shell_path(path: Path, family: str) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    posix = resolved.as_posix()
    if family == "wsl":
        # WSL 挂载点为 /mnt/<drive>/...，不认 MSYS 的 /<drive>/...
        return f"/mnt/{drive}/{posix[3:]}" if drive else posix
    # MSYS/unknown：cygpath 优先（与 bash 同源的转换器），回退 /<drive>/...
    cygpath = shutil.which("cygpath")
    if cygpath:
        converted = subprocess.run(
            [cygpath, "-u", str(resolved)],
            check=False,
            capture_output=True,
            text=True,
        )
        if converted.returncode == 0 and converted.stdout.strip():
            return converted.stdout.strip()
    if not drive:
        return posix
    return f"/{drive}/{posix[3:]}"


@pytest.fixture
def bash_path():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise Linux package scripts")
    return bash


@pytest.fixture
def bash_family(bash_path):
    """bash 解释器家族（msys/wsl/posix/unknown），供路径映射选择。"""
    return _bash_family(bash_path)


def _run_helper(bash_path: str, action: str, *, state_dir: Path, profile_file: Path, family: str):
    environment = os.environ.copy()
    environment.update(
        {
            "QLH_ENVREG_STATE_DIR": _shell_path(state_dir, family),
            "QLH_ENVREG_PROFILE_FILE": _shell_path(profile_file, family),
            "QLH_ENVREG_APP_BIN": "/opt/qlh-edge-inference/bin",
        }
    )
    if os.name == "nt":
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in environment.items()
            if key.startswith("QLH_ENVREG_")
        )
        command = f"{assignments} {shlex.quote(_shell_path(HELPER, family))} {shlex.quote(action)}"
        arguments = [bash_path, "-c", command]
        environment = os.environ.copy()
    else:
        arguments = [bash_path, _shell_path(HELPER, family), action]

    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_linux_environment_registration_is_explicit_and_idempotent(tmp_path, bash_path, bash_family):
    state_dir = tmp_path / "state"
    profile_file = tmp_path / "profile.d" / "qlh.sh"

    enabled = _run_helper(bash_path, "enable", state_dir=state_dir, profile_file=profile_file, family=bash_family)
    assert enabled.returncode == 0, enabled.stderr
    assert (state_dir / "env-register").read_text(encoding="utf-8") == "enabled\n"
    profile = profile_file.read_text(encoding="utf-8")
    assert profile.count("/opt/qlh-edge-inference/bin") == 2

    repeated = _run_helper(bash_path, "enable", state_dir=state_dir, profile_file=profile_file, family=bash_family)
    assert repeated.returncode == 0, repeated.stderr
    assert profile_file.read_text(encoding="utf-8") == profile

    status = _run_helper(bash_path, "status", state_dir=state_dir, profile_file=profile_file, family=bash_family)
    assert status.stdout == "enabled\n"

    disabled = _run_helper(bash_path, "disable", state_dir=state_dir, profile_file=profile_file, family=bash_family)
    assert disabled.returncode == 0, disabled.stderr
    assert (state_dir / "env-register").read_text(encoding="utf-8") == "disabled\n"
    assert not profile_file.exists()


def test_linux_environment_registration_profile_keeps_path_deduplicated(tmp_path, bash_path, bash_family):
    state_dir = tmp_path / "state"
    profile_file = tmp_path / "profile.d" / "qlh.sh"
    assert _run_helper(bash_path, "enable", state_dir=state_dir, profile_file=profile_file, family=bash_family).returncode == 0

    runner = tmp_path / "apply-profile.sh"
    profile = profile_file.read_text(encoding="utf-8")
    runner.write_text(
        profile + profile + 'printf "%s" "$PATH"\n',
        encoding="utf-8",
        newline="\n",
    )
    shell = subprocess.run(
        [bash_path, _shell_path(runner, bash_family)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert shell.returncode == 0, shell.stderr
    segments = shell.stdout.split(":")
    assert segments[0] == "/opt/qlh-edge-inference/bin"
    assert segments.count("/opt/qlh-edge-inference/bin") == 1


def test_linux_package_environment_registration_contract_is_wired():
    helper = HELPER.read_text(encoding="utf-8")
    build = (LINUX_DIR / "build-deb.sh").read_text(encoding="utf-8")
    postinst = (LINUX_DIR / "postinst").read_text(encoding="utf-8")
    postrm = (LINUX_DIR / "postrm").read_text(encoding="utf-8")

    assert 'QLH_ENVREG=1' in postinst
    assert 'QLH_ENVREG=1 or =0' in postinst
    assert '/usr/sbin/qlh-env-register enable' in postinst
    assert '/usr/sbin/qlh-env-register apply' in postinst
    assert 'qlh-env-register" "$BUILD_DIR/usr/sbin/qlh-env-register"' in build
    assert 'rm -f "$ENV_PROFILE_FILE"' in postrm
    assert 'rm -f "$ENV_STATE_FILE"' in postrm
    assert 'case "${PATH}:"' not in helper
    assert 'case ":${PATH}:" in' in helper

    for installer in ("setup.iss", "setup-cuda.iss"):
        source = (PACKAGING_DIR / installer).read_text(encoding="utf-8")
        assert "ChangesEnvironment=yes" in source
        assert 'Name: "envreg"' in source
        assert 'EnvRegistrationParameterIsValid' in source
        assert 'ConfigureRegisteredUserEnvironment' in source
        assert 'RemoveRegisteredUserEnvironment' in source
