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


def _shell_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
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
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    return f"/{drive}/{resolved.as_posix()[3:]}"


@pytest.fixture
def bash_path():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise Linux package scripts")
    return bash


def _run_helper(bash_path: str, action: str, *, state_dir: Path, profile_file: Path):
    environment = os.environ.copy()
    environment.update(
        {
            "QLH_ENVREG_STATE_DIR": _shell_path(state_dir),
            "QLH_ENVREG_PROFILE_FILE": _shell_path(profile_file),
            "QLH_ENVREG_APP_BIN": "/opt/qlh-edge-inference/bin",
        }
    )
    if os.name == "nt":
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in environment.items()
            if key.startswith("QLH_ENVREG_")
        )
        command = f"{assignments} {shlex.quote(_shell_path(HELPER))} {shlex.quote(action)}"
        arguments = [bash_path, "-c", command]
        environment = os.environ.copy()
    else:
        arguments = [bash_path, _shell_path(HELPER), action]

    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_linux_environment_registration_is_explicit_and_idempotent(tmp_path, bash_path):
    state_dir = tmp_path / "state"
    profile_file = tmp_path / "profile.d" / "qlh.sh"

    enabled = _run_helper(bash_path, "enable", state_dir=state_dir, profile_file=profile_file)
    assert enabled.returncode == 0, enabled.stderr
    assert (state_dir / "env-register").read_text(encoding="utf-8") == "enabled\n"
    profile = profile_file.read_text(encoding="utf-8")
    assert profile.count("/opt/qlh-edge-inference/bin") == 2

    repeated = _run_helper(bash_path, "enable", state_dir=state_dir, profile_file=profile_file)
    assert repeated.returncode == 0, repeated.stderr
    assert profile_file.read_text(encoding="utf-8") == profile

    status = _run_helper(bash_path, "status", state_dir=state_dir, profile_file=profile_file)
    assert status.stdout == "enabled\n"

    disabled = _run_helper(bash_path, "disable", state_dir=state_dir, profile_file=profile_file)
    assert disabled.returncode == 0, disabled.stderr
    assert (state_dir / "env-register").read_text(encoding="utf-8") == "disabled\n"
    assert not profile_file.exists()


def test_linux_environment_registration_profile_keeps_path_deduplicated(tmp_path, bash_path):
    state_dir = tmp_path / "state"
    profile_file = tmp_path / "profile.d" / "qlh.sh"
    assert _run_helper(bash_path, "enable", state_dir=state_dir, profile_file=profile_file).returncode == 0

    runner = tmp_path / "apply-profile.sh"
    profile = profile_file.read_text(encoding="utf-8")
    runner.write_text(
        profile + profile + 'printf "%s" "$PATH"\n',
        encoding="utf-8",
        newline="\n",
    )
    shell = subprocess.run(
        [bash_path, _shell_path(runner)],
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
