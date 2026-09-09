"""OS-level process sandbox for model metadata/template probes.

The probe worker may import repository-provided Python modules when a curated
model requires ``trust_remote_code``.  A dedicated virtualenv is not an OS
security boundary, so this module refuses to run the worker unless the host
can provide a restricted process/container.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class SandboxUnavailable(RuntimeError):
    """The current host cannot provide the required OS sandbox."""


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    sandbox: dict[str, Any]


def _metadata(backend: str, *, network_disabled: bool) -> dict[str, Any]:
    return {
        "available": True,
        "backend": backend,
        "os_level": True,
        "low_privilege": True,
        "network_disabled": network_disabled,
    }


def _sandbox_environment(env: Mapping[str, str], sandbox: dict[str, Any]) -> dict[str, str]:
    child_env = {str(key): str(value) for key, value in env.items()}
    child_env["QLH_OS_SANDBOX_BACKEND"] = str(sandbox["backend"])
    child_env["QLH_OS_SANDBOX_NETWORK_DISABLED"] = "1" if sandbox["network_disabled"] else "0"
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if sandbox["backend"].startswith(("bubblewrap-", "macos-")):
        child_env["HF_HOME"] = "/tmp/qlh-hf-home"
        child_env["TRANSFORMERS_CACHE"] = "/tmp/qlh-hf-home/transformers"
        child_env["HF_DATASETS_CACHE"] = "/tmp/qlh-hf-home/datasets"
    return child_env


def _run_bwrap(
    python: Path,
    worker: Path,
    *,
    input_text: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> SandboxResult:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable("bubblewrap is not installed")
    sandbox = _metadata("bubblewrap-user-network-mount-namespace", network_disabled=True)
    child_env = _sandbox_environment(env, sandbox)
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--ro-bind", "/", "/",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    snapshot = child_env.get("QLH_TEMPLATE_SNAPSHOT", "")
    if snapshot:
        command.extend(("--ro-bind", snapshot, snapshot))
    command.extend(("--chdir", str(cwd)))
    for key, value in sorted(child_env.items()):
        command.extend(("--setenv", key, value))
    command.extend(("--", str(python), "-u", str(worker)))
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            timeout=timeout_seconds,
            check=False,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(124, "", "sandbox worker timed out", sandbox)
    return SandboxResult(completed.returncode, completed.stdout, completed.stderr, sandbox)


def _run_macos(
    python: Path,
    worker: Path,
    *,
    input_text: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> SandboxResult:
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        raise SandboxUnavailable("sandbox-exec is not installed")
    sandbox = _metadata("macos-sandbox-exec-readonly", network_disabled=True)
    child_env = _sandbox_environment(env, sandbox)
    profile = (
        "(version 1) "
        "(deny default) "
        "(allow process-fork) "
        "(allow process-exec) "
        "(allow file-read*) "
        "(allow sysctl-read) "
        "(allow mach-lookup) "
        "(allow file-write* (subpath \"/tmp\")) "
        "(deny network*)"
    )
    command = [sandbox_exec, "-p", profile, str(python), "-u", str(worker)]
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            timeout=timeout_seconds,
            check=False,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(124, "", "sandbox worker timed out", sandbox)
    return SandboxResult(completed.returncode, completed.stdout, completed.stderr, sandbox)


if os.name == "nt":
    _DWORD = ctypes.c_uint32
    _HANDLE = wintypes.HANDLE
    _LPVOID = ctypes.c_void_p
    _LPWSTR = wintypes.LPWSTR

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", _DWORD),
            ("lpSecurityDescriptor", _LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    class _StartupInfo(ctypes.Structure):
        _fields_ = [
            ("cb", _DWORD),
            ("lpReserved", _LPWSTR),
            ("lpDesktop", _LPWSTR),
            ("lpTitle", _LPWSTR),
            ("dwX", _DWORD),
            ("dwY", _DWORD),
            ("dwXSize", _DWORD),
            ("dwYSize", _DWORD),
            ("dwXCountChars", _DWORD),
            ("dwYCountChars", _DWORD),
            ("dwFillAttribute", _DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
            ("hStdInput", _HANDLE),
            ("hStdOutput", _HANDLE),
            ("hStdError", _HANDLE),
        ]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", _HANDLE),
            ("hThread", _HANDLE),
            ("dwProcessId", _DWORD),
            ("dwThreadId", _DWORD),
        ]

    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", _LPVOID), ("Attributes", _DWORD)]

    class _TokenMandatoryLabel(ctypes.Structure):
        _fields_ = [("Label", _SidAndAttributes)]

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", _DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", _DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", _DWORD),
            ("SchedulingClass", _DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _close_handle(handle: Any) -> None:
    if os.name == "nt" and handle:
        ctypes.windll.kernel32.CloseHandle(handle)


def _free_local_sid(sid: Any) -> None:
    if os.name == "nt" and sid:
        ctypes.windll.kernel32.LocalFree(sid)


def _run_windows(
    python: Path,
    worker: Path,
    *,
    input_text: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> SandboxResult:
    """Run under a disabled-privilege, low-integrity restricted token."""
    if os.name != "nt":
        raise SandboxUnavailable("Windows sandbox requested on a non-Windows host")

    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = _HANDLE
    kernel32.CreatePipe.argtypes = [ctypes.POINTER(_HANDLE), ctypes.POINTER(_HANDLE), ctypes.POINTER(_SecurityAttributes), _DWORD]
    kernel32.SetHandleInformation.argtypes = [_HANDLE, _DWORD, _DWORD]
    kernel32.CreateJobObjectW.argtypes = [_LPVOID, _LPWSTR]
    kernel32.SetInformationJobObject.argtypes = [_HANDLE, ctypes.c_int, _LPVOID, _DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [_HANDLE, _HANDLE]
    kernel32.WaitForSingleObject.argtypes = [_HANDLE, _DWORD]
    kernel32.GetExitCodeProcess.argtypes = [_HANDLE, ctypes.POINTER(_DWORD)]
    kernel32.TerminateProcess.argtypes = [_HANDLE, _DWORD]
    advapi32.OpenProcessToken.argtypes = [_HANDLE, _DWORD, ctypes.POINTER(_HANDLE)]
    advapi32.CreateRestrictedToken.argtypes = [
        _HANDLE, _DWORD, _DWORD, _LPVOID, _DWORD, _LPVOID, _DWORD, _LPVOID,
        ctypes.POINTER(_HANDLE),
    ]
    advapi32.SetTokenInformation.argtypes = [_HANDLE, ctypes.c_int, _LPVOID, _DWORD]
    advapi32.ConvertStringSidToSidW.argtypes = [_LPWSTR, ctypes.POINTER(_LPVOID)]
    advapi32.GetLengthSid.argtypes = [_LPVOID]
    advapi32.CreateProcessAsUserW.argtypes = [
        _HANDLE, _LPWSTR, _LPWSTR, _LPVOID, _LPVOID, ctypes.wintypes.BOOL,
        _DWORD, _LPVOID, _LPWSTR, ctypes.POINTER(_StartupInfo),
        ctypes.POINTER(_ProcessInformation),
    ]

    token = _HANDLE()
    restricted_token = _HANDLE()
    low_sid = _LPVOID()
    job = _HANDLE()
    child_stdin = _HANDLE()
    parent_stdin = _HANDLE()
    child_stdout = _HANDLE()
    parent_stdout = _HANDLE()
    process_info = _ProcessInformation()
    parent_fds: list[int] = []
    process_created = False
    sandbox = _metadata("windows-restricted-token-low-integrity", network_disabled=False)

    def fail(message: str) -> SandboxUnavailable:
        error = ctypes.get_last_error()
        return SandboxUnavailable(f"{message} (winerror={error})")

    try:
        security = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, True)
        if not kernel32.CreatePipe(ctypes.byref(child_stdin), ctypes.byref(parent_stdin), ctypes.byref(security), 0):
            raise fail("CreatePipe(stdin) failed")
        if not kernel32.CreatePipe(ctypes.byref(parent_stdout), ctypes.byref(child_stdout), ctypes.byref(security), 0):
            raise fail("CreatePipe(stdout) failed")
        if not kernel32.SetHandleInformation(parent_stdin, 1, 0):
            raise fail("SetHandleInformation(stdin) failed")
        if not kernel32.SetHandleInformation(parent_stdout, 1, 0):
            raise fail("SetHandleInformation(stdout) failed")

        desired = 0x0001 | 0x0002 | 0x0008 | 0x0080 | 0x0100
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), desired, ctypes.byref(token)):
            raise fail("OpenProcessToken failed")
        if not advapi32.CreateRestrictedToken(
            token, 0x1, 0, None, 0, None, 0, None, ctypes.byref(restricted_token)
        ):
            raise fail("CreateRestrictedToken failed")

        if not advapi32.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(low_sid)):
            raise fail("ConvertStringSidToSidW failed")
        sid_length = int(advapi32.GetLengthSid(low_sid))
        if not sid_length:
            raise fail("GetLengthSid failed")
        label = _TokenMandatoryLabel(_SidAndAttributes(low_sid, 0x20))
        if not advapi32.SetTokenInformation(
            restricted_token, 25, ctypes.byref(label), ctypes.sizeof(label)
        ):
            raise fail("SetTokenInformation(TokenIntegrityLevel) failed")

        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x2000
        job = kernel32.CreateJobObjectW(None, None)
        if not job or not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise fail("Job Object setup failed")

        child_env = _sandbox_environment(env, sandbox)
        environment_block = "\0".join(
            f"{key}={value}" for key, value in sorted(child_env.items(), key=lambda item: item[0].upper())
        ) + "\0\0"
        environment_buffer = ctypes.create_unicode_buffer(environment_block)
        command_line = subprocess.list2cmdline([str(python), "-u", str(worker)])
        command_buffer = ctypes.create_unicode_buffer(command_line)
        startup = _StartupInfo()
        startup.cb = ctypes.sizeof(_StartupInfo)
        startup.dwFlags = 0x100
        startup.hStdInput = child_stdin
        startup.hStdOutput = child_stdout
        startup.hStdError = child_stdout
        if not advapi32.CreateProcessAsUserW(
            restricted_token,
            None,
            command_buffer,
            None,
            None,
            True,
            0x00000400 | 0x08000000,
            ctypes.cast(environment_buffer, _LPVOID),
            str(cwd),
            ctypes.byref(startup),
            ctypes.byref(process_info),
        ):
            raise fail("CreateProcessAsUserW failed")
        process_created = True
        if not kernel32.AssignProcessToJobObject(job, process_info.hProcess):
            kernel32.TerminateProcess(process_info.hProcess, 1)
            raise fail("AssignProcessToJobObject failed")
        _close_handle(child_stdin)
        child_stdin = _HANDLE()
        _close_handle(child_stdout)
        child_stdout = _HANDLE()

        input_fd = msvcrt.open_osfhandle(int(parent_stdin.value), os.O_BINARY)
        output_fd = msvcrt.open_osfhandle(int(parent_stdout.value), os.O_BINARY)
        parent_stdin = _HANDLE()
        parent_stdout = _HANDLE()
        parent_fds.extend((input_fd, output_fd))
        output_holder: list[bytes] = []
        reader_error: list[Exception] = []

        def write_request() -> None:
            try:
                with os.fdopen(input_fd, "wb") as handle:
                    handle.write(input_text.encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                reader_error.append(exc)

        def read_response() -> None:
            try:
                with os.fdopen(output_fd, "rb") as handle:
                    output_holder.append(handle.read())
            except Exception as exc:  # noqa: BLE001
                reader_error.append(exc)

        writer = threading.Thread(target=write_request, daemon=True)
        reader = threading.Thread(target=read_response, daemon=True)
        writer.start()
        reader.start()
        wait_ms = max(1, int(timeout_seconds * 1000))
        wait_result = kernel32.WaitForSingleObject(process_info.hProcess, wait_ms)
        if wait_result == 0x102:
            kernel32.TerminateProcess(process_info.hProcess, 124)
            kernel32.WaitForSingleObject(process_info.hProcess, 5000)
            return SandboxResult(124, "", "sandbox worker timed out", sandbox)
        if wait_result != 0:
            raise fail("WaitForSingleObject failed")
        writer.join(2.0)
        reader.join(2.0)
        exit_code = _DWORD()
        if not kernel32.GetExitCodeProcess(process_info.hProcess, ctypes.byref(exit_code)):
            raise fail("GetExitCodeProcess failed")
        if reader_error and not output_holder:
            raise SandboxUnavailable(f"sandbox pipe failed: {reader_error[0]}")
        output = output_holder[0].decode("utf-8", errors="replace") if output_holder else ""
        return SandboxResult(int(exit_code.value), output, "", sandbox)
    finally:
        if process_created:
            _close_handle(process_info.hThread)
            _close_handle(process_info.hProcess)
        for handle in (child_stdin, parent_stdin, child_stdout, parent_stdout, token, restricted_token, job):
            _close_handle(handle)
        _free_local_sid(low_sid)
        for fd in parent_fds:
            try:
                os.close(fd)
            except OSError:
                pass


def run_sandboxed(
    python: Path,
    worker: Path,
    *,
    input_text: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> SandboxResult:
    """Run a trusted worker only inside a verified OS-level sandbox."""
    if os.name == "nt":
        return _run_windows(
            python, worker, input_text=input_text, cwd=cwd, env=env,
            timeout_seconds=timeout_seconds,
        )
    if sys.platform == "darwin":
        return _run_macos(
            python, worker, input_text=input_text, cwd=cwd, env=env,
            timeout_seconds=timeout_seconds,
        )
    return _run_bwrap(
        python, worker, input_text=input_text, cwd=cwd, env=env,
        timeout_seconds=timeout_seconds,
    )


__all__ = ["SandboxResult", "SandboxUnavailable", "run_sandboxed"]
