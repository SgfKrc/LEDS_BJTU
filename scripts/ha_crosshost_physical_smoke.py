"""Run the HA-CROSSHOST-01 physical Surface/y700 smoke gate.

The Surface worker is streamed over SSH stdin and never written to the remote
disk.  The control frames travel through an SSH reverse TCP tunnel.  The y700
probe uses Android wireless debugging through an explicitly supplied or
currently online ``adb`` serial; it never assumes a fixed Android port.  This
is a physical process/network check, not a production availability claim.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cluster_transport import TransportEnvelope  # noqa: E402


REMOTE_WORKER = r'''
import argparse
import base64
import hashlib
import json
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--port", required=True, type=int)
args = parser.parse_args()
sys.path.insert(0, args.root + r"\src")
from cluster_transport import TransportContractError, TransportEnvelope

def send(out, value):
    out.sendall((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))

def read_line(inp):
    value = inp.readline()
    if not value:
        return None
    return json.loads(value.decode("utf-8"))

generation = 0
attempt_id = ""
received = {}
with socket.create_connection(("127.0.0.1", args.port), timeout=15) as sock:
    sock.settimeout(15)
    inp = sock.makefile("rb")
    send(sock, {"ok": True, "event": "ready"})
    while True:
        command = read_line(inp)
        if command is None:
            break
        try:
            operation = str(command.get("op", ""))
            if operation == "stop":
                send(sock, {"ok": True, "event": "stopped"})
                break
            if operation == "set_generation":
                generation = int(command["generation"])
                attempt_id = str(command["attempt_id"])
                received = {}
                send(sock, {"ok": True, "event": "generation_set", "generation": generation})
                continue
            if operation != "receive":
                raise TransportContractError("worker_operation_invalid", "unsupported operation")
            envelope = TransportEnvelope.decode(json.dumps(command["envelope"], sort_keys=True))
            payload = base64.b64decode(command["payload_b64"].encode("ascii"), validate=True)
            now_ms = int(command["now_ms"])
            if envelope.connection_generation != generation:
                raise TransportContractError("generation_stale", "old generation")
            if envelope.attempt_id != attempt_id:
                raise TransportContractError("attempt_fenced", "old attempt")
            if envelope.is_expired(now_ms=now_ms):
                raise TransportContractError("deadline_exceeded", "expired envelope")
            if envelope.payload_size != len(payload) or envelope.payload_digest != hashlib.sha256(payload).hexdigest():
                raise TransportContractError("payload_mismatch", "payload does not match envelope")
            previous = received.get(envelope.channel, -1)
            if envelope.sequence <= previous:
                raise TransportContractError("sequence_duplicate", "duplicate sequence")
            if envelope.sequence != previous + 1:
                raise TransportContractError("sequence_out_of_order", "non-contiguous sequence")
            received[envelope.channel] = envelope.sequence
            send(sock, {"ok": True, "event": "accepted", "generation": generation, "sequence": envelope.sequence})
        except TransportContractError as exc:
            send(sock, {"ok": False, "code": exc.code})
        except Exception as exc:
            send(sock, {"ok": False, "code": getattr(exc, "code", "worker_error")})
'''


class PhysicalSmokeError(RuntimeError):
    pass


def _send_line(sock: socket.socket, value: dict[str, Any]) -> None:
    sock.sendall((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _recv_line(sock: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise PhysicalSmokeError("physical worker disconnected")
        if chunk == b"\n":
            break
        data.extend(chunk)
        if len(data) > 64 * 1024:
            raise PhysicalSmokeError("physical worker response is too large")
    value = json.loads(bytes(data).decode("utf-8"))
    if not isinstance(value, dict):
        raise PhysicalSmokeError("physical worker response is not an object")
    return value


def _request(sock: socket.socket, value: dict[str, Any]) -> dict[str, Any]:
    _send_line(sock, value)
    return _recv_line(sock)


def _expect_ok(sock: socket.socket, value: dict[str, Any]) -> dict[str, Any]:
    response = _request(sock, value)
    if not response.get("ok"):
        raise PhysicalSmokeError(str(response.get("code", "worker_error")))
    return response


def _ssh_process(target: str, remote_root: str, remote_port: int, local_port: int) -> subprocess.Popen[bytes]:
    command = f'cd /d "{remote_root}" && python - --root "{remote_root}" --port {remote_port}'
    process = subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            f"127.0.0.1:{remote_port}:127.0.0.1:{local_port}",
            target,
            command,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(REMOTE_WORKER.encode("utf-8"))
    process.stdin.close()
    return process


def _stop_ssh(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)


def _run_surface(target: str, remote_root: str, *, long_steps: int = 1) -> dict[str, Any]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(20)
    local_port = int(listener.getsockname()[1])
    remote_port = 43000 + (local_port % 1000)
    events: list[dict[str, Any]] = []
    payload = b"physical-ha-control-metadata"
    now_ms = int(time.time() * 1000)
    first_process = _ssh_process(target, remote_root, remote_port, local_port)
    try:
        connection, _ = listener.accept()
        connection.settimeout(15)
        with connection:
            ready = _recv_line(connection)
            if not ready.get("ok") or ready.get("event") != "ready":
                raise PhysicalSmokeError("surface worker did not become ready")
            _expect_ok(connection, {"op": "set_generation", "generation": 1, "attempt_id": "physical-1"})
            first = TransportEnvelope.from_payload(
                payload,
                request_id="physical-accepted",
                connection_generation=1,
                attempt_id="physical-1",
                channel="control",
                sequence=0,
                deadline_ms=now_ms + 20_000,
            )
            accepted = _expect_ok(connection, {
                "op": "receive",
                "envelope": first.to_dict(),
                "payload_b64": base64.b64encode(payload).decode("ascii"),
                "now_ms": now_ms,
            })
            events.append({"event": "surface_control_frame_accepted", "generation": accepted["generation"]})
            requested_frames = max(1, int(long_steps))
            for sequence in range(1, requested_frames):
                long_frame = TransportEnvelope.from_payload(
                    payload,
                    request_id=f"physical-long-{sequence}",
                    connection_generation=1,
                    attempt_id="physical-1",
                    channel="control",
                    sequence=sequence,
                    deadline_ms=int(time.time() * 1000) + max(20_000, requested_frames * 100),
                )
                _expect_ok(connection, {
                    "op": "receive",
                    "envelope": long_frame.to_dict(),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "now_ms": int(time.time() * 1000),
                })
            long_running = {
                "requested_frames": requested_frames,
                "accepted_frames": requested_frames,
                "transport": "ssh_reverse_tcp",
            }
            old = TransportEnvelope.from_payload(
                payload,
                request_id="physical-stale",
                connection_generation=1,
                attempt_id="physical-1",
                channel="control",
                sequence=0,
                deadline_ms=now_ms + 20_000,
            )
        failure_started = time.perf_counter()
        _stop_ssh(first_process)
        events.append({"event": "surface_worker_stopped", "reason": "simulated_crash"})

        second_process = _ssh_process(target, remote_root, remote_port, local_port)
        try:
            connection, _ = listener.accept()
            connection.settimeout(15)
            with connection:
                ready = _recv_line(connection)
                if not ready.get("ok"):
                    raise PhysicalSmokeError("surface replacement worker did not become ready")
                _expect_ok(connection, {"op": "set_generation", "generation": 2, "attempt_id": "physical-2"})
                stale = _request(connection, {
                    "op": "receive",
                    "envelope": old.to_dict(),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "now_ms": int(time.time() * 1000),
                })
                if stale.get("ok") or stale.get("code") != "generation_stale":
                    raise PhysicalSmokeError("Surface accepted an old generation")
                events.append({"event": "surface_old_generation_rejected", "code": "generation_stale"})
                second = TransportEnvelope.from_payload(
                    payload,
                    request_id="physical-reconnected",
                    connection_generation=2,
                    attempt_id="physical-2",
                    channel="control",
                    sequence=0,
                    deadline_ms=int(time.time() * 1000) + 20_000,
                )
                recovered = _expect_ok(connection, {
                    "op": "receive",
                    "envelope": second.to_dict(),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "now_ms": int(time.time() * 1000),
                })
                events.append({"event": "surface_control_frame_accepted_after_reconnect", "generation": recovered["generation"]})
        finally:
            _stop_ssh(second_process)
        return {
            "status": "passed",
            "target": "surface",
            "physical_nodes": True,
            "transport": "ssh_reverse_tcp",
            "events": events,
            "rto_ms": max(0, int((time.perf_counter() - failure_started) * 1000)),
            "rpo": {"last_durable_sequence": 0, "lost_events": 0, "scope": "transport_only"},
            "long_running": long_running,
        }
    finally:
        _stop_ssh(first_process)
        listener.close()


def _find_adb(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    discovered = shutil.which("adb")
    if discovered:
        return discovered
    sdk_candidates = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")]
    if os.name == "nt":
        sdk_candidates.extend([
            str(Path.home() / "AppData" / "Local" / "Android" / "Sdk"),
            str(Path.home() / "Android" / "Sdk"),
        ])
    for sdk in sdk_candidates:
        if sdk:
            candidate = Path(sdk) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if candidate.is_file():
                return str(candidate)
    return None


def _discover_y700_serial(adb: str, host: str) -> str | None:
    completed = subprocess.run(
        [adb, "devices", "-l"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    online: list[str] = []
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "device":
            online.append(fields[0])
    matching = [item for item in online if item == host or item.startswith(f"{host}:")]
    if len(matching) == 1:
        return matching[0]
    if not matching and len(online) == 1:
        return online[0]
    return None


def _run_y700(
    serial: str | None = None,
    *,
    adb: str | None = None,
    host: str = "100.99.211.13",
) -> dict[str, Any]:
    adb_path = _find_adb(adb)
    if not adb_path:
        return {
            "status": "failed",
            "target": "y700",
            "physical_nodes": True,
            "transport": "adb_wireless_debugging",
            "error_code": "adb_not_found",
            "model_gate": "blocked_no_gguf",
        }
    serial = serial or _discover_y700_serial(adb_path, host)
    if not serial:
        return {
            "status": "failed",
            "target": "y700",
            "physical_nodes": True,
            "transport": "adb_wireless_debugging",
            "error_code": "dynamic_adb_serial_required",
            "model_gate": "blocked_no_gguf",
            "hint": f"run adb connect {host}:<dynamic_port>, then pass --y700-serial",
        }
    command = (
        "printf 'model=%s\\n' \"$(getprop ro.product.model)\"; "
        "printf 'sdk=%s\\n' \"$(getprop ro.build.version.sdk)\"; "
        "printf 'abi=%s\\n' \"$(getprop ro.product.cpu.abilist)\"; "
        "printf 'nproc=%s\\n' \"$(nproc)\"; "
        "printf 'available_mem_kb=%s\\n' \"$(awk '/MemAvailable/ {print $2}' /proc/meminfo)\"; "
        "printf 'model_root=/sdcard/Download/QLH/models\\n'; "
        "if test -d /sdcard/Download/QLH/models; then printf 'model_root_exists=true\\n'; "
        "else printf 'model_root_exists=false\\n'; fi; "
        "printf 'gguf_count=%s\\n' \"$(find /sdcard/Download/QLH/models -maxdepth 1 -type f -name '*.gguf' 2>/dev/null | wc -l | tr -d ' ')\""
    )
    completed = subprocess.run(
        [adb_path, "-s", serial, "shell", command],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    observations: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            observations[key.strip()] = value.strip()
    online = completed.returncode == 0 and "arm64-v8a" in {
        item.strip() for item in observations.get("abi", "").split(",")
    }
    try:
        gguf_count = int(observations.get("gguf_count", "0"))
    except ValueError:
        gguf_count = 0
    return {
        "status": "passed" if online else "failed",
        "target": "y700",
        "physical_nodes": True,
        "transport": "adb_wireless_debugging",
        "serial": serial,
        "observations": observations,
        "model_gate": (
            "ready_for_arm64_model_smoke"
            if gguf_count > 0
            else "blocked_no_gguf"
        ),
        "error_code": "adb_failed" if completed.returncode else "",
        "stderr_tail": completed.stderr[-500:] if completed.returncode else "",
    }


def _failed_result(target: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "target": target,
        "physical_nodes": True,
        "error_type": type(error).__name__,
        "error": str(error)[:500],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-target", default="surface@100.100.52.106")
    parser.add_argument("--surface-root", default=r"C:\Users\surface\Documents\LEDS_BJTU")
    parser.add_argument("--y700-serial", help="adb wireless serial, for example IP:<dynamic_port>")
    parser.add_argument("--y700-host", default="100.99.211.13")
    parser.add_argument("--adb", help="path to adb; defaults to PATH or the local Android SDK")
    parser.add_argument("--long-steps", type=int, default=1,
                        help="Surface control frames before restart (default: 1)")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    long_steps = max(1, min(int(args.long_steps), 10_000))
    try:
        surface = _run_surface(args.surface_target, args.surface_root, long_steps=long_steps)
    except Exception as exc:
        surface = _failed_result("surface", exc)
    try:
        y700 = _run_y700(args.y700_serial, adb=args.adb, host=args.y700_host)
    except Exception as exc:
        y700 = _failed_result("y700", exc)
    report = {
        "schema_version": "qlh.cluster.crosshost.physical.v1",
        "scenario": "physical_surface_y700_smoke",
        "surface": surface,
        "y700": y700,
        "long_steps": long_steps,
        "production_availability_claim": False,
    }
    rendered = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        partial = args.evidence.with_name(f".{args.evidence.name}.part")
        partial.write_text(rendered, encoding="utf-8")
        partial.replace(args.evidence)
    print(rendered, end="")
    return 0 if report["surface"]["status"] == "passed" and report["y700"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
