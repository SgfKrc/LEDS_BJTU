"""Test-only Qwen3 network node process.

The process deliberately exposes only the authenticated loopback control/data
routers. It is not a production service entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

import uvicorn
from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_pipeline_control import router as control_router  # noqa: E402
from qwen3_pipeline_data_plane import Qwen3ArtifactTransferRuntime, router as transfer_router  # noqa: E402
from qwen3_pipeline_network import Qwen3NetworkTransferCoordinator  # noqa: E402
from qwen3_pipeline_peer_auth import Qwen3PeerAuthMiddleware, Qwen3PeerRequestVerifier  # noqa: E402


class _RegistrationProjection:
    """Small test projection of TCP registration/disconnect events."""

    def __init__(self, peers: list[str]) -> None:
        self._lock = threading.RLock()
        self._epochs = {str(peer): 0 for peer in peers if str(peer)}

    def register(self, peer: str) -> int:
        with self._lock:
            epoch = int(self._epochs.get(str(peer), 0)) + 1
            self._epochs[str(peer)] = epoch
            return epoch

    def disconnect(self, peer: str) -> None:
        with self._lock:
            self._epochs.pop(str(peer), None)

    def authenticated(self, peer: str) -> bool:
        with self._lock:
            return str(peer) in self._epochs

    def authenticated_epoch(self, peer: str, epoch: int) -> bool:
        with self._lock:
            return self._epochs.get(str(peer)) == int(epoch)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._epochs)


class _SyntheticNetworkSidecarExecutor:
    """Test-only target executor used by the QW3.15 process-chain fixture."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.outputs: list[Path] = []

    def __call__(self, input_path: Path, request: dict) -> dict:
        incoming = input_path.read_bytes()
        segment_index = int(request["segment_index"])
        output = self.artifact_root / (
            f"qwen3-consume-{request['transfer_id']}-{request['phase']}-"
            f"{request['generation']}-{segment_index}.pt"
        )
        output.write_bytes(
            f"node-{segment_index}:{request['phase']}:".encode("ascii") + incoming,
        )
        self.outputs.append(output)
        return {
            "status": "executed",
            "gate_passed": True,
            "output_path": str(output),
            "execution": {
                "full_model_materialized": False,
                "segment_materialized": True,
            },
            "hidden_handoff": {
                "dtype": request["dtype"],
                "device": request["device"],
                "shape": [request["batch_size"], request["sequence_length"], 4],
            },
            "kv_contract": {
                "present": True,
                "shape": [request["batch_size"], request["sequence_length"]],
            },
        }

    def cleanup(self, _request: dict, _reason_code: str) -> None:
        for output in self.outputs:
            output.unlink(missing_ok=True)
        self.outputs.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--allowed-peer", action="append", default=[])
    parser.add_argument("--synthetic-sidecar", action="store_true")
    args = parser.parse_args()

    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=args.state_dir,
        cluster_secret=args.secret,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id=args.node_id,
        runtime=runtime,
    )
    ledger_path = Path(args.state_dir) / "qwen3-network-ledger.json"

    def load_ledger() -> dict:
        if not ledger_path.is_file():
            return {
                "local_node_id": args.node_id,
                "last_generation": -1,
                "active_contract": {},
                "transfers": {},
                "outputs": {},
            }
        return json.loads(ledger_path.read_text(encoding="utf-8"))

    def save_ledger(value: dict) -> dict:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ledger_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True), encoding="utf-8",
        )
        temporary.replace(ledger_path)
        return value

    coordinator.configure_persistent_ledger(load=load_ledger, save=save_ledger)
    registry = _RegistrationProjection(args.allowed_peer)
    verifier = Qwen3PeerRequestVerifier(
        args.secret,
        is_authenticated_peer=registry.authenticated,
        is_authenticated_peer_epoch=registry.authenticated_epoch,
        require_peer_epoch=False,
    )
    app = FastAPI()
    app.state.qwen3_network_sidecar_executor = (
        _SyntheticNetworkSidecarExecutor(runtime.receiver.root)
        if args.synthetic_sidecar else None
    )
    app.state.qwen3_artifact_transfer = runtime
    app.state.qwen3_network_transfer_coordinator = coordinator
    app.add_middleware(Qwen3PeerAuthMiddleware, verifier=verifier)
    app.include_router(transfer_router)
    app.include_router(control_router)

    @app.post("/__fixture/registration")
    async def project_registration(payload: dict):
        peer = str(payload.get("peer", ""))
        action = str(payload.get("action", "register"))
        if action == "disconnect":
            registry.disconnect(peer)
        else:
            registry.register(peer)
        return {"peers": registry.snapshot()}
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="error",
        access_log=False,
    )


if __name__ == "__main__":
    main()
