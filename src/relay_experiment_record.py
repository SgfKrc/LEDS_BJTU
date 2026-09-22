"""P1：跨框架接力实验记录（relay experiment record v1）的统一构建与校验。

为什么需要（`docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md` §4 P1）：
D→L 的历史数字来自裸 runner、主仓引擎和 L→L 对照三套口径，混在一张表里就会反复产生
「拿探针数字当端到端基线」的错误。本模块把记录的**字段集**与**kind/path 一致性**固定下来：

* `kind` 只有三类，报告必须按它分表：`mainrepo_end_to_end` / `raw_binding_probe` /
  `capacity_only`；
* `path` 记录实际链路（`d2l_mainrepo` / `d2l_raw_binding` / `l2l_llama` / `capacity_scan`）；
* 两个字段与 `engines.*_iface` 必须自洽 —— 不一致会被 `validate_record()` 拒绝（fail-closed），
  这就是「防止再次混表」的机器判据。

本模块是**纯函数 + 惰性依赖**：不 import torch / llama_cpp（只有 `default_env()` 会尝试探测
并允许失败），因此可以在任何环境里被测试与复用。schema 见
`schemas/relay-experiment-record.schema.json`。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.relay_contract import RELAY_ACCEPTANCE

SCHEMA_VERSION = "qlh.relay_experiment.v1"
SCHEMA_RELPATH = Path("schemas") / "relay-experiment-record.schema.json"

#: ★ 三类互斥（P1 票面）。报告表格按此分组，禁止跨类比较数字。
KIND_MAINREPO_END_TO_END = "mainrepo_end_to_end"
KIND_RAW_BINDING_PROBE = "raw_binding_probe"
KIND_CAPACITY_ONLY = "capacity_only"
EXPERIMENT_KINDS = (KIND_MAINREPO_END_TO_END, KIND_RAW_BINDING_PROBE, KIND_CAPACITY_ONLY)

PATH_D2L_MAINREPO = "d2l_mainrepo"
PATH_D2L_RAW = "d2l_raw_binding"
PATH_L2L = "l2l_llama"
PATH_L2L_KEEP_HEAD = "l2l_keep_head"
PATH_D2L2L_KEEP_HEAD = "d2l2l_keep_head"
PATH_D2L2L_KEEP_HEAD_NET = "d2l2l_keep_head_net"
PATH_CAPACITY = "capacity_scan"
EXPERIMENT_PATHS = (PATH_D2L_MAINREPO, PATH_D2L_RAW, PATH_L2L, PATH_L2L_KEEP_HEAD,
                    PATH_D2L2L_KEEP_HEAD, PATH_D2L2L_KEEP_HEAD_NET, PATH_CAPACITY)

#: 引擎接口标识串（必须与实际调用的入口一一对应，不允许同义改写）。
IFACE_MODEL_MODULE_UPSTREAM = "model_module.forward_layers"
IFACE_LLAMA_UPSTREAM = "llama_engine.forward_layers_to_hidden"
IFACE_KEEP_HEAD_UPSTREAM = "llama_keep_head.KeepHeadUpstream.forward_tokens_to_hidden"
IFACE_LLAMA_ENGINE_DOWNSTREAM = "llama_engine.forward_layers_from_hidden"
IFACE_RAW_LLAMA_DOWNSTREAM = "llama_cpp.llama_decode"
#: 跨机中间段：hidden 经 Relay TCP 交给远端段（远端用同一 keep-head 语义）。
IFACE_RELAY_MIDDLE = "relay_transport.RelayTcpClient.request_hidden"
#: 容量专项只加载、不生成 ⇒ 接口是两侧的加载入口。
IFACE_MODEL_MODULE_LOADER = "model_module.load_layer_range"
IFACE_LLAMA_MODEL_LOADER = "llama_cpp.llama_model_load_from_file"

#: (上游接口, 下游接口, 中间段接口) -> (kind, path)。未登记的组合一律拒绝：新链路必须显式登记。
#: 中间段为空串表示两段链路。三段必须在这里登记，否则会被误判成两段 D→L。
_PATH_BY_IFACES: dict[tuple[str, str, str], tuple[str, str]] = {
    (IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, ""): (
        KIND_MAINREPO_END_TO_END, PATH_D2L_MAINREPO),
    (IFACE_MODEL_MODULE_UPSTREAM, IFACE_RAW_LLAMA_DOWNSTREAM, ""): (
        KIND_RAW_BINDING_PROBE, PATH_D2L_RAW),
    (IFACE_LLAMA_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, ""): (
        KIND_RAW_BINDING_PROBE, PATH_L2L),
    (IFACE_LLAMA_UPSTREAM, IFACE_RAW_LLAMA_DOWNSTREAM, ""): (
        KIND_RAW_BINDING_PROBE, PATH_L2L),
    # 补丁版 keep-head 通道（P2 路线 C）：llama 能当上游/中间段 ⇒ 真正的 L→L 与三段。
    (IFACE_KEEP_HEAD_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, ""): (
        KIND_RAW_BINDING_PROBE, PATH_L2L_KEEP_HEAD),
    (IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, IFACE_KEEP_HEAD_UPSTREAM): (
        KIND_MAINREPO_END_TO_END, PATH_D2L2L_KEEP_HEAD),
    # 跨机三段：中间段在远端，经 Relay TCP（loopback + SSH 隧道）往返 hidden。
    (IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM, IFACE_RELAY_MIDDLE): (
        KIND_MAINREPO_END_TO_END, PATH_D2L2L_KEEP_HEAD_NET),
    (IFACE_MODEL_MODULE_LOADER, IFACE_LLAMA_MODEL_LOADER, ""): (
        KIND_CAPACITY_ONLY, PATH_CAPACITY),
}


def _iface_key(upstream_iface: str, downstream_iface: str,
               middle_iface: str | None = None) -> tuple[str, str, str]:
    return (str(upstream_iface), str(downstream_iface), str(middle_iface or ""))


def classify_path(upstream_iface: str, downstream_iface: str,
                  middle_iface: str | None = None) -> tuple[str, str]:
    """由引擎接口反推 `(kind, path)`。未登记组合抛 `ValueError`（而不是猜一个）。"""
    key = _iface_key(upstream_iface, downstream_iface, middle_iface)
    try:
        return _PATH_BY_IFACES[key]
    except KeyError:
        known = ", ".join(f"{u} + {m or '-'} + {d}" for u, d, m in _PATH_BY_IFACES)
        raise ValueError(
            f"未登记的引擎接口组合 {key}；已登记：{known}。"
            "新链路必须在 src/relay_experiment_record.py 显式登记后才允许写记录。") from None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_schema(root: Path | None = None) -> dict[str, Any]:
    path = (root or repo_root()) / SCHEMA_RELPATH
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_record(record: Mapping[str, Any], root: Path | None = None) -> None:
    """按 schema 校验记录；不合法抛 `ValueError`，缺 `jsonschema` 抛 `RuntimeError`。"""
    _validate_engine_identity(record)
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - 依赖缺失时 fail-loud，不静默放行
        raise RuntimeError("校验接力实验记录需要 jsonschema（requirements-test.txt 已声明）") from exc

    validator_cls = getattr(jsonschema, "Draft202012Validator", None) or jsonschema.Draft7Validator
    validator = validator_cls(load_schema(root))
    errors = sorted(validator.iter_errors(dict(record)), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        location = "/".join(str(p) for p in first.path) or "<root>"
        detail = "; ".join(f"{location}: {e.message}" for e in errors[:5])
        raise ValueError(f"接力实验记录不合法（{len(errors)} 处）：{detail}")


# The JSON schema checks the shape. Keep the interface registry as the
# authoritative semantic check so callers cannot write a relabeled record by
# bypassing build_record().
def _validate_engine_identity(record: Mapping[str, Any]) -> None:
    engines = record.get("engines")
    if not isinstance(engines, Mapping):
        raise ValueError("record.engines must be an object")
    upstream_iface = engines.get("upstream_iface")
    downstream_iface = engines.get("downstream_iface")
    middle_iface = engines.get("middle_iface")
    if not isinstance(upstream_iface, str) or not isinstance(downstream_iface, str):
        raise ValueError("record engines must contain string upstream_iface/downstream_iface")
    if middle_iface is not None and not isinstance(middle_iface, str):
        raise ValueError("record engines.middle_iface must be a string or null")
    inferred_kind, inferred_path = classify_path(upstream_iface, downstream_iface, middle_iface)
    if record.get("kind") != inferred_kind or record.get("path") != inferred_path:
        raise ValueError(
            "\u4e0d\u5408\u6cd5 record kind/path does not match engines: "
            f"declared {record.get('kind')!r}/{record.get('path')!r}, "
            f"expected {inferred_kind!r}/{inferred_path!r}"
        )


def git_head(root: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root or repo_root()),
                              capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001 - 无 git 时如实返回 None
        return None
    head = proc.stdout.strip()
    return head or None


def dependency_lock(root: Path | None = None) -> dict[str, Any] | None:
    """环境锁指纹：`requirements-lock/main.lock.txt` 的 sha256（缺失则 None）。"""
    lock = (root or repo_root()) / "requirements-lock" / "main.lock.txt"
    if not lock.is_file():
        return None
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    return {"file": str(lock.relative_to(root or repo_root())), "sha256": digest}


def default_env(root: Path | None = None) -> dict[str, Any]:
    """环境画像：OS / Python / torch / llama-cpp-python / CUDA / 环境锁。探测失败如实置 None。"""
    env: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": None,
        "llama_cpp_python": None,
        "cuda": None,
        "dependency_lock": dependency_lock(root),
    }
    try:
        import torch  # noqa: PLC0415 - 探测用惰性导入

        env["torch"] = str(torch.__version__)
        env["cuda"] = str(torch.version.cuda) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    try:
        import llama_cpp  # noqa: PLC0415

        env["llama_cpp_python"] = str(getattr(llama_cpp, "__version__", "")) or None
    except Exception:  # noqa: BLE001
        pass
    return env


def make_experiment_id(path: str, model_id: str | None, upstream_layers: int | None,
                       prefill_tokens: int | None, gen_tokens: int | None,
                       batch: int | None) -> str:
    parts = [f"relay-{path}"]
    if model_id:
        parts.append("".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in model_id))
    if upstream_layers is not None:
        parts.append(f"k{int(upstream_layers)}")
    if prefill_tokens is not None:
        parts.append(f"p{int(prefill_tokens)}")
    if gen_tokens is not None:
        parts.append(f"g{int(gen_tokens)}")
    if batch is not None:
        parts.append(f"b{int(batch)}")
    return "-".join(parts)


def build_record(
    *,
    upstream_iface: str,
    downstream_iface: str,
    middle_iface: str | None = None,
    models: Mapping[str, Any],
    layer_layout: Mapping[str, Any],
    handoff: Mapping[str, Any],
    load: Mapping[str, Any],
    verdict: Mapping[str, Any],
    metrics: Mapping[str, Any],
    kind: str | None = None,
    path: str | None = None,
    env: Mapping[str, Any] | None = None,
    device_profile: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    commit: str | None = None,
    timestamp: str | None = None,
    experiment_id: str | None = None,
    criterion: str = RELAY_ACCEPTANCE,
    root: Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """组装记录。`kind`/`path` 缺省由接口反推；显式传入时**必须与接口自洽**。

    `validate=True`（默认）会在返回前跑 schema 校验 —— 保证「强制写入合法记录」，
    而不是等报告阶段才发现混表。
    """
    inferred_kind, inferred_path = classify_path(upstream_iface, downstream_iface, middle_iface)
    if kind is not None and kind != inferred_kind:
        raise ValueError(
            f"kind={kind!r} 与引擎接口不一致（按 {upstream_iface} + {downstream_iface} 应为 "
            f"{inferred_kind!r}）—— 禁止把探针/容量数字标成端到端基线")
    if path is not None and path != inferred_path:
        raise ValueError(
            f"path={path!r} 与引擎接口不一致（应为 {inferred_path!r}）")

    model_id = None
    upstream_models = models.get("upstream") or {}
    if isinstance(upstream_models, Mapping):
        model_id = upstream_models.get("id")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id or make_experiment_id(
            inferred_path, model_id,
            layer_layout.get("upstream_layers"),
            load.get("prefill_tokens"), load.get("gen_tokens"), load.get("batch")),
        "kind": kind or inferred_kind,
        "path": path or inferred_path,
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit if commit is not None else (git_head(root) or ""),
        "env": dict(env) if env is not None else default_env(root),
        "device_profile": dict(device_profile or {}),
        "models": {k: dict(v) for k, v in models.items()},
        "layer_layout": dict(layer_layout),
        "engines": {"upstream_iface": upstream_iface, "downstream_iface": downstream_iface,
                    "middle_iface": middle_iface},
        "handoff": dict(handoff),
        "load": dict(load),
        "criterion": criterion,
        "verdict": dict(verdict),
        "metrics": dict(metrics),
        "evidence": dict(evidence) if evidence is not None else None,
        "artifacts": dict(artifacts or {}),
        "record_origin": {"kind": "raw_measurement", "rounds": 1},
    }
    if validate:
        validate_record(record, root)
    return record


def write_record(record: Mapping[str, Any], path: str | Path,
                 root: Path | None = None) -> Path:
    """校验后落盘（UTF-8, indent=1，与本仓既有实验 JSON 一致）。"""
    validate_record(record, root)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    artifacts = dict(payload.get("artifacts") or {})
    artifacts.setdefault("record_path", str(target))
    payload["artifacts"] = artifacts
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return target
