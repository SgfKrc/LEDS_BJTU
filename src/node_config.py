"""Local node bootstrap configuration.

The regular .env file is intentionally not bundled with installers because it
contains secrets.  This module provides a small non-source-controlled runtime
configuration file used after a trusted first-connect bootstrap.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_node_config_path() -> Path:
    override = os.environ.get("QLH_NODE_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
            return base / "QLH-Edge-Inference" / "node_config.json"
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "qlh" / "node_config.json"
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "qlh" / "node_config.json"
    return get_app_root() / "node_config.json"


def load_node_config() -> dict[str, Any]:
    path = get_node_config_path()
    candidates = [path]
    legacy_path = get_app_root() / "node_config.json"
    if legacy_path != path:
        candidates.append(legacy_path)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if candidate != path:
                try:
                    write_node_config(data)
                except Exception:
                    pass
            return data
        except Exception:
            continue
    return {}


def write_node_config(data: dict[str, Any]) -> Path:
    path = get_node_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)
    return path


def _set_env_value(name: str, value: Any, *, overwrite: bool = False) -> None:
    if value is None:
        return
    value_str = str(value).strip()
    if not value_str:
        return
    if overwrite:
        os.environ[name] = value_str
    else:
        os.environ.setdefault(name, value_str)


def apply_node_config_to_env(
    config_data: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    data = config_data if config_data is not None else load_node_config()
    if not data:
        return {}

    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    node = data.get("node") if isinstance(data.get("node"), dict) else {}

    _set_env_value("QLH_NODE_ROLE", node.get("role"), overwrite=overwrite)
    _set_env_value("QLH_NODE_ID", node.get("node_id"), overwrite=overwrite)
    _set_env_value("QLH_NODE_TYPE", node.get("node_type"), overwrite=overwrite)
    _set_env_value("QLH_CLUSTER_SECRET", cluster.get("cluster_secret"), overwrite=overwrite)
    _set_env_value(
        "QLH_MASTER_HOST",
        cluster.get("master_tcp_host") or cluster.get("master_host"),
        overwrite=overwrite,
    )
    _set_env_value(
        "QLH_MASTER_PORT",
        cluster.get("master_tcp_port") or cluster.get("master_port"),
        overwrite=overwrite,
    )
    _set_env_value(
        "QLH_CLIENT_MASTER_HOST",
        cluster.get("master_tcp_host") or cluster.get("master_host"),
        overwrite=overwrite,
    )
    _set_env_value(
        "QLH_CLIENT_MASTER_PORT",
        cluster.get("master_tcp_port") or cluster.get("master_port"),
        overwrite=overwrite,
    )
    _set_env_value("QLH_MASTER_API_HOST", cluster.get("master_api_host"), overwrite=overwrite)
    _set_env_value("QLH_MASTER_API_PORT", cluster.get("master_api_port"), overwrite=overwrite)
    _set_env_value(
        "QLH_API_PORT",
        cluster.get("master_api_port") if node.get("role") == "master" else None,
        overwrite=overwrite,
    )
    return data


def build_bootstrap_config(response: dict[str, Any]) -> dict[str, Any]:
    cluster = response.get("cluster") if isinstance(response.get("cluster"), dict) else {}
    node = response.get("node") if isinstance(response.get("node"), dict) else {}
    return {
        "bootstrapped": True,
        "cluster": {
            "cluster_id": cluster.get("cluster_id", "qlh-default"),
            "master_api_host": cluster.get("master_api_host", ""),
            "master_api_port": int(cluster.get("master_api_port", 8000) or 8000),
            "master_tcp_host": cluster.get("master_tcp_host", ""),
            "master_tcp_port": int(cluster.get("master_tcp_port", 8888) or 8888),
            "cluster_secret": cluster.get("cluster_secret", ""),
        },
        "node": {
            "node_id": node.get("node_id", ""),
            "role": node.get("role", "client"),
            "node_type": node.get("node_type", "pc"),
            "pipeline_worker": bool(node.get("pipeline_worker", True)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def persist_bootstrap_response(response: dict[str, Any]) -> Path:
    config_data = build_bootstrap_config(response)
    path = write_node_config(config_data)
    apply_node_config_to_env(config_data, overwrite=True)
    return path


def ensure_local_cluster_secret() -> str:
    """Return an existing cluster secret or create one in node_config.json."""
    current = os.environ.get("QLH_CLUSTER_SECRET", "").strip()
    if current:
        return current

    data = load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    existing = str(cluster.get("cluster_secret", "")).strip()
    if existing:
        os.environ.setdefault("QLH_CLUSTER_SECRET", existing)
        return existing

    secret = secrets.token_urlsafe(32)
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    explicit_role = os.environ.get("QLH_NODE_ROLE", "").strip()
    role_confirmed = bool(node.get("role_confirmed", False) or data.get("bootstrapped", False))
    if not data and explicit_role:
        role_confirmed = True
    data.update({
        "bootstrapped": bool(data.get("bootstrapped", False)),
        "cluster": {
            **cluster,
            "cluster_id": cluster.get("cluster_id", "qlh-default"),
            "cluster_secret": secret,
        },
        "node": {
            "role": node.get("role", os.environ.get("QLH_NODE_ROLE", "master")),
            "role_confirmed": role_confirmed,
            "node_id": node.get("node_id", os.environ.get("QLH_NODE_ID", "master")),
            "node_type": node.get("node_type", os.environ.get("QLH_NODE_TYPE", "pc")),
            "pipeline_worker": bool(node.get("pipeline_worker", True)),
        },
    })
    write_node_config(data)
    os.environ.setdefault("QLH_CLUSTER_SECRET", secret)
    return secret


def _sync_loaded_module_attr(module_name: str, attr: str, value: Any) -> None:
    module = sys.modules.get(module_name)
    if module is not None:
        try:
            setattr(module, attr, value)
        except Exception:
            pass


def apply_runtime_config(response: dict[str, Any]) -> None:
    """Update already-imported runtime modules after bootstrap."""
    cluster = response.get("cluster") if isinstance(response.get("cluster"), dict) else {}
    node = response.get("node") if isinstance(response.get("node"), dict) else {}
    apply_node_config_to_env(build_bootstrap_config(response), overwrite=True)
    try:
        import config as cfg

        if cluster.get("cluster_secret"):
            cfg.CLUSTER_SECRET = str(cluster["cluster_secret"])
        if cluster.get("master_tcp_host"):
            cfg.CLIENT_MASTER_HOST = str(cluster["master_tcp_host"])
        if cluster.get("master_tcp_port"):
            cfg.CLIENT_MASTER_PORT = int(cluster["master_tcp_port"])
        if node.get("node_id"):
            cfg.NODE_ID = str(node["node_id"])
            _sync_loaded_module_attr("scheduler", "NODE_ID", cfg.NODE_ID)
        if node.get("role"):
            cfg.NODE_ROLE = str(node["role"])
            _sync_loaded_module_attr("scheduler", "NODE_ROLE", cfg.NODE_ROLE)
    except Exception:
        pass
