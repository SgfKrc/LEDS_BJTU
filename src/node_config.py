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

from network_address import canonical_host


def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_node_config_path() -> Path:
    override = os.environ.get("QLH_NODE_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    # Keep identity outside the checkout in both frozen and source mode.  A
    # source checkout is disposable (and patch delivery may hard-reset/clean
    # it), while the node identity and cluster secret belong to the user.
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "QLH-Edge-Inference" / "node_config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "qlh" / "node_config.json"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "qlh" / "node_config.json"


def resolve_initial_node_role() -> str:
    """Resolve startup role without promoting an unconfigured source clone.

    Packaged installs retain the historical master-first behavior for the
    first-run setup UI.  A source checkout with no persisted configuration is
    treated as a worker until the user explicitly selects ``master``.
    """
    explicit = os.environ.get("QLH_NODE_ROLE", "").strip().lower()
    if explicit:
        if explicit == "master":
            return "master"
        if explicit in {"slave", "worker", "client"}:
            return "client"
        # An unrecognised role must not promote a source checkout.
        return "master" if getattr(sys, "frozen", False) else "client"
    data = load_node_config()
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    configured = str(node.get("role", "")).strip().lower()
    if configured:
        if configured == "master":
            return "master"
        if configured in {"slave", "worker", "client"}:
            return "client"
        return "master" if getattr(sys, "frozen", False) else "client"
    return "master" if getattr(sys, "frozen", False) else "client"


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


def _normalize_master_endpoint(host: str | None, port: int | str | None) -> dict[str, Any] | None:
    """Return a canonical, safe-to-persist master TCP endpoint."""
    normalized_host = canonical_host(host)
    if not normalized_host:
        return None
    try:
        normalized_port = int(port or 0)
    except (TypeError, ValueError):
        return None
    if not 1 <= normalized_port <= 65535:
        return None

    address_family = "hostname"
    try:
        import ipaddress

        address = ipaddress.ip_address(normalized_host.split("%", 1)[0])
        address_family = "ipv6" if address.version == 6 else "ipv4"
    except ValueError:
        pass
    return {
        "host": normalized_host,
        "port": normalized_port,
        "address_family": address_family,
    }


def get_preferred_master_endpoint(
    config_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Read the user-selected endpoint without trusting malformed old state."""
    data = config_data if config_data is not None else load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    preferred = (
        cluster.get("preferred_master_endpoint")
        if isinstance(cluster.get("preferred_master_endpoint"), dict)
        else {}
    )
    return _normalize_master_endpoint(preferred.get("host"), preferred.get("port"))


def get_bootstrap_master_endpoint(
    config_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Read the bootstrap-provided endpoint kept as a fallback to a preference."""
    data = config_data if config_data is not None else load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    return _normalize_master_endpoint(
        cluster.get("master_tcp_host") or cluster.get("master_host"),
        cluster.get("master_tcp_port") or cluster.get("master_port"),
    )


def persist_preferred_master_endpoint(host: str, port: int) -> dict[str, Any]:
    """Persist an explicit successful connection without replacing bootstrap data.

    The selected endpoint belongs to the user-owned node configuration.  The
    original bootstrap endpoint remains available for deterministic recovery
    when the preferred address cannot be reached.
    """
    endpoint = _normalize_master_endpoint(host, port)
    if endpoint is None:
        raise ValueError("invalid master endpoint")

    data = load_node_config()
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    data["cluster"] = {
        **cluster,
        "preferred_master_endpoint": endpoint,
    }
    write_node_config(data)
    apply_node_config_to_env(data, overwrite=True)
    _sync_loaded_module_attr("config", "CLIENT_MASTER_HOST", endpoint["host"])
    _sync_loaded_module_attr("config", "CLIENT_MASTER_PORT", endpoint["port"])
    return endpoint


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
    preferred_endpoint = get_preferred_master_endpoint(data)
    configured_host = (
        preferred_endpoint["host"]
        if preferred_endpoint is not None
        else cluster.get("master_tcp_host") or cluster.get("master_host")
    )
    configured_port = (
        preferred_endpoint["port"]
        if preferred_endpoint is not None
        else cluster.get("master_tcp_port") or cluster.get("master_port")
    )
    # An explicit successful connection is user-owned runtime state.  It must
    # win over a stale installer or source-checkout .env bootstrap address.
    endpoint_overwrite = overwrite or preferred_endpoint is not None

    _set_env_value("QLH_NODE_ROLE", node.get("role"), overwrite=overwrite)
    _set_env_value("QLH_NODE_ID", node.get("node_id"), overwrite=overwrite)
    _set_env_value("QLH_NODE_TYPE", node.get("node_type"), overwrite=overwrite)
    _set_env_value("QLH_CLUSTER_SECRET", cluster.get("cluster_secret"), overwrite=overwrite)
    _set_env_value(
        "QLH_MASTER_HOST",
        configured_host,
        overwrite=endpoint_overwrite,
    )
    _set_env_value(
        "QLH_MASTER_PORT",
        configured_port,
        overwrite=endpoint_overwrite,
    )
    _set_env_value(
        "QLH_CLIENT_MASTER_HOST",
        configured_host,
        overwrite=endpoint_overwrite,
    )
    _set_env_value(
        "QLH_CLIENT_MASTER_PORT",
        configured_port,
        overwrite=endpoint_overwrite,
    )
    _set_env_value("QLH_MASTER_API_HOST", cluster.get("master_api_host"), overwrite=overwrite)
    _set_env_value("QLH_MASTER_API_PORT", cluster.get("master_api_port"), overwrite=overwrite)
    _set_env_value(
        "QLH_API_PORT",
        cluster.get("master_api_port") if node.get("role") == "master" else None,
        overwrite=overwrite,
    )
    # Feature gates are user-owned runtime preferences.  Keeping them in the
    # node config lets the settings UI survive a restart without putting the
    # flags (or any secrets) into the repository.
    features = data.get("features") if isinstance(data.get("features"), dict) else {}
    # Unlike transport identity, these switches are deliberately controlled
    # by the user's settings UI and therefore override a stale process-level
    # .env value on startup.
    _set_env_value("QLH_TASK_GRAPH_ENABLED", features.get("task_graph_enabled"), overwrite=True)
    _set_env_value(
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED",
        features.get("task_worker_experimental_enabled"),
        overwrite=True,
    )
    return data


def build_bootstrap_config(response: dict[str, Any]) -> dict[str, Any]:
    cluster = response.get("cluster") if isinstance(response.get("cluster"), dict) else {}
    node = response.get("node") if isinstance(response.get("node"), dict) else {}
    existing = load_node_config()
    existing_cluster = (
        existing.get("cluster") if isinstance(existing.get("cluster"), dict) else {}
    )
    features = existing.get("features") if isinstance(existing.get("features"), dict) else {}
    result = {
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
    if features:
        result["features"] = dict(features)
    # A preference is valid only inside the same cluster.  A successful join
    # to another cluster must not silently retain its previous master.
    existing_cluster_id = str(existing_cluster.get("cluster_id", "") or "")
    response_cluster_id = str(cluster.get("cluster_id", "qlh-default") or "")
    if not existing_cluster_id or existing_cluster_id == response_cluster_id:
        preferred_endpoint = get_preferred_master_endpoint(existing)
        if preferred_endpoint is not None:
            result["cluster"]["preferred_master_endpoint"] = preferred_endpoint
    return result


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
            _node_id = str(node["node_id"])
            try:
                from node_runtime import node_runtime
                node_runtime.set_node_id(_node_id)
            except Exception:
                pass
            _sync_loaded_module_attr("scheduler", "NODE_ID", _node_id)
        if node.get("role"):
            _role = str(node["role"])
            try:
                from node_runtime import node_runtime
                node_runtime.set_node_role(_role)
            except Exception:
                pass
            _sync_loaded_module_attr("scheduler", "NODE_ROLE", _role)
    except Exception:
        pass
