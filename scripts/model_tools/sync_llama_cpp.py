"""Check or apply the QLH llama.cpp version and patch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(__file__).with_name("llama_quantize.lock.json")


class SyncError(RuntimeError):
    """Raised when the checked-out llama.cpp cannot satisfy the contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise SyncError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.rstrip("\r\n")


def _android_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().absolute()
    configured = os.environ.get("QLH_ANDROID_ROOT")
    if configured:
        return Path(configured).expanduser().absolute()
    return ROOT / "android"


def _contract(lock_path: Path = LOCK_PATH) -> tuple[str, list[dict[str, Any]]]:
    """Read the lock and return (revision, patches).

    ★ 2026-09-20：契约由「**恰好一个**补丁」放宽为「**一个或多个**补丁」。
    每个补丁条目声明：
      * ``path``    —— 补丁文件（相对仓库根）
      * ``targets`` —— 它改动的文件（相对 submodule 根）；兼容旧字段 ``target``（单数）
      * ``marker``  —— 判定「是否已应用」的标记串
      * ``sha256``  —— 补丁文件的摘要（防止内容漂移）
    """
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError("cannot read llama.cpp lock") from exc
    if not isinstance(lock, dict):
        raise SyncError("llama.cpp lock must be an object")
    upstream = lock.get("upstream")
    patch_list = lock.get("patches")
    if not isinstance(upstream, dict) or not isinstance(patch_list, list) or not patch_list:
        raise SyncError("llama.cpp lock must contain one upstream and at least one patch contract")
    revision = upstream.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise SyncError("invalid llama.cpp revision in lock")
    # ★ 2026-09-20：允许 lock 声明一个「本地补丁 commit」（submodule HEAD 应当是它）。
    #   未声明时回落到 upstream.revision（即「补丁只在工作树、不提交」的形态）。
    local_revision = lock.get("revision_local")
    if local_revision is not None and (
        not isinstance(local_revision, str) or len(local_revision) != 40
    ):
        raise SyncError("invalid llama.cpp local revision in lock")
    expected_revision = local_revision or revision

    patches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in patch_list:
        if not isinstance(entry, dict):
            raise SyncError("invalid llama.cpp patch contract")
        if any(not isinstance(entry.get(key), str) for key in ("path", "marker", "sha256")):
            raise SyncError("invalid llama.cpp patch contract")
        targets = _patch_targets(entry)
        if not targets:
            raise SyncError("llama.cpp patch contract needs target or targets")
        digest = entry["sha256"].lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SyncError("invalid llama.cpp patch SHA-256 in lock")
        patch_id = entry.get("id") or entry["path"]
        if patch_id in seen_ids:
            raise SyncError(f"duplicate llama.cpp patch id in lock: {patch_id}")
        seen_ids.add(patch_id)
        patches.append({**entry, "id": patch_id, "targets": targets})
    return expected_revision, patches


def _patch_targets(entry: dict[str, Any]) -> list[str]:
    """Normalize a patch entry's targets, accepting both `targets` and legacy `target`."""
    raw = entry.get("targets")
    if raw is None:
        single = entry.get("target")
        raw = [single] if isinstance(single, str) and single else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def _safe_relative(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SyncError(f"contract path escapes repository: {value}") from exc
    return candidate


def _status_paths(source: Path) -> set[str]:
    output = _git(source, "status", "--porcelain", "--untracked-files=all")
    paths: set[str] = set()
    for line in output.splitlines():
        if not line:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        paths.add(path.replace("\\", "/"))
    return paths


def inspect_sync(
    *,
    android_root: Path | None = None,
    lock_path: Path = LOCK_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate the llama.cpp checkout against **every** patch in the contract.

    ★ 2026-09-20：由「单补丁」改为「多补丁」。每个补丁独立判定 applied /
    ready_to_apply，并逐个校验：
      * 补丁文件存在且 sha256 与 lock 一致；
      * submodule 的 HEAD 等于 lock 的 revision；
      * **worktree 中允许出现的改动，恰好是所有补丁 targets 的并集**
        （原先要求「只有唯一 target 有改动」⇒ 多补丁必然误报 unrelated changes）；
      * marker 在 ⇒ 必须可 reverse（已应用）；marker 不在 ⇒ 必须可 apply（待应用）。
    """
    revision, patch_contracts = _contract(lock_path)
    source = _android_root(android_root) / "app" / "src" / "main" / "cpp" / "llama.cpp"
    if not source.is_dir():
        raise SyncError("Android llama.cpp submodule is missing")

    checked_revision = _git(source, "rev-parse", "HEAD")
    if checked_revision != revision:
        raise SyncError(
            f"llama.cpp revision mismatch: expected {revision}, got {checked_revision}"
        )

    # 先把所有补丁的路径/target 解析并校验，再统一比对 worktree 改动集合。
    entries: list[dict[str, Any]] = []
    allowed_paths: set[str] = set()
    for contract in patch_contracts:
        patch_path = _safe_relative(ROOT, contract["path"])
        if not patch_path.is_file():
            raise SyncError(f"managed llama.cpp patch is missing: {contract['path']}")
        if _sha256(patch_path) != contract["sha256"].lower():
            raise SyncError(
                f"managed llama.cpp patch SHA-256 does not match the lock: {contract['path']}"
            )
        relative_targets: list[str] = []
        for raw_target in contract["targets"]:
            target = (source / raw_target).resolve()
            try:
                target.relative_to(source.resolve())
            except ValueError as exc:
                raise SyncError("llama.cpp patch target escapes the submodule") from exc
            if not target.is_file():
                raise SyncError(
                    f"llama.cpp patch target is missing: {raw_target} (patch {contract['id']})"
                )
            relative_targets.append(target.relative_to(source).as_posix())
        allowed_paths.update(relative_targets)
        entries.append({**contract, "patch_path": patch_path, "relative_targets": relative_targets})

    status_paths = _status_paths(source)
    unexpected = sorted(path for path in status_paths if path not in allowed_paths)
    if unexpected:
        raise SyncError(f"llama.cpp worktree has unrelated changes: {', '.join(unexpected[:8])}")

    results: list[dict[str, Any]] = []
    for entry in entries:
        marker = entry["marker"]
        applied = any(
            marker in (source / rel).read_text(encoding="utf-8", errors="replace")
            for rel in entry["relative_targets"]
        )
        if applied:
            # ★ 2026-09-20：区分「补丁在 git commit 里」与「补丁只在工作树」。
            #   若该补丁的 targets 都不在 worktree 改动集合里，说明它已经被提交
            #   （HEAD 就是 revision_local），此时 reverse-check 必然失败 ——
            #   不能据此判为错误，而要如实报 `applied_committed`。
            dirty = any(rel in status_paths for rel in entry["relative_targets"])
            if not dirty:
                state = "applied_committed"
            else:
                reverse = subprocess.run(
                    ["git", "-C", str(source), "apply", "--reverse", "--check",
                     str(entry["patch_path"])],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                if reverse.returncode != 0:
                    raise SyncError(
                        f"llama.cpp target contains the marker but the managed patch "
                        f"{entry['id']} is not reversible"
                    )
                state = "applied"
        else:
            check = subprocess.run(
                ["git", "-C", str(source), "apply", "--check", str(entry["patch_path"])],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if check.returncode != 0:
                detail = (check.stderr or check.stdout).strip()[:500]
                raise SyncError(f"llama.cpp patch {entry['id']} does not apply cleanly: {detail}")
            state = "ready_to_apply"
        results.append(
            {
                "id": entry["id"],
                "patch": entry["path"],
                "patch_sha256": entry["sha256"].lower(),
                "targets": entry["relative_targets"],
                "state": state,
            }
        )

    # 顶层字段保留首个补丁的信息，兼容既有消费方；新增 `patches` 给出全量。
    first = results[0]
    result: dict[str, Any] = {
        "revision": checked_revision,
        "patch": first["patch"],
        "patch_sha256": first["patch_sha256"],
        "target": first["targets"][0],
        "state": ("applied" if all(item["state"].startswith("applied") for item in results)
                  else "ready_to_apply"),
        "patches": results,
        "dry_run": dry_run,
    }
    if dry_run:
        pending = [item["id"] for item in results if item["state"] == "ready_to_apply"]
        if pending:
            result["action"] = "would_apply"
            result["pending"] = pending
    return result


def apply_sync(*, android_root: Path | None = None, lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    result = inspect_sync(android_root=android_root, lock_path=lock_path, dry_run=False)
    pending = [item for item in result["patches"] if item["state"] == "ready_to_apply"]
    if not pending:
        result["action"] = "already_applied"
        return result
    source = _android_root(android_root) / "app" / "src" / "main" / "cpp" / "llama.cpp"
    applied_ids: list[str] = []
    for item in pending:
        _git(source, "apply", str(_safe_relative(ROOT, item["patch"])))
        applied_ids.append(item["id"])
    result = inspect_sync(android_root=android_root, lock_path=lock_path, dry_run=False)
    result["action"] = "applied"
    result["applied"] = applied_ids
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-root", type=Path, help="path to the qlh-android checkout")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    parser.add_argument("--apply", action="store_true", help="apply the managed patch")
    args = parser.parse_args(argv)
    if args.dry_run and args.apply:
        parser.error("--dry-run and --apply are mutually exclusive")
    try:
        result = (
            apply_sync(android_root=args.android_root)
            if args.apply
            else inspect_sync(android_root=args.android_root, dry_run=args.dry_run)
        )
    except SyncError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
