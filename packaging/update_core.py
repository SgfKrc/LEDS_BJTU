"""Small, dependency-free update core shared by the GUI and TUI launcher.

The core deliberately does not start installers.  It only parses a manifest,
selects a matching artifact, downloads it atomically, and verifies its size
and SHA-256.  Manifest signatures (UP-N2) are verified against the bundled
pubkeys directory via packaging/signing.py; the core stays frontend-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


class UpdateError(RuntimeError):
    """Expected update failure that can be rendered by GUI or CLI."""


class ManifestError(UpdateError):
    """Manifest is malformed or cannot be trusted by the current parser."""


class DownloadError(UpdateError):
    """Artifact download or verification failed."""


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_ASSET_BYTES = 32 * 1024 * 1024 * 1024
_VERSION_RE = re.compile(
    r"^[vV]?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)(?:\.(?P<revision>0|[1-9]\d*))?"
    r"(?:-(?P<pre>[0-9A-Za-z.-]+))?$"
)


def _pre_key(value: str) -> tuple[tuple[int, int | str], ...]:
    result: list[tuple[int, int | str]] = []
    for token in value.split("."):
        if token.isdigit():
            result.append((0, int(token)))
        else:
            result.append((1, token.lower()))
    return tuple(result)


def version_key(
    value: str,
) -> tuple[int, int, int, int, int, tuple[tuple[int, int | str], ...]]:
    """Return a comparable project-version key; releases sort after prereleases.

    QLH historically uses both three-part versions and a fourth numeric package
    revision (for example 0.1.8.1), so both forms are accepted.
    """
    match = _VERSION_RE.fullmatch(str(value).strip())
    if not match:
        raise ManifestError(f"invalid semantic version: {value!r}")
    pre = match.group("pre")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        int(match.group("revision") or 0),
        1 if pre is None else 0,
        () if pre is None else _pre_key(pre),
    )


def normalize_arch(value: str | None) -> str:
    raw = (value or "").lower().replace("-", "_")
    if raw in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if raw in {"arm64", "aarch64"}:
        return "aarch64"
    if raw in {"armv7", "armv7l"}:
        return "armv7"
    return raw or "unknown"


@dataclass(frozen=True)
class UpdateAsset:
    name: str
    url: str
    size: int
    sha256: str
    platform: str
    variant: str
    arch: str = "any"
    kind: str = "installer"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UpdateAsset":
        required = ("name", "url", "size", "sha256", "platform", "variant")
        missing = [key for key in required if key not in value]
        if missing:
            raise ManifestError(f"asset missing fields: {', '.join(missing)}")
        name = str(value["name"])
        if (
            not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ManifestError(f"unsafe asset name: {name!r}")
        try:
            size = int(value["size"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"invalid asset size: {name}") from exc
        if size < 0 or size > _MAX_ASSET_BYTES:
            raise ManifestError(f"asset size out of range: {name}")
        sha256 = str(value["sha256"]).lower()
        if not _SHA256_RE.fullmatch(sha256):
            raise ManifestError(f"invalid asset sha256: {name}")
        url = str(value["url"])
        if not url:
            raise ManifestError(f"empty asset URL: {name}")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"", "http", "https"}:
            raise ManifestError(f"unsupported asset URL scheme: {url}")
        platform = str(value["platform"]).lower().strip()
        variant = str(value["variant"]).lower().strip()
        if not platform or not variant:
            raise ManifestError(f"empty asset platform or variant: {name}")
        return cls(
            name=name,
            url=url,
            size=size,
            sha256=sha256,
            platform=platform,
            variant=variant,
            arch=normalize_arch(str(value.get("arch", "any"))),
            kind=str(value.get("kind", "installer")).lower(),
        )


@dataclass(frozen=True)
class UpdateManifest:
    schema_version: int
    tag: str
    channel: str
    assets: tuple[UpdateAsset, ...]
    source_url: str = ""
    signature_present: bool = False
    signature_verified: bool = False
    signature_key_id: str = ""
    signature_signed_at: str = ""
    signature_error: str = ""

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_url: str = ""
    ) -> "UpdateManifest":
        try:
            schema_version = int(value.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise ManifestError("manifest schema_version must be an integer") from exc
        if schema_version != 1:
            raise ManifestError("unsupported update manifest schema")
        tag = str(value.get("tag", ""))
        version_key(tag)
        raw_assets = value.get("assets")
        if not isinstance(raw_assets, list):
            raise ManifestError("manifest assets must be a list")
        if any(not isinstance(item, Mapping) for item in raw_assets):
            raise ManifestError("manifest asset must be an object")
        assets = tuple(UpdateAsset.from_mapping(item) for item in raw_assets)
        return cls(
            schema_version=schema_version,
            tag=tag,
            channel=str(value.get("channel", "stable")).lower(),
            assets=assets,
            source_url=source_url,
            signature_present=bool(value.get("signature")),
            # A signature field is only metadata until a trusted-key verifier
            # has validated it.  Never promote presence to trust.
            signature_verified=False,
            signature_key_id=str(value.get("key_id", "")),
            signature_signed_at=str(value.get("signed_at", "")),
        )


def _read_json_url(
    url: str,
    *,
    timeout: float,
    opener: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UpdateError("update source must use HTTP or HTTPS")
    request = opener or urllib.request.urlopen
    try:
        with request(url, timeout=timeout) as response:
            payload = response.read(_MAX_MANIFEST_BYTES + 1)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise UpdateError(f"cannot fetch update manifest: {url}: {exc}") from exc
    try:
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise ManifestError(f"update manifest is too large: {url}")
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid JSON manifest from {url}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest root must be an object: {url}")
    return value


def fetch_manifest(
    url: str,
    *,
    timeout: float = 8.0,
    opener: Callable[..., Any] | None = None,
    trusted_keys_dir: str | os.PathLike[str] | None = None,
) -> UpdateManifest:
    """Fetch and parse one manifest, resolving relative asset URLs.

    When ``trusted_keys_dir`` is provided (or set via QLH_TRUSTED_KEYS_DIR),
    a present ``signature`` field is verified against the trusted Ed25519
    key set; verification failure only ever leaves
    ``signature_verified=False`` with a reason in ``signature_error``.
    """
    raw = _read_json_url(url, timeout=timeout, opener=opener)
    # Verify the signature against the raw manifest body BEFORE any URL
    # resolution rewrites it; the signature binds the bytes the publisher
    # actually signed.
    from signing import default_trusted_keys_dir, verify_manifest_signature

    signature_error = ""
    signature_verified = False
    if raw.get("signature"):
        trusted = trusted_keys_dir or default_trusted_keys_dir()
        signature_verified, signature_error = verify_manifest_signature(
            raw, trusted_keys_dir=trusted,
        )
    resolved = dict(raw)
    assets = []
    raw_assets = raw.get("assets", [])
    if isinstance(raw_assets, list):
        for item in raw_assets:
            if not isinstance(item, Mapping):
                raise ManifestError("manifest asset must be an object")
            copied = dict(item)
            copied["url"] = urllib.parse.urljoin(url, str(copied.get("url", "")))
            assets.append(copied)
        resolved["assets"] = assets
    manifest = UpdateManifest.from_mapping(resolved, source_url=url)
    if raw.get("signature"):
        from dataclasses import replace

        if not signature_verified:
            manifest = replace(
                manifest, signature_verified=False, signature_error=signature_error,
            )
        else:
            manifest = replace(manifest, signature_verified=True)
    return manifest


def fetch_latest(
    urls: Iterable[str],
    *,
    timeout: float = 8.0,
    fetcher: Callable[..., UpdateManifest] | None = None,
) -> tuple[UpdateManifest, tuple[str, ...]]:
    """Fetch all usable sources and choose the greatest version deterministically."""
    manifests: list[UpdateManifest] = []
    failures_by_url: dict[str, str] = {}
    fetch = fetcher or fetch_manifest
    seen: set[str] = set()
    candidates: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(url)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(candidates)))) as executor:
        pending = {
            executor.submit(fetch, url, timeout=timeout): url for url in candidates
        }
        for future in as_completed(pending):
            url = pending[future]
            try:
                manifests.append(future.result())
            except UpdateError as exc:
                failures_by_url[url] = f"{url}: {exc}"
            except Exception as exc:
                failures_by_url[url] = f"{url}: unexpected update source error: {exc}"
    failures = [failures_by_url[url] for url in candidates if url in failures_by_url]
    if not manifests:
        detail = "; ".join(failures) or "no update source configured"
        raise UpdateError(detail)
    return max(manifests, key=lambda item: version_key(item.tag)), tuple(failures)


def select_asset(
    manifest: UpdateManifest,
    *,
    platform: str,
    variant: str,
    arch: str,
    kind: str = "installer",
) -> UpdateAsset:
    platform = platform.lower()
    variant = variant.lower()
    arch = normalize_arch(arch)
    exact = [
        asset for asset in manifest.assets
        if asset.platform == platform
        and asset.variant == variant
        and asset.kind == kind
        and asset.arch in {"any", arch}
    ]
    if not exact:
        raise UpdateError(
            f"no matching {kind}: platform={platform}, variant={variant}, arch={arch}"
        )
    exact.sort(key=lambda item: (item.arch != arch, item.name.lower()))
    return exact[0]


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | os.PathLike[str], asset: UpdateAsset) -> bool:
    candidate = Path(path)
    return (
        candidate.is_file()
        and candidate.stat().st_size == asset.size
        and sha256_file(candidate) == asset.sha256
    )


def download_asset(
    asset: UpdateAsset,
    destination: str | os.PathLike[str],
    *,
    timeout: float = 30.0,
    opener: Callable[..., Any] | None = None,
) -> Path:
    """Download one asset to a verified file using a sibling .part file."""
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / asset.name
    if verify_file(target, asset):
        return target
    if target.exists():
        target.unlink()
    part = Path(str(target) + ".part")
    if part.exists():
        part.unlink()
    request = opener or urllib.request.urlopen
    try:
        with request(asset.url, timeout=timeout) as response, open(part, "wb") as output:
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > asset.size:
                    raise DownloadError(f"asset exceeds manifest size: {asset.name}")
                output.write(chunk)
    except DownloadError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(f"download failed: {asset.name}: {exc}") from exc
    if not verify_file(part, asset):
        part.unlink(missing_ok=True)
        raise DownloadError(f"asset verification failed: {asset.name}")
    os.replace(part, target)
    return target


def default_state_dir() -> Path:
    override = os.environ.get("QLH_LAUNCHER_STATE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        return Path(root or Path.home()) / "QLH-Edge-Inference"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "qlh"


def load_json_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_json_state(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False,
    ) as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)
