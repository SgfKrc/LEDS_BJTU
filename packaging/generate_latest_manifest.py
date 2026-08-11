"""Generate a signed ``latest.json`` from a GitHub Release.

The GitHub Releases API already publishes the size and SHA-256 digest of each
uploaded asset.  Using that metadata avoids downloading release installers
again merely to prepare the update manifest.  The resulting file still uses
the normal UP-N2 Ed25519 signing and trusted-key verification gate.

Run this after the release payloads have been uploaded, then upload the
generated ``latest.json`` to the same GitHub Release.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import signing
import update_core
from serve import _classify_update_asset


DEFAULT_REPOSITORY = "SgfKrc/LEDS_BJTU"
DEFAULT_TRUSTED_KEYS_DIR = Path(__file__).with_name("pubkeys")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GITHUB_SHA256_RE = re.compile(r"^sha256:([0-9a-fA-F]{64})$")


class ManifestGenerationError(RuntimeError):
    """Expected release metadata or manifest generation failure."""


def normalize_tag(value: str) -> str:
    """Validate a project version and return it without an optional ``v``."""
    tag = str(value or "").strip()
    if tag[:1].lower() == "v":
        tag = tag[1:]
    try:
        update_core.version_key(tag)
    except update_core.ManifestError as exc:
        raise ManifestGenerationError(f"invalid release tag: {value!r}") from exc
    return tag


def _validate_repository(value: str) -> str:
    repository = str(value or "").strip()
    if not _REPOSITORY_RE.fullmatch(repository):
        raise ManifestGenerationError("repository must be in owner/name form")
    return repository


def github_release_url(repository: str, tag: str) -> str:
    repository = _validate_repository(repository)
    return (
        f"https://api.github.com/repos/{repository}/releases/tags/"
        f"{urllib.parse.quote(tag, safe='')}"
    )


def fetch_release_metadata(
    repository: str,
    tag: str,
    *,
    timeout: float = 20.0,
    github_token: str = "",
) -> Mapping[str, Any]:
    """Fetch one public or authenticated GitHub Release metadata document."""
    if timeout <= 0:
        raise ManifestGenerationError("timeout must be greater than zero")
    request = urllib.request.Request(
        github_release_url(repository, tag),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "QLH-latest-manifest-generator",
        },
    )
    if github_token:
        request.add_header("Authorization", f"Bearer {github_token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ManifestGenerationError(
            f"GitHub Release metadata request failed with HTTP {exc.code}"
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise ManifestGenerationError("could not fetch GitHub Release metadata") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise ManifestGenerationError("GitHub Release metadata response is too large")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestGenerationError("GitHub Release metadata is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ManifestGenerationError("GitHub Release metadata must be a JSON object")
    return value


def build_manifest_from_release(
    release: Mapping[str, Any],
    *,
    tag: str,
    channel: str = "stable",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Convert GitHub Release assets to the project's v1 manifest format.

    Only assets recognized by ``serve._classify_update_asset`` are included.
    A GitHub-provided SHA-256 digest is mandatory: never substitute an
    unverified asset URL for an integrity value.
    """
    expected_tag = normalize_tag(tag)
    release_tag = normalize_tag(str(release.get("tag_name", "")))
    if release_tag != expected_tag:
        raise ManifestGenerationError(
            f"GitHub Release tag {release_tag!r} does not match requested tag "
            f"{expected_tag!r}"
        )
    if release.get("draft") is True:
        raise ManifestGenerationError("refusing to generate a manifest from a draft Release")
    normalized_channel = str(channel or "").strip().lower()
    if not normalized_channel:
        raise ManifestGenerationError("channel must not be empty")
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        raise ManifestGenerationError("GitHub Release metadata has no assets list")

    assets: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_targets: set[tuple[str, str, str, str]] = set()
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            raise ManifestGenerationError("GitHub Release asset must be an object")
        name = str(raw_asset.get("name", ""))
        classification = _classify_update_asset(name)
        if classification is None:
            continue
        if name in seen_names:
            raise ManifestGenerationError(f"duplicate GitHub Release asset name: {name}")
        target = tuple(classification)
        if target in seen_targets:
            raise ManifestGenerationError(
                "multiple assets map to the same update target "
                f"{target}: {name}"
            )
        digest = str(raw_asset.get("digest", ""))
        digest_match = _GITHUB_SHA256_RE.fullmatch(digest)
        if digest_match is None:
            raise ManifestGenerationError(
                f"GitHub Release asset has no usable SHA-256 digest: {name}"
            )
        size = raw_asset.get("size")
        if isinstance(size, bool):
            raise ManifestGenerationError(f"GitHub Release asset has invalid size: {name}")
        url = str(raw_asset.get("browser_download_url", ""))
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ManifestGenerationError(
                f"GitHub Release asset does not have an HTTPS download URL: {name}"
            )
        platform, variant, arch, kind = classification
        asset = {
            "name": name,
            "url": url,
            "size": size,
            "sha256": digest_match.group(1).lower(),
            "platform": platform,
            "variant": variant,
            "arch": arch,
            "kind": kind,
        }
        try:
            update_core.UpdateAsset.from_mapping(asset)
        except update_core.ManifestError as exc:
            raise ManifestGenerationError(f"invalid GitHub Release asset: {name}") from exc
        seen_names.add(name)
        seen_targets.add(target)
        assets.append(asset)

    if not assets:
        raise ManifestGenerationError(
            "GitHub Release contains no recognized QLH update assets"
        )
    assets.sort(key=lambda asset: str(asset["name"]).casefold())
    manifest = {
        "schema_version": 1,
        "tag": expected_tag,
        "channel": normalized_channel,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }
    try:
        update_core.UpdateManifest.from_mapping(manifest)
    except update_core.ManifestError as exc:
        raise ManifestGenerationError("generated manifest does not meet schema v1") from exc
    return manifest


def sign_and_verify_manifest(
    manifest: Mapping[str, Any],
    *,
    private_key_path: str | os.PathLike[str],
    trusted_keys_dir: str | os.PathLike[str],
    key_id: str | None = None,
) -> dict[str, Any]:
    """Sign the manifest, then verify it against the shipped trusted keys."""
    signed = signing.sign_manifest(
        manifest, private_key_path=private_key_path, key_id=key_id,
    )
    verified, reason = signing.verify_manifest_signature(
        signed, trusted_keys_dir=trusted_keys_dir,
    )
    if not verified:
        raise ManifestGenerationError(
            f"signed manifest did not verify against trusted public keys: {reason}"
        )
    return signed


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
    """Atomically write the already-signed manifest as UTF-8 JSON."""
    update_core.save_json_state(path, manifest)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and verify a signed latest.json from GitHub Release assets."
    )
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY, help="GitHub owner/repository")
    parser.add_argument("--tag", required=True, help="Release tag, for example 0.1.8.2")
    parser.add_argument("--channel", default="stable", help="Manifest update channel")
    parser.add_argument("--output", default="latest.json", help="Output manifest path")
    parser.add_argument(
        "--key", default=os.environ.get("QLH_SIGNING_KEY", ""),
        help="Ed25519 release private key path (default: QLH_SIGNING_KEY)",
    )
    parser.add_argument("--key-id", help="Optional signing key ID override")
    parser.add_argument(
        "--trusted-keys-dir", default=str(DEFAULT_TRUSTED_KEYS_DIR),
        help="Trusted public-key directory used for post-sign verification",
    )
    parser.add_argument(
        "--github-token-env", default="GITHUB_TOKEN",
        help="Optional environment variable holding a GitHub API token",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="GitHub API timeout")
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch, generate, sign and verify without writing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if not args.key:
        print("error: provide --key or set QLH_SIGNING_KEY", file=sys.stderr)
        return 2
    try:
        tag = normalize_tag(args.tag)
        token = os.environ.get(args.github_token_env, "") if args.github_token_env else ""
        release = fetch_release_metadata(
            args.repo, tag, timeout=args.timeout, github_token=token,
        )
        manifest = build_manifest_from_release(
            release, tag=tag, channel=args.channel,
        )
        signed = sign_and_verify_manifest(
            manifest,
            private_key_path=args.key,
            trusted_keys_dir=args.trusted_keys_dir,
            key_id=args.key_id,
        )
        if not args.dry_run:
            write_manifest(args.output, signed)
    except (ManifestGenerationError, signing.SigningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    action = "verified without writing" if args.dry_run else "wrote"
    print(
        f"{action} {args.output}: tag {signed['tag']}, "
        f"{len(signed['assets'])} update assets, key {signed['key_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
