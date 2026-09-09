"""User-owned local registry for immutable model profiles."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .capability_gate import CapabilityGate
from .schema import ModelProfile, ProfileValidationError


class ProfileAdmissionError(ValueError):
    """Raised when a profile cannot be registered or promoted."""


_FILENAME = re.compile(r"^[0-9a-f]{32}\.json$")


def _profile_filename(profile: ModelProfile) -> str:
    return profile.digest[:32] + ".json"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def profile_diff(left: ModelProfile, right: ModelProfile) -> dict[str, dict[str, Any]]:
    """Return a stable field-level diff without exposing local paths."""

    left_fields = _flatten(left.as_dict(include_digest=False))
    right_fields = _flatten(right.as_dict(include_digest=False))
    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(left_fields) | set(right_fields)):
        before = left_fields.get(key)
        after = right_fields.get(key)
        if before != after:
            changed[key] = {"before": before, "after": after}
    return changed


class ProfileRegistry:
    """Persist profiles as atomic JSON files under a user-selected directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def register(self, profile: ModelProfile, *, overwrite: bool = False) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / _profile_filename(profile)
        if destination.exists() and not overwrite:
            existing = self._read(destination)
            if existing.digest != profile.digest:
                raise ProfileAdmissionError("profile digest collision or attempted overwrite")
            return destination
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(profile.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def list_profiles(self, *, include_rejected: bool = True) -> tuple[ModelProfile, ...]:
        if not self.root.is_dir():
            return ()
        profiles: list[ModelProfile] = []
        for path in sorted(self.root.glob("*.json")):
            if not _FILENAME.fullmatch(path.name):
                continue
            try:
                profile = self._read(path)
            except (OSError, json.JSONDecodeError, ProfileValidationError):
                continue
            if include_rejected or profile.status != "rejected":
                profiles.append(profile)
        return tuple(sorted(profiles, key=lambda item: item.profile_id))

    def get(
        self,
        model_id: str,
        *,
        backend: str | None = None,
        artifact_sha256: str | None = None,
        revision: str | None = None,
        include_rejected: bool = False,
    ) -> ModelProfile | None:
        candidates = [
            profile for profile in self.list_profiles(include_rejected=include_rejected)
            if profile.model_id == model_id
            and (backend is None or profile.backend == backend)
            and (artifact_sha256 is None or profile.artifact_sha256 == artifact_sha256)
            and (revision is None or profile.revision == revision)
        ]
        if not candidates:
            return None
        rank = {"verified": 3, "candidate": 2, "unknown": 1, "rejected": 0}
        return sorted(candidates, key=lambda item: (-rank[item.status], item.revision), reverse=False)[0]

    def select(self, model_id: str, *, backend: str, artifact_sha256: str | None = None) -> ModelProfile | None:
        """Select the strongest non-rejected profile, preferring exact assets."""

        exact = self.get(model_id, backend=backend, artifact_sha256=artifact_sha256)
        if exact is not None:
            return exact
        return self.get(model_id, backend=backend)

    def promote(
        self,
        profile: ModelProfile,
        *,
        status: str,
        evidence: dict[str, Any] | None = None,
        production_eligible: bool = False,
    ) -> ModelProfile:
        """Create and persist a new immutable profile revision.

        Verified promotion is refused unless the capability gate can prove the
        requested production role.  A candidate can always remain experimental.
        """

        merged_evidence = dict(profile.evidence)
        merged_evidence.update(evidence or {})
        promoted = replace(
            profile,
            revision=f"{profile.revision}+{status}",
            status=status,
            evidence=merged_evidence,
            production_eligible=production_eligible,
        )
        decision = CapabilityGate().evaluate(promoted)
        if status == "verified" and decision.status != "verified":
            raise ProfileAdmissionError(
                "profile cannot be verified: " + ", ".join(decision.reasons)
            )
        if production_eligible and not decision.production_eligible:
            raise ProfileAdmissionError(
                "profile cannot be production eligible: " + ", ".join(decision.reasons)
            )
        self.register(promoted)
        return promoted

    def rollback(self, model_id: str, *, backend: str, revision: str) -> ModelProfile:
        profile = self.get(model_id, backend=backend, revision=revision, include_rejected=True)
        if profile is None:
            raise ProfileAdmissionError("rollback target profile was not found")
        return profile

    @staticmethod
    def _read(path: Path) -> ModelProfile:
        value = json.loads(path.read_text(encoding="utf-8"))
        profile = ModelProfile.from_dict(value)
        if _FILENAME.fullmatch(path.name) and path.stem != profile.digest[:32]:
            raise ProfileValidationError("profile filename digest does not match content")
        return profile
