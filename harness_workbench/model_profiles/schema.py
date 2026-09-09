"""Serializable schema for a model x quantization x backend profile."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


PROFILE_SCHEMA = "qlh.harness.model_profile.v1"
PROFILE_STATUSES = frozenset({"unknown", "candidate", "verified", "rejected"})
CAPABILITY_NAMES = (
    "json_output",
    "tool_call_generation",
    "tool_result_reinjection",
    "multimodal",
    "thinking_control",
)
CAPABILITY_STATUSES = frozenset({"unknown", "declared", "verified", "rejected"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")


class ProfileValidationError(ValueError):
    """Raised when a profile would be unsafe or not reproducible."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_paths(value: Any, *, key: str = "") -> None:
    if isinstance(value, str):
        if _ABSOLUTE_PATH.match(value) or "\\" in value and key.lower().endswith("path"):
            raise ProfileValidationError("profiles cannot contain absolute paths")
        return
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _reject_paths(child_value, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child_value in value:
            _reject_paths(child_value, key=key)


@dataclass(frozen=True, slots=True)
class CapabilityState:
    """One capability with evidence separate from admission status."""

    status: str = "unknown"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in CAPABILITY_STATUSES:
            raise ProfileValidationError(f"invalid capability status: {self.status}")
        if any(not isinstance(item, str) or not item for item in self.evidence):
            raise ProfileValidationError("capability evidence must contain non-empty strings")
        object.__setattr__(self, "evidence", tuple(sorted(set(self.evidence))))

    @classmethod
    def from_value(cls, value: Any) -> "CapabilityState":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProfileValidationError("capability must be an object")
        evidence = value.get("evidence", ())
        if isinstance(evidence, str):
            evidence = (evidence,)
        if not isinstance(evidence, (list, tuple)):
            raise ProfileValidationError("capability evidence must be a list")
        return cls(status=str(value.get("status", "unknown")), evidence=tuple(evidence))

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "evidence": list(self.evidence)}


def _capability_map(value: Mapping[str, Any] | None) -> dict[str, CapabilityState]:
    raw = value or {}
    if not isinstance(raw, Mapping):
        raise ProfileValidationError("capabilities must be an object")
    unknown = set(raw) - set(CAPABILITY_NAMES)
    if unknown:
        raise ProfileValidationError(
            "unknown capability names: " + ", ".join(sorted(str(item) for item in unknown))
        )
    return {
        name: CapabilityState.from_value(raw.get(name, {"status": "unknown"}))
        for name in CAPABILITY_NAMES
    }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """All model-specific policy needed before an adapter can run."""

    model_id: str
    revision: str
    backend: str
    artifact_sha256: str | None = None
    tokenizer_digest: str | None = None
    chat_template_digest: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)
    adaptation: Mapping[str, Any] = field(default_factory=dict)
    roles: tuple[str, ...] = ("answer",)
    aliases: tuple[str, ...] = ()
    resources: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, CapabilityState] = field(default_factory=dict)
    status: str = "unknown"
    production_eligible: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_id", "revision", "backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProfileValidationError(f"{name} must be a non-empty string")
        if self.status not in PROFILE_STATUSES:
            raise ProfileValidationError(f"invalid profile status: {self.status}")
        if self.artifact_sha256 is not None and (
            not isinstance(self.artifact_sha256, str)
            or not _SHA256.fullmatch(self.artifact_sha256)
        ):
            raise ProfileValidationError("artifact_sha256 must be a lowercase SHA-256 digest")
        for name in ("tokenizer_digest", "chat_template_digest"):
            digest = getattr(self, name)
            if digest is not None and (
                not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            ):
                raise ProfileValidationError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.roles, (list, tuple)) or not self.roles:
            raise ProfileValidationError("roles must contain at least one role")
        if any(not isinstance(role, str) or not role.strip() for role in self.roles):
            raise ProfileValidationError("roles must contain non-empty strings")
        if not isinstance(self.aliases, (list, tuple)):
            raise ProfileValidationError("aliases must be an array")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise ProfileValidationError("aliases must contain non-empty strings")
        if not isinstance(self.production_eligible, bool):
            raise ProfileValidationError("production_eligible must be boolean")
        for value in (self.context, self.generation, self.adaptation, self.resources, self.evidence):
            if not isinstance(value, Mapping):
                raise ProfileValidationError("profile sections must be objects")
        _reject_paths(self.context)
        _reject_paths(self.generation)
        _reject_paths(self.adaptation)
        _reject_paths(self.resources)
        _reject_paths(self.evidence)
        object.__setattr__(self, "roles", tuple(dict.fromkeys(self.roles)))
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(alias.strip() for alias in self.aliases if alias.strip())))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "generation", dict(self.generation))
        object.__setattr__(self, "adaptation", dict(self.adaptation))
        object.__setattr__(self, "resources", dict(self.resources))
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "capabilities", _capability_map(self.capabilities))

    @property
    def profile_id(self) -> str:
        return f"{self.model_id}:{self.backend}:{self.revision}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict(include_digest=False))).hexdigest()

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "profile_schema": PROFILE_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "backend": self.backend,
            "artifact_sha256": self.artifact_sha256,
            "tokenizer_digest": self.tokenizer_digest,
            "chat_template_digest": self.chat_template_digest,
            "context": dict(self.context),
            "generation": dict(self.generation),
            "adaptation": dict(self.adaptation),
            "roles": list(self.roles),
            "aliases": list(self.aliases),
            "resources": dict(self.resources),
            "capabilities": {
                name: capability.as_dict() for name, capability in self.capabilities.items()
            },
            "status": self.status,
            "production_eligible": self.production_eligible,
            "evidence": dict(self.evidence),
        }
        if include_digest:
            value["profile_digest"] = self.digest
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelProfile":
        if not isinstance(value, Mapping):
            raise ProfileValidationError("profile must be an object")
        if value.get("profile_schema") != PROFILE_SCHEMA:
            raise ProfileValidationError("unsupported or missing profile_schema")
        roles_value = value.get("roles", ("answer",))
        if not isinstance(roles_value, (list, tuple)):
            raise ProfileValidationError("roles must be an array")
        aliases_value = value.get("aliases", ())
        if not isinstance(aliases_value, (list, tuple)):
            raise ProfileValidationError("aliases must be an array")
        profile = cls(
            model_id=value.get("model_id", ""),
            revision=value.get("revision", ""),
            backend=value.get("backend", ""),
            artifact_sha256=value.get("artifact_sha256"),
            tokenizer_digest=value.get("tokenizer_digest"),
            chat_template_digest=value.get("chat_template_digest"),
            context=value.get("context", {}),
            generation=value.get("generation", {}),
            adaptation=value.get("adaptation", {}),
            roles=tuple(roles_value),
            aliases=tuple(aliases_value),
            resources=value.get("resources", {}),
            capabilities=value.get("capabilities", {}),
            status=value.get("status", "unknown"),
            production_eligible=value.get("production_eligible", False),
            evidence=value.get("evidence", {}),
        )
        supplied_digest = value.get("profile_digest")
        if supplied_digest is not None and supplied_digest != profile.digest:
            raise ProfileValidationError("profile_digest does not match canonical profile")
        return profile

    def with_updates(self, **changes: Any) -> "ModelProfile":
        """Return a new revisioned value; the original profile stays immutable."""

        return replace(self, **changes)
