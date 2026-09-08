"""Fail-closed capability and profile admission decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schema import CapabilityState, ModelProfile


@dataclass(frozen=True, slots=True)
class GateDecision:
    """A serializable decision for one profile and one requested role."""

    profile_id: str
    status: str
    production_eligible: bool
    allowed_modes: tuple[str, ...]
    reasons: tuple[str, ...]
    capabilities: Mapping[str, CapabilityState]

    def can(self, mode: str) -> bool:
        return mode in self.allowed_modes

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "status": self.status,
            "production_eligible": self.production_eligible,
            "allowed_modes": list(self.allowed_modes),
            "reasons": list(self.reasons),
            "capabilities": {
                name: capability.as_dict() for name, capability in self.capabilities.items()
            },
        }


class CapabilityGate:
    """Decide what a profile may do without guessing unknown capabilities."""

    def evaluate(self, profile: ModelProfile, *, role: str | None = None) -> GateDecision:
        reasons: list[str] = []
        status = profile.status
        rejected = [
            name for name, capability in profile.capabilities.items()
            if capability.status == "rejected"
        ]
        if rejected:
            reasons.append("capability_rejected:" + ",".join(sorted(rejected)))
        if profile.artifact_sha256 is None:
            reasons.append("artifact_digest_missing")
        if profile.evidence.get("artifact_digest_mode") != "full_stream":
            reasons.append("artifact_content_hash_pending")
        if not profile.evidence.get("fixture_set"):
            reasons.append("fixture_evidence_missing")
        if profile.status == "rejected" or rejected:
            status = "rejected"
        elif profile.status == "verified":
            if (
                profile.artifact_sha256 is None
                or profile.evidence.get("artifact_digest_mode") != "full_stream"
                or not profile.evidence.get("fixture_set")
            ):
                status = "candidate"
                reasons.append("verified_requires_bound_artifact_and_fixture")
            else:
                status = "verified"
        elif profile.status == "candidate":
            status = "candidate"
        else:
            status = "unknown"

        allowed: list[str] = []
        if status != "rejected":
            allowed.extend(("answer", "host_router"))
            if status == "candidate":
                allowed.append("experimental")
            if (
                status == "verified"
                and self._verified(profile, "tool_call_generation")
                and self._verified(profile, "tool_result_reinjection")
            ):
                allowed.append("autonomous_tools")
            if status == "verified" and self._verified(profile, "multimodal"):
                allowed.append("multimodal")
        if role == "tool_router" and "autonomous_tools" not in allowed:
            reasons.append("tool_router_requires_verified_tool_contract")
        if role == "vision" and "multimodal" not in allowed:
            reasons.append("vision_role_requires_verified_multimodal")
        production_eligible = (
            status == "verified"
            and profile.production_eligible
            and not reasons
        )
        if profile.production_eligible and not production_eligible:
            reasons.append("production_eligibility_conditions_not_met")
        return GateDecision(
            profile_id=profile.profile_id,
            status=status,
            production_eligible=production_eligible,
            allowed_modes=tuple(allowed),
            reasons=tuple(dict.fromkeys(reasons)),
            capabilities=profile.capabilities,
        )

    @staticmethod
    def _verified(profile: ModelProfile, name: str) -> bool:
        capability = profile.capabilities.get(name)
        return capability is not None and capability.status == "verified"
