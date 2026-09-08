"""Versioned model profiles and capability gates for the harness."""

from .builtin import builtin_profiles
from .capability_gate import CapabilityGate, GateDecision
from .probe import ProbeResult, probe_local_model
from .registry import ProfileAdmissionError, ProfileRegistry, profile_diff
from .schema import (
    CAPABILITY_NAMES,
    PROFILE_SCHEMA,
    PROFILE_STATUSES,
    CapabilityState,
    ModelProfile,
    ProfileValidationError,
)

__all__ = [
    "CAPABILITY_NAMES",
    "PROFILE_SCHEMA",
    "PROFILE_STATUSES",
    "CapabilityGate",
    "CapabilityState",
    "GateDecision",
    "ModelProfile",
    "ProbeResult",
    "ProfileAdmissionError",
    "ProfileRegistry",
    "ProfileValidationError",
    "builtin_profiles",
    "probe_local_model",
    "profile_diff",
]
