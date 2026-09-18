"""Small lazy-loading boundary for the optional PyTorch runtime."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType


def loaded_torch() -> ModuleType | None:
    """Return an already-loaded torch module without importing it."""

    module = sys.modules.get("torch")
    return module if isinstance(module, ModuleType) else None


def torch_available() -> bool:
    """Check whether torch can be resolved without importing its heavy package."""

    if loaded_torch() is not None:
        return True
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def require_torch() -> ModuleType:
    """Import and return torch for an execution path that explicitly needs it."""

    module = loaded_torch()
    if module is not None:
        return module
    return importlib.import_module("torch")


def cuda_available(*, load: bool = False) -> bool:
    """Return CUDA status, optionally loading torch when the caller opts in."""

    module = require_torch() if load else loaded_torch()
    if module is None:
        return False
    try:
        return bool(module.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


class LazyTorch:
    """Attribute-compatible proxy that imports torch only on first use."""

    def __getattr__(self, name: str):
        return getattr(require_torch(), name)

    def __bool__(self) -> bool:
        return torch_available()
