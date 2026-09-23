"""Shared setup for APIRouters that keep api_server as their compatibility facade."""

from __future__ import annotations

from types import ModuleType


def configure_route_module(
    namespace: dict[str, object],
    module: ModuleType,
    resolution_names: tuple[str, ...] = (),
) -> None:
    namespace["_api_module"] = module
    api_globals = vars(module)
    for name in resolution_names:
        if name not in namespace and name in api_globals:
            namespace[name] = api_globals[name]
