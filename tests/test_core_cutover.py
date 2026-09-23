"""Small boundary checks for the core-only runtime cutover."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server


def test_core_api_does_not_mount_a_product_frontend():
    assert all(getattr(route, "name", None) != "frontend" for route in api_server.app.routes)


def test_core_api_has_no_retired_image_generation_surface():
    paths = set(api_server.app.openapi()["paths"])
    paths.update(
        route.path
        for router in api_server._api_route_modules
        for route in router.router.routes
    )
    assert paths.isdisjoint({"/api/diffusion", "/v1/images/generations"})
