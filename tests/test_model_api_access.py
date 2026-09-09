"""Model API network-boundary tests."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from model_api_access import is_model_api_source_trusted, require_model_api_source


def _request(host: str):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_model_api_defaults_to_loopback_only():
    assert is_model_api_source_trusted(_request("127.0.0.1")) is True
    assert is_model_api_source_trusted(_request("100.64.10.20")) is False
    assert is_model_api_source_trusted(SimpleNamespace(client=None)) is False


def test_model_api_accepts_explicit_remote_cidr():
    request = _request("100.64.10.20")
    assert is_model_api_source_trusted(request, trusted_cidrs="100.64.0.0/10") is True


def test_model_api_rejects_untrusted_source_with_stable_code():
    with pytest.raises(HTTPException) as exc:
        require_model_api_source(_request("192.0.2.10"))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "MODEL_API_SOURCE_UNTRUSTED"


@pytest.mark.parametrize("path", ["/api/models/registry", "/api/models/downloads"])
def test_model_api_http_routes_reject_remote_source(path):
    with TestClient(api_server.app, client=("192.0.2.10", 50000)) as client:
        response = client.get(path)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "MODEL_API_SOURCE_UNTRUSTED"
