"""P7 CLI URL, authentication and output-boundary contracts."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import qlh_log


def test_build_base_url_brackets_ipv6_and_preserves_hostname():
    assert qlh_log._build_base_url("::1", 8000) == "http://[::1]:8000"
    assert qlh_log._build_base_url("[fd7a:115c:a1e0::1]", 9000) == (
        "http://[fd7a:115c:a1e0::1]:9000"
    )
    assert qlh_log._build_base_url("master.example", 8000) == (
        "http://master.example:8000"
    )


def test_build_query_escapes_filters_and_bounds_lines():
    query = qlh_log._build_query(5000, "WARN&ERROR", "api/log?x=1")
    assert query == "limit=1000&level=WARN%26ERROR&name=api%2Flog%3Fx%3D1"
    assert qlh_log._build_query(0) == "limit=1"


def test_request_uses_server_log_token_header_and_structured_url(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(qlh_log.urllib.request, "urlopen", fake_urlopen)
    payload = qlh_log._request(
        "http://[::1]:8000", 
        "/api/cluster/nodes/log-aggregate?limit=2&name=a%2Fb",
        "secret",
    )

    assert payload == {"ok": True}
    assert captured["url"] == (
        "http://[::1]:8000/api/cluster/nodes/log-aggregate?limit=2&name=a%2Fb"
    )
    assert captured["headers"]["X-qlh-log-token"] == "secret"
    assert "Authorization" not in captured["headers"]
    assert captured["timeout"] == 10


def test_print_node_honors_requested_lines(capsys):
    qlh_log._print_node("node-1", ["line-1", "line-2", "line-3"], 2)
    output = capsys.readouterr().out
    assert "line-1" not in output
    assert "line-2" in output
    assert "line-3" in output


def test_build_base_url_rejects_authority_injection():
    with pytest.raises(ValueError):
        qlh_log._build_base_url("host/path", 8000)
