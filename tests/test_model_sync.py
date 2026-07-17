"""Tests for Tailnet PyTorch model synchronization."""

import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_missing_deepseek_model_is_downloaded_and_verified(tmp_path, monkeypatch):
    from model_sync import ensure_model_available
    import model_sync

    model_id = "deepseek-r1-distill-qwen-1.5b"
    source_files = {
        "config.json": b'{"num_hidden_layers": 28}',
        "tokenizer_config.json": b'{"model_max_length": 4096}',
        "model.safetensors": b"deepseek-test-weights",
    }
    weight_sha = hashlib.sha256(source_files["model.safetensors"]).hexdigest()
    digest = hashlib.sha256()
    for name, content in sorted(source_files.items()):
        file_sha = hashlib.sha256(content).hexdigest()
        digest.update(f"{name}\0{len(content)}\0{file_sha}\n".encode())
    model_sha = digest.hexdigest()
    manifest = {
        "model_id": model_id,
        "sha256": model_sha,
        "total_layers": 28,
        "files": [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in source_files.items()
        ],
    }

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            requests.append(parsed.path)
            if parsed.path == "/api/models/downloadable":
                payload = json.dumps(manifest).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            prefix = f"/api/models/files/{model_id}/"
            if parsed.path.startswith(prefix):
                relative_path = unquote(parsed.path[len(prefix):])
                payload = source_files[relative_path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker_path = tmp_path / "models" / model_id
    monkeypatch.setattr(
        model_sync,
        "resolve_worker_model_path",
        lambda requested_id: str(worker_path),
    )
    try:
        result = ensure_model_available(
            "127.0.0.1",
            server.server_port,
            model_id,
            model_sha,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == str(worker_path)
    assert (worker_path / "model.safetensors").read_bytes() == source_files["model.safetensors"]
    assert (worker_path / "config.json").read_bytes() == source_files["config.json"]
    assert (worker_path / "model.sha256").read_text(encoding="utf-8").startswith(model_sha)
    assert len([path for path in requests if "/api/models/files/" in path]) == 3


def test_model_digest_uses_verified_weight_content(tmp_path):
    import model_sync

    root = tmp_path / "deepseek"
    root.mkdir()
    weight = root / "model-00001-of-000001.safetensors"
    weight.write_bytes(b"small fixture standing in for a large shard")
    metadata = root / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    oid = "a" * 64
    (metadata / f"{weight.name}.metadata").write_text(
        f"revision\n{oid}\ntimestamp\n",
        encoding="utf-8",
    )

    value = model_sync.compute_model_sha256(str(root), use_cache=False)

    content_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    expected = hashlib.sha256(
        f"{weight.name}\0{weight.stat().st_size}\0{content_sha}\n".encode("utf-8")
    ).hexdigest()
    assert value == expected


def test_model_digest_changes_with_config_tokenizer_and_remote_code(tmp_path):
    import model_sync

    root = tmp_path / "qwen"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (root / "qwen.tiktoken").write_bytes(b"tokenizer-v1")
    (root / "modeling_qwen.py").write_text("VERSION = 1\n", encoding="utf-8")

    baseline = model_sync.compute_model_sha256(str(root), use_cache=False)
    for name, content in (
        ("config.json", '{"model_type":"qwen2"}'),
        ("qwen.tiktoken", b"tokenizer-v2"),
        ("modeling_qwen.py", "VERSION = 2\n"),
    ):
        path = root / name
        original = path.read_bytes()
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        assert model_sync.compute_model_sha256(str(root), use_cache=False) != baseline
        path.write_bytes(original)


def test_same_size_corrupt_weight_is_redownloaded(tmp_path, monkeypatch):
    """大模型不能只凭文件大小和缓存摘要跳过内容校验。"""
    from model_sync import ensure_model_available
    import model_sync

    model_id = "deepseek-corrupt"
    good_weight = b"verified-weight-content"
    corrupt_weight = b"x" * len(good_weight)
    file_sha = hashlib.sha256(good_weight).hexdigest()
    model_sha = hashlib.sha256(
        f"model.safetensors\0{len(good_weight)}\0{file_sha}\n".encode()
    ).hexdigest()
    manifest = {
        "model_id": model_id,
        "sha256": model_sha,
        "files": [{
            "path": "model.safetensors",
            "size_bytes": len(good_weight),
            "sha256": file_sha,
        }],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/models/downloadable":
                payload = json.dumps(manifest).encode("utf-8")
            else:
                payload = good_weight
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    worker_path = tmp_path / model_id
    worker_path.mkdir()
    (worker_path / "model.safetensors").write_bytes(corrupt_weight)
    (worker_path / "model.sha256").write_text(f"{model_sha}  stale\n", encoding="utf-8")
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(worker_path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ensure_model_available("127.0.0.1", server.server_port, model_id, model_sha)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (worker_path / "model.safetensors").read_bytes() == good_weight


def test_stale_weight_shard_is_removed(tmp_path, monkeypatch):
    """同一模型 ID 更新分片布局后，清单外旧权重不能污染模型摘要。"""
    from model_sync import ensure_model_available
    import model_sync

    model_id = "deepseek-updated"
    weight = b"current-weight"
    file_sha = hashlib.sha256(weight).hexdigest()
    model_sha = hashlib.sha256(
        f"model.safetensors\0{len(weight)}\0{file_sha}\n".encode()
    ).hexdigest()
    manifest = {
        "model_id": model_id,
        "sha256": model_sha,
        "files": [{
            "path": "model.safetensors",
            "size_bytes": len(weight),
            "sha256": file_sha,
        }],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            payload = (
                json.dumps(manifest).encode("utf-8")
                if parsed.path == "/api/models/downloadable" else weight
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    worker_path = tmp_path / model_id
    worker_path.mkdir()
    (worker_path / "model.safetensors").write_bytes(weight)
    stale = worker_path / "model-00002-of-00002.safetensors"
    stale.write_bytes(b"old-shard")
    monkeypatch.setattr(model_sync, "resolve_worker_model_path", lambda _mid: str(worker_path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = ensure_model_available(
            "127.0.0.1", server.server_port, model_id, model_sha
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == str(worker_path)
    assert not stale.exists()
