from harness_workbench.adapters import AdapterRequest, QLHAdapter, QLHAdapterConfig
from harness_workbench.adapters.qlh import _usage_from_metrics


class _CacheUsageTransport:
    def get_json(self, path):
        if path == "/api/status":
            return {"model_loaded": True, "current_model": "cache-fixture"}
        raise AssertionError(f"unexpected GET path: {path}")

    def post_json(self, path, payload):
        assert path == "/api/chat"
        return {
            "generation_id": "gen-1",
            "model": payload["model"],
            "content": "answer",
            "metrics": {
                "prompt_tokens": 120,
                "completion_tokens": 8,
                "prompt_cache_hit_tokens": 96,
                "prompt_cache_miss_tokens": 24,
            },
        }

    def post_stream(self, path, payload):
        assert path == "/api/chat/stream"
        return [
            'data: {"token": "answer"}\n',
            'data: {"done": true, "metrics": {"completion_tokens": 8, "prompt_cache_hit_tokens": 96, "prompt_cache_miss_tokens": 24}}\n',
        ]

    def close(self):
        return None


def _request(stream=False):
    return AdapterRequest(
        model="cache-fixture",
        messages=({"role": "user", "content": "hello"},),
        stream=stream,
        max_tokens=16,
    )


def test_qlh_usage_preserves_cache_hit_and_miss_tokens():
    metrics = {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "prompt_cache_hit_tokens": 96,
        "prompt_cache_miss_tokens": 24,
    }
    assert _usage_from_metrics(metrics) == {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "prompt_cache_hit_tokens": 96,
        "prompt_cache_miss_tokens": 24,
        "total_tokens": 128,
    }


def test_qlh_complete_and_stream_expose_cache_usage():
    adapter = QLHAdapter(QLHAdapterConfig("http://master:8000"), transport=_CacheUsageTransport())
    assert adapter.complete(_request()).usage["prompt_cache_hit_tokens"] == 96
    chunks = list(adapter.stream(_request(stream=True)))
    assert chunks[-1].usage == {
        "completion_tokens": 8,
        "prompt_cache_hit_tokens": 96,
        "prompt_cache_miss_tokens": 24,
    }
