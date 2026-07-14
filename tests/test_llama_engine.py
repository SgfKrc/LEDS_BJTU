import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llama_engine import LlamaCppEngine


class FakeChatStream:
    def __init__(self, cancel_event):
        self.cancel_event = cancel_event
        self.closed = False
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= 3:
            raise StopIteration
        self.index += 1
        if self.index == 1:
            self.cancel_event.set()
        return {
            "choices": [{
                "delta": {"content": f"token-{self.index}"},
                "finish_reason": None,
            }],
        }

    def close(self):
        self.closed = True


class FakeLlamaModel:
    def __init__(self, cancel_event):
        self.stream = FakeChatStream(cancel_event)
        self.call_kwargs = None

    def create_chat_completion(self, **kwargs):
        self.call_kwargs = kwargs
        return self.stream

    def tokenize(self, text, add_bos=True, special=False):
        return [1] if text else []


def test_chat_cancel_event_stops_llama_stream_at_token_boundary():
    cancel_event = threading.Event()
    model = FakeLlamaModel(cancel_event)
    engine = LlamaCppEngine()
    engine._model = model
    engine._model_path = "fake.gguf"
    engine._loaded = True

    result = engine.chat(
        [{"role": "user", "content": "question"}],
        max_tokens=10,
        _cancel_event=cancel_event,
    )

    assert model.call_kwargs["stream"] is True
    assert model.stream.index == 1
    assert model.stream.closed is True
    assert result["content"] == "token-1"
    assert result["finish_reason"] == "cancelled"
    assert result["usage"]["completion_tokens"] == 1
    assert result["usage_estimated"] is True


def test_chat_pre_cancelled_does_not_start_llama_generation():
    cancel_event = threading.Event()
    cancel_event.set()
    model = FakeLlamaModel(cancel_event)
    engine = LlamaCppEngine()
    engine._model = model
    engine._model_path = "fake.gguf"
    engine._loaded = True

    result = engine.chat(
        [{"role": "user", "content": "question"}],
        _cancel_event=cancel_event,
    )

    assert model.call_kwargs is None
    assert result["content"] == ""
    assert result["finish_reason"] == "cancelled"
