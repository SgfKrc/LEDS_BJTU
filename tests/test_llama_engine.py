import os
import sys
import threading
import ctypes
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llama_engine import LlamaCppEngine


def make_fake_mtmd():
    calls = {
        "free_context": 0,
        "free_bitmap": 0,
        "free_chunks": 0,
        "free_batch": 0,
    }
    mtmd = types.SimpleNamespace()
    class FakeInputText(ctypes.Structure):
        _fields_ = [
            ("text", ctypes.c_char_p),
            ("add_special", ctypes.c_bool),
            ("parse_special", ctypes.c_bool),
        ]

    mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT = 0
    mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
    mtmd.mtmd_bitmap_p_ctypes = ctypes.c_void_p
    mtmd.mtmd_input_text = lambda data, add_special, parse_special: FakeInputText(
        data, add_special, parse_special,
    )
    mtmd.mtmd_helper_post_decode_callback = lambda callback: callback
    mtmd.mtmd_context_params_default = lambda: types.SimpleNamespace(use_gpu=False, n_threads=0)
    mtmd.mtmd_init_from_file = lambda path, model, params: "mtmd-context"
    mtmd.mtmd_free = lambda context: calls.__setitem__("free_context", calls["free_context"] + 1)
    mtmd.mtmd_support_vision = lambda context: True
    mtmd.mtmd_support_audio = lambda context: False
    mtmd.mtmd_default_marker = lambda: b"<image>"
    mtmd.mtmd_helper_bitmap_init_from_file = lambda context, path, placeholder: types.SimpleNamespace(
        bitmap=123, video_ctx=None,
    )
    mtmd.mtmd_bitmap_free = lambda bitmap: calls.__setitem__("free_bitmap", calls["free_bitmap"] + 1)
    mtmd.mtmd_input_chunks_init = lambda: [0, 1]
    mtmd.mtmd_input_chunks_free = lambda chunks: calls.__setitem__("free_chunks", calls["free_chunks"] + 1)
    mtmd.mtmd_input_chunks_size = lambda chunks: len(chunks)
    mtmd.mtmd_input_chunks_get = lambda chunks, index: {"type": chunks[index]}
    mtmd.mtmd_input_chunk_get_type = lambda chunk: chunk["type"]
    mtmd.mtmd_tokenize = lambda context, chunks, text, bitmaps, count: 0

    def eval_text(context, llama_context, chunk, n_past, seq_id, n_batch, logits_last, new_n_past):
        new_n_past._obj.value += 2
        return 0

    def decode_image(context, llama_context, chunk, embedding, n_past, seq_id, n_batch, new_n_past, callback, user_data):
        new_n_past._obj.value += 3
        return 0

    mtmd.mtmd_helper_eval_chunk_single = eval_text
    mtmd.mtmd_batch_init = lambda context: "mtmd-batch"
    mtmd.mtmd_batch_free = lambda batch: calls.__setitem__("free_batch", calls["free_batch"] + 1)
    mtmd.mtmd_batch_add_chunk = lambda batch, chunk: 0
    mtmd.mtmd_batch_encode = lambda batch: 0
    mtmd.mtmd_batch_get_output_embd = lambda batch, chunk: object()
    mtmd.mtmd_helper_decode_image_chunk = decode_image
    mtmd.calls = calls
    return mtmd


class FakeNativeContext:
    def __init__(self):
        self.ctx = object()
        self.cleared = 0

    def kv_cache_clear(self):
        self.cleared += 1

    def decode(self, batch):
        return None


class FakeNativeModel:
    def __init__(self):
        self.model = object()
        self._ctx = FakeNativeContext()
        self.n_tokens = 4

    def detokenize(self, tokens):
        assert tokens == [7]
        return b"answer"


class FakeNativeBatch:
    closed = 0

    def __init__(self, **kwargs):
        self.last = None

    def set_batch(self, tokens, n_past, logits_all):
        self.last = (tokens, n_past, logits_all)

    def close(self):
        FakeNativeBatch.closed += 1


class FakeNativeSampler:
    closed = 0

    def __init__(self):
        self.tokens = iter([101, 7, 9])

    def add_greedy(self):
        pass

    def add_top_k(self, value):
        pass

    def add_top_p(self, value):
        pass

    def add_temp(self, value):
        pass

    def add_dist(self, value):
        pass

    def sample(self, context, index):
        return next(self.tokens)

    def accept(self, token):
        pass

    def close(self):
        FakeNativeSampler.closed += 1


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


def test_mtmd_capabilities_are_registered_and_released(tmp_path):
    mtmd = make_fake_mtmd()
    engine = LlamaCppEngine()
    engine._model = FakeNativeModel()
    engine._model_path = "gemma-4.gguf"
    engine._loaded = True
    engine._mtmd_module = mtmd
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"projector")

    capabilities = engine.load_mmproj(str(mmproj))

    assert capabilities["native_mtmd"] is True
    assert capabilities["vision"] is True
    assert capabilities["audio"] is False
    assert engine.get_model_info()["capabilities"] == capabilities
    engine.unload()
    assert mtmd.calls["free_context"] == 1


def test_chat_image_runs_native_pipeline_and_frees_resources(monkeypatch, tmp_path):
    mtmd = make_fake_mtmd()
    fake_llama = types.ModuleType("llama_cpp")
    fake_llama.llama_pos = ctypes.c_int
    fake_llama.llama_token_bos = lambda context: 1
    fake_llama.llama_token_eos = lambda context: 9
    fake_internals = types.SimpleNamespace(LlamaBatch=FakeNativeBatch, LlamaSampler=FakeNativeSampler)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)
    monkeypatch.setitem(sys.modules, "llama_cpp._internals", fake_internals)

    image = tmp_path / "sample.png"
    image.write_bytes(b"not-a-real-image-for-fake-binding")
    engine = LlamaCppEngine()
    engine._model = FakeNativeModel()
    engine._model_path = "gemma-4.gguf"
    engine._loaded = True
    engine._mtmd_module = mtmd
    engine._mtmd_context = "mtmd-context"
    engine._mtmd_capabilities = {"vision": True, "audio": False}

    result = engine.chat_image(str(image), prompt="What is this? <__media__>", max_tokens=8)

    assert result["content"] == "answer"
    assert result["finish_reason"] == "stop"
    assert result["native_mtmd"] is True
    assert result["usage"]["prompt_tokens"] == 5
    assert result["usage"]["completion_tokens"] == 2
    assert engine._model._ctx.cleared == 1
    assert mtmd.calls["free_bitmap"] == 1
    assert mtmd.calls["free_chunks"] == 1
    assert mtmd.calls["free_batch"] == 1
    assert FakeNativeBatch.closed == 1
    assert FakeNativeSampler.closed == 1


def test_chat_image_requires_registered_vision(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    engine = LlamaCppEngine()
    engine._model = FakeNativeModel()
    engine._loaded = True

    try:
        engine.chat_image(str(image))
    except RuntimeError as exc:
        assert "vision capability" in str(exc)
    else:
        raise AssertionError("chat_image should require native vision registration")


# ================================================================
# G4.5：GPU offload 预算门与便捷加载
# ================================================================

def test_estimate_gpu_layers_budget_gate():
    """8GB 显存场景：部分 offload 层数在合理区间（不承诺全量，不归零）。"""
    layers = LlamaCppEngine.estimate_gpu_layers(
        36, 7_662_533_088, int(7.4 * 2**30),
    )
    # 每层 ~0.2GB；预算 = 7.4*1.15 - 0.5 ≈ 8.0GB → ~40 层 → 封顶 36
    assert layers == 36
    # 显存紧张（4GB，与 SD 并驻留场景）：显著少于全量
    low = LlamaCppEngine.estimate_gpu_layers(
        36, 7_662_533_088, int(4 * 2**30),
    )
    assert 0 < low < 36
    # 显存不足 → 0（不强行 offload）
    zero = LlamaCppEngine.estimate_gpu_layers(
        36, 7_662_533_088, int(0.4 * 2**30),
    )
    assert zero == 0


def test_load_gemma4_native_fails_closed_on_vram_shortage(monkeypatch):
    """require_gpu_layers 不满足时 fail-closed（不加载）。"""
    monkeypatch.setattr(LlamaCppEngine, "_vram_free_bytes", staticmethod(lambda: int(0.5 * 2**30)))
    engine = LlamaCppEngine()
    try:
        engine.load_gemma4_native(gpu_layers=-1, require_gpu_layers=8)
    except RuntimeError as exc:
        assert "显存预算不足" in str(exc)
    else:
        raise AssertionError("显存不足时应 fail-closed")
