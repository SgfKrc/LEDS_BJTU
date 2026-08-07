import io
import os
import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusion.artifacts import DiffusionArtifact
from diffusion.sd15_engine import (
    GenerationCancelled,
    SD15EngineConfig,
    SD15GenerationRequest,
)
from diffusion.service import (
    DiffusionBlobReferencedError,
    DiffusionConflictError,
    DiffusionInputError,
    DiffusionNotFoundError,
    DiffusionService,
    DiffusionUnsupportedError,
    MemoryImageBlobStore,
    SD15EditRequest,
    build_sd15_generation_request,
)


def test_preset_generation_request_carries_the_pinned_scheduler():
    request = build_sd15_generation_request(preset_id="sd15_original_v1")

    assert request.scheduler == "DPMSolverMultistepScheduler"


class _Inspector:
    def __init__(self, artifact_kind="sd15_pipeline", loadable=True):
        self.artifact_kind = artifact_kind
        self.loadable = loadable

    def inspect(self, path, *, compute_hash=False):
        return DiffusionArtifact(
            path=str(path),
            artifact_kind=self.artifact_kind,
            precision="fp16",
            sha256="a" * 64 if compute_hash else "",
            size_bytes=123,
            loadable=self.loadable,
            warnings=[] if self.artifact_kind != "unknown" else ["unknown fixture"],
        )


class _Image:
    def __init__(self, payload=b"fake-png"):
        self.payload = payload

    def save(self, output, *, format):
        assert format == "PNG"
        output.write(self.payload)


class _Engine:
    def __init__(self, config, *, block=False, fail=False):
        self.config = config
        self.block = block
        self.fail = fail
        self.is_loaded = False
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.unload_calls = 0

    @property
    def capabilities(self):
        return {"engine": "fake_sd15"}

    def load(self, path):
        self.path = path
        self.is_loaded = True
        return DiffusionArtifact(path=path, artifact_kind="sd15_pipeline", loadable=True)

    def generate(self, request, *, callback=None):
        self.started.set()
        if self.fail:
            raise RuntimeError("fake generation failed")
        for step in range(request.steps):
            if self.cancelled.is_set():
                raise GenerationCancelled("cancelled")
            if callback:
                callback(step + 1, request.steps)
            if self.block:
                time.sleep(0.005)
        return SimpleNamespace(
            image=_Image(),
            seed=request.seed,
            elapsed_seconds=0.25,
            metadata={"engine": "fake_sd15"},
        )

    def edit(self, request, *, image, mask=None, adapter=None, callback=None):
        self.started.set()
        self.last_edit = {
            'request': request,
            'image': image,
            'mask': mask,
            'adapter': adapter,
        }
        for step in range(request.denoising_steps):
            if self.cancelled.is_set():
                raise GenerationCancelled('cancelled')
            if callback:
                callback(step + 1, request.denoising_steps)
        return SimpleNamespace(
            image=_Image(b'edited-png'),
            seed=request.seed,
            elapsed_seconds=0.5,
            metadata={'engine': 'fake_sd15_img2img', 'strength': request.strength},
        )

    def cancel(self):
        self.cancelled.set()

    def unload(self):
        self.unload_calls += 1
        self.is_loaded = False


class _EngineFactory:
    def __init__(self, *, block=False, fail=False):
        self.block = block
        self.fail = fail
        self.instances = []

    def __call__(self, config):
        engine = _Engine(config, block=self.block, fail=self.fail)
        self.instances.append(engine)
        return engine


class _ResettingCancelEngine(_Engine):
    """Emulate engines that clear their cancel event when generate starts."""

    def __init__(self, config):
        super().__init__(config)
        self.proceed = threading.Event()

    def generate(self, request, *, callback=None):
        self.started.set()
        assert self.proceed.wait(1)
        self.cancelled.clear()
        for step in range(request.steps):
            if callback:
                callback(step + 1, request.steps)
        return SimpleNamespace(
            image=_Image(),
            seed=request.seed,
            elapsed_seconds=0.25,
            metadata={"engine": "fake_sd15"},
        )


class _BlockingEncodeStore(MemoryImageBlobStore):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.proceed = threading.Event()

    def encode_png(self, image):
        data = super().encode_png(image)
        self.entered.set()
        assert self.proceed.wait(1)
        return data


def _wait_for_state(service, job_id, expected, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = service.get_job(job_id)
        if snapshot["state"] == expected:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"job did not reach {expected}: {service.get_job(job_id)}")


def _loaded_service(tmp_path, *, block=False, fail=False, blob_store=None):
    factory = _EngineFactory(block=block, fail=fail)
    service = DiffusionService(
        inspector=_Inspector(),
        engine_factory=factory,
        blob_store=blob_store,
    )
    registered = service.register_artifact(str(tmp_path), artifact_id="sd-test")
    service.load(registered.artifact_id, SD15EngineConfig(device="cpu", dtype="float32"))
    return service, factory


def test_service_registers_loads_generates_and_serves_png(tmp_path):
    service, _factory = _loaded_service(tmp_path)
    try:
        submitted = service.submit_generation(
            SD15GenerationRequest(prompt="test", seed=7, steps=3)
        )
        completed = _wait_for_state(service, submitted["job_id"], "completed")

        assert completed["progress"] == {"step": 3, "total": 3}
        assert completed["metrics"]["engine"] == "fake_sd15"
        blob = service.get_blob(completed["blob"]["blob_id"])
        assert blob.data == b"fake-png"
        assert blob.metadata["job_id"] == submitted["job_id"]
        assert service.snapshot()["active_job"] is None
        assert completed["parameters"]["prompt"] == "test"
        assert completed["parameters"]["negative_prompt"] == ""
        assert completed["output_blob_id"] == completed["blob"]["blob_id"]
        assert completed["error_code"] is None
    finally:
        service.close()


def test_service_rejects_parallel_generation_and_cancels_active_job(tmp_path):
    service, factory = _loaded_service(tmp_path, block=True)
    try:
        submitted = service.submit_generation(
            SD15GenerationRequest(prompt="slow", steps=100)
        )
        assert factory.instances[0].started.wait(1)
        with pytest.raises(DiffusionConflictError, match="another"):
            service.submit_generation(SD15GenerationRequest(prompt="second"))

        cancellation = service.cancel_job(submitted["job_id"])
        assert cancellation["accepted"] is True
        cancelled = _wait_for_state(service, submitted["job_id"], "cancelled")
        assert cancelled["blob"] is None
        assert cancelled["error_code"] == "DIFFUSION_CANCELLED"
        assert service.cancel_job(submitted["job_id"])["accepted"] is False
    finally:
        service.close()


def test_unload_waits_for_active_job_before_releasing_engine(tmp_path):
    service, factory = _loaded_service(tmp_path, block=True)
    submitted = service.submit_generation(
        SD15GenerationRequest(prompt="unload race", steps=100)
    )
    try:
        assert factory.instances[0].started.wait(1)
        unloaded = service.unload()

        assert unloaded["state"] == "unloaded"
        assert service.get_job(submitted["job_id"])["state"] == "cancelled"
        assert factory.instances[0].unload_calls == 1
    finally:
        service.close()


def test_generation_failure_is_contained_in_job_state(tmp_path):
    service, factory = _loaded_service(tmp_path, fail=True)
    try:
        submitted = service.submit_generation(SD15GenerationRequest(prompt="fail"))
        failed = _wait_for_state(service, submitted["job_id"], "failed")
        assert failed["error"] == "fake generation failed"
        assert service.snapshot()["state"] == "loaded"
        assert service.snapshot()["last_error"] == "fake generation failed"

        factory.instances[0].fail = False
        retried = service.submit_generation(SD15GenerationRequest(prompt="retry"))
        _wait_for_state(service, retried["job_id"], "completed")
        assert service.snapshot()["last_error"] is None
    finally:
        service.close()


def test_engine_factory_failure_does_not_leave_service_loading(tmp_path):
    def fail_factory(_config):
        raise RuntimeError("factory failed")

    service = DiffusionService(inspector=_Inspector(), engine_factory=fail_factory)
    registered = service.register_artifact(str(tmp_path), artifact_id="sd-test")
    try:
        with pytest.raises(RuntimeError, match="factory failed"):
            service.load(
                registered.artifact_id,
                SD15EngineConfig(device="cpu", dtype="float32"),
            )
        assert service.snapshot()["state"] == "error"
        assert service.snapshot()["last_error"] == "factory failed"
        assert service.is_busy is False
        assert service.is_loaded is False
    finally:
        service.close()


def test_cancel_before_first_step_survives_engine_cancel_reset(tmp_path):
    engine = _ResettingCancelEngine(SD15EngineConfig(device="cpu", dtype="float32"))
    service = DiffusionService(
        inspector=_Inspector(),
        engine_factory=lambda _config: engine,
    )
    service.register_artifact(str(tmp_path), artifact_id="sd-test")
    service.load("sd-test", SD15EngineConfig(device="cpu", dtype="float32"))
    submitted = service.submit_generation(
        SD15GenerationRequest(prompt="cancel race", steps=3)
    )
    try:
        assert engine.started.wait(1)
        assert service.cancel_job(submitted["job_id"])["accepted"] is True
        engine.proceed.set()
        cancelled = _wait_for_state(service, submitted["job_id"], "cancelled")
        assert cancelled["progress"]["step"] == 0
        assert cancelled["blob"] is None
    finally:
        engine.proceed.set()
        service.close()


def test_cancel_during_png_encoding_never_inserts_blob(tmp_path):
    store = _BlockingEncodeStore()
    factory = _EngineFactory()
    service = DiffusionService(
        inspector=_Inspector(),
        engine_factory=factory,
        blob_store=store,
    )
    service.register_artifact(str(tmp_path), artifact_id="sd-test")
    service.load("sd-test", SD15EngineConfig(device="cpu", dtype="float32"))
    submitted = service.submit_generation(
        SD15GenerationRequest(prompt="encode race", steps=1)
    )
    try:
        assert store.entered.wait(1)
        assert service.cancel_job(submitted["job_id"])["accepted"] is True
        store.proceed.set()
        cancelled = _wait_for_state(service, submitted["job_id"], "cancelled")
        assert cancelled["blob"] is None
        assert store.snapshot()["items"] == 0
    finally:
        store.proceed.set()
        service.close()


def test_unknown_artifact_is_not_registered(tmp_path):
    service = DiffusionService(inspector=_Inspector("unknown", loadable=False))
    try:
        with pytest.raises(ValueError, match="unknown fixture"):
            service.register_artifact(str(tmp_path))
        assert service.list_artifacts() == []
    finally:
        service.close()


def test_incomplete_ip_adapter_is_not_registered(tmp_path):
    service = DiffusionService(
        inspector=_Inspector('sd15_ip_adapter', loadable=False)
    )
    try:
        with pytest.raises(ValueError, match='incomplete'):
            service.register_artifact(str(tmp_path), artifact_id='ip-adapter')
        assert service.list_artifacts() == []
    finally:
        service.close()


def test_switching_loaded_artifact_requires_explicit_unload(tmp_path):
    service, _factory = _loaded_service(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    service.register_artifact(str(other), artifact_id="sd-other")
    config = SD15EngineConfig(device="cpu", dtype="float32")
    try:
        with pytest.raises(DiffusionConflictError, match="unload"):
            service.load("sd-other", config)
        unloaded = service.unload()
        assert unloaded["loaded"] is False
        assert service.load("sd-other", config)["loaded"] is True
    finally:
        service.close()


def test_blob_store_expires_and_evicts_oldest_item():
    now = [100.0]
    store = MemoryImageBlobStore(
        max_items=1,
        max_total_bytes=32,
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    first = store.put_image(_Image(b"first"))
    second = store.put_image(_Image(b"second"))
    with pytest.raises(DiffusionNotFoundError):
        store.get(first.blob_id)
    assert store.get(second.blob_id).data == b"second"

    now[0] = 111.0
    with pytest.raises(DiffusionNotFoundError):
        store.get(second.blob_id)
    assert store.snapshot()["items"] == 0


def test_blob_store_rejects_single_image_larger_than_total_limit():
    store = MemoryImageBlobStore(max_total_bytes=4)
    with pytest.raises(Exception, match="exceeds"):
        store.put_image(_Image(b"12345"))


def _encoded_image(*, size=(16, 12), image_format='PNG', mode='RGB'):
    from PIL import Image

    output = io.BytesIO()
    Image.new(mode, size, 127).save(output, format=image_format)
    return output.getvalue()


def test_blob_store_normalizes_supported_uploads_and_masks():
    from PIL import Image

    store = MemoryImageBlobStore()
    source = store.put_upload(
        _encoded_image(image_format='JPEG'),
        purpose='input_image',
        owner_scope='test-owner',
    )
    mask = store.put_upload(
        _encoded_image(mode='L'),
        purpose='mask',
        owner_scope='test-owner',
    )

    assert source.data.startswith(b'\x89PNG\r\n\x1a\n')
    assert source.descriptor()['purpose'] == 'input_image'
    assert source.descriptor()['owner_scope'] == 'test-owner'
    assert (source.width, source.height) == (16, 12)
    assert Image.open(io.BytesIO(source.data)).mode == 'RGB'
    assert Image.open(io.BytesIO(mask.data)).mode == 'L'

    oriented_output = io.BytesIO()
    oriented_image = Image.new('RGB', (10, 20), 127)
    exif = oriented_image.getexif()
    exif[274] = 6
    oriented_image.save(oriented_output, format='JPEG', exif=exif)
    oriented = store.put_upload(
        oriented_output.getvalue(),
        purpose='input_image',
        owner_scope='test-owner',
    )
    assert (oriented.width, oriented.height) == (20, 10)


def test_blob_store_rejects_invalid_or_oversized_decoded_images():
    store = MemoryImageBlobStore(max_image_pixels=20)
    with pytest.raises(DiffusionInputError, match='valid supported image'):
        store.put_upload(b'not-an-image', purpose='input_image', owner_scope='local')
    with pytest.raises(DiffusionInputError, match='pixel limit'):
        store.put_upload(
            _encoded_image(size=(5, 5)),
            purpose='input_image',
            owner_scope='local',
        )


def test_blob_lease_blocks_deletion_and_expiry_until_released():
    now = [100.0]
    store = MemoryImageBlobStore(ttl_seconds=10, clock=lambda: now[0])
    blob = store.put_upload(
        _encoded_image(),
        purpose='input_image',
        owner_scope='local',
    )
    store.acquire_lease(blob.blob_id)
    now[0] = 111.0

    assert store.get(blob.blob_id).lease_count == 1
    with pytest.raises(DiffusionConflictError, match='in use'):
        store.delete(blob.blob_id)
    store.release_lease(blob.blob_id)
    with pytest.raises(DiffusionNotFoundError):
        store.get(blob.blob_id)


def test_blob_parent_reference_blocks_deletion_until_result_is_removed():
    store = MemoryImageBlobStore()
    parent = store.put_upload(
        _encoded_image(),
        purpose='input_image',
        owner_scope='local',
    )
    result = store.put_png(
        b'result',
        parent_blob_ids=(parent.blob_id,),
        owner_scope='local',
    )

    assert store.get(parent.blob_id).reference_count == 1
    with pytest.raises(DiffusionBlobReferencedError, match='referenced'):
        store.delete(parent.blob_id)

    assert store.delete(result.blob_id) is True
    assert store.get(parent.blob_id).reference_count == 0
    assert store.delete(parent.blob_id) is True


def test_blob_upload_records_original_hash_and_normalized_format():
    source = _encoded_image(image_format='JPEG')
    store = MemoryImageBlobStore()
    blob = store.put_upload(source, purpose='input_image', owner_scope='local')

    assert blob.metadata['upload_sha256'] == hashlib.sha256(source).hexdigest()
    assert blob.metadata['normalized_format'] == 'PNG'


def test_blob_result_creation_never_evicts_its_parent():
    store = MemoryImageBlobStore(max_items=2, max_total_bytes=64)
    parent = store.put_png(b'parent', purpose='input_image')
    unrelated = store.put_png(b'unrelated')

    child = store.put_png(b'child', parent_blob_ids=(parent.blob_id,))

    assert child.parent_blob_ids == (parent.blob_id,)
    assert store.get(parent.blob_id).reference_count == 1
    with pytest.raises(DiffusionNotFoundError):
        store.get(unrelated.blob_id)


def test_edit_contract_validates_blob_purpose_owner_and_mask_dimensions(tmp_path):
    store = MemoryImageBlobStore()
    source = store.put_upload(
        _encoded_image(size=(16, 16)),
        purpose='input_image',
        owner_scope='local',
    )
    mask = store.put_upload(
        _encoded_image(size=(8, 8), mode='L'),
        purpose='mask',
        owner_scope='local',
    )
    service, _factory = _loaded_service(tmp_path)
    service._blob_store = store
    try:
        request = SD15EditRequest(
            mode='inpaint',
            source_blob_id=source.blob_id,
            mask_blob_id=mask.blob_id,
            prompt='restore the image',
            width=512,
            height=512,
        )
        with pytest.raises(DiffusionInputError, match='dimensions'):
            service.validate_edit(request)

        valid = SD15EditRequest(
            mode='img2img',
            source_blob_id=source.blob_id,
            prompt='change the lighting',
            width=512,
            height=512,
        )
        assert service.validate_edit(valid)['source_blob']['blob_id'] == source.blob_id
        submitted = service.submit_edit(valid)
        assert submitted['kind'] == 'edit'
        assert submitted['input_blob_ids'] == [source.blob_id]
        assert submitted['parameters']['denoising_steps'] == 21
        assert submitted['progress']['total'] == 21
        assert store.get(source.blob_id).lease_count == 1
        completed = _wait_for_state(service, submitted['job_id'], 'completed')
        assert completed['blob']['parent_blob_ids'] == [source.blob_id]
        assert completed['blob']['metadata']['edit_mode'] == 'img2img'
        assert completed['blob']['metadata']['source_sha256'] == source.sha256
        assert store.get(source.blob_id).lease_count == 0
        assert completed['progress'] == {'step': 21, 'total': 21}
    finally:
        service.close()


def test_inpaint_queues_source_and_mask_with_a_dedicated_pipeline(tmp_path):
    base_path = tmp_path / 'base'
    inpaint_path = tmp_path / 'inpaint'
    base_path.mkdir()
    inpaint_path.mkdir()

    class _KindByPathInspector:
        def inspect(self, path, *, compute_hash=False):
            is_inpaint = Path(path).name == 'inpaint'
            return DiffusionArtifact(
                path=str(path),
                artifact_kind=(
                    'sd15_inpaint_pipeline' if is_inpaint else 'sd15_pipeline'
                ),
                precision='fp16',
                sha256=('b' * 64 if is_inpaint else 'a' * 64) if compute_hash else '',
                loadable=True,
            )

    store = MemoryImageBlobStore()
    source = store.put_upload(
        _encoded_image(size=(16, 16)),
        purpose='input_image',
        owner_scope='local',
    )
    mask = store.put_upload(
        _encoded_image(size=(16, 16), mode='L'),
        purpose='mask',
        owner_scope='local',
    )
    factory = _EngineFactory()
    service = DiffusionService(
        inspector=_KindByPathInspector(),
        engine_factory=factory,
        blob_store=store,
    )
    service.register_artifact(str(base_path), artifact_id='sd-base')
    service.register_artifact(str(inpaint_path), artifact_id='sd-inpaint')
    service.load(
        'sd-base',
        SD15EngineConfig(device='cpu', dtype='float32'),
    )
    request = SD15EditRequest(
        mode='inpaint',
        source_blob_id=source.blob_id,
        mask_blob_id=mask.blob_id,
        prompt='replace the selected window',
        edit_adapter_id='sd-inpaint',
        width=512,
        height=512,
        steps=2,
        strength=0.5,
    )
    try:
        submitted = service.submit_edit(request)
        completed = _wait_for_state(service, submitted['job_id'], 'completed')

        assert submitted['input_blob_ids'] == [source.blob_id, mask.blob_id]
        assert completed['progress'] == {'step': 1, 'total': 1}
        assert completed['blob']['parent_blob_ids'] == [source.blob_id, mask.blob_id]
        assert completed['blob']['metadata']['edit_mode'] == 'inpaint'
        assert completed['blob']['metadata']['mask_sha256'] == mask.sha256
        assert completed['blob']['metadata']['edit_adapter_id'] == 'sd-inpaint'
        assert factory.instances[0].last_edit['mask'].mode == 'L'
        assert factory.instances[0].last_edit['adapter'].artifact_kind == 'sd15_inpaint_pipeline'
        assert store.get(source.blob_id).lease_count == 0
        assert store.get(mask.blob_id).lease_count == 0
    finally:
        service.close()


def test_edit_request_rejects_zero_effective_denoising_steps():
    request = SD15EditRequest(
        mode='img2img',
        source_blob_id='img_source',
        prompt='change the lighting',
        steps=2,
        strength=0.35,
        width=512,
        height=512,
    )

    with pytest.raises(DiffusionInputError, match='at least one denoising step'):
        request.validate()


def test_reference_edit_requires_registered_ip_adapter_and_tracks_metadata(tmp_path):
    store = MemoryImageBlobStore()
    source = store.put_upload(
        _encoded_image(size=(16, 16)),
        purpose='input_image',
        owner_scope='local',
    )
    service, _factory = _loaded_service(tmp_path, blob_store=store)
    request = SD15EditRequest(
        mode='reference',
        source_blob_id=source.blob_id,
        prompt='keep the same character in a new scene',
        edit_adapter_id='ip-adapter',
        ip_adapter_scale=0.6,
        width=512,
        height=512,
        steps=2,
    )
    try:
        with pytest.raises(DiffusionNotFoundError, match='ip-adapter'):
            service.validate_edit(request)

        service._inspector = _Inspector('sd15_ip_adapter', loadable=True)
        service.register_artifact(
            str(tmp_path / 'ip-adapter'),
            artifact_id='ip-adapter',
        )
        validated = service.validate_edit(request)
        assert validated['edit_adapter']['artifact_id'] == 'ip-adapter'
        assert validated['edit_adapter']['artifact']['sha256'] == 'a' * 64

        submitted = service.submit_edit(request)
        completed = _wait_for_state(service, submitted['job_id'], 'completed')

        assert completed['progress'] == {'step': 2, 'total': 2}
        assert completed['blob']['metadata']['edit_mode'] == 'reference'
        assert completed['blob']['metadata']['edit_adapter_id'] == 'ip-adapter'
        assert completed['blob']['metadata']['ip_adapter_scale'] == 0.6
    finally:
        service.close()


def test_instruction_edit_requires_dedicated_pipeline_and_tracks_guidance(tmp_path):
    store = MemoryImageBlobStore()
    source = store.put_upload(
        _encoded_image(size=(16, 16)),
        purpose='input_image',
        owner_scope='local',
    )
    service, factory = _loaded_service(tmp_path, blob_store=store)
    service._inspector = _Inspector('sd15_instruction_pipeline', loadable=True)
    service.register_artifact(
        str(tmp_path / 'instruct-pix2pix'),
        artifact_id='instruction-pipeline',
    )
    request = SD15EditRequest(
        mode='instruction',
        source_blob_id=source.blob_id,
        prompt='make it a snowy winter day',
        instruction='make it a snowy winter day',
        edit_adapter_id='instruction-pipeline',
        image_guidance_scale=1.0,
        width=512,
        height=512,
        steps=2,
    )
    try:
        validated = service.validate_edit(request)
        assert validated['edit_adapter']['artifact_id'] == 'instruction-pipeline'

        submitted = service.submit_edit(request)
        completed = _wait_for_state(service, submitted['job_id'], 'completed')

        assert completed['progress'] == {'step': 2, 'total': 2}
        assert completed['blob']['metadata']['edit_mode'] == 'instruction'
        assert completed['blob']['metadata']['instruction'] == request.instruction
        assert completed['blob']['metadata']['image_guidance_scale'] == 1.0
        assert completed['blob']['metadata']['edit_adapter_id'] == 'instruction-pipeline'
        assert factory.instances[0].last_edit['adapter'].artifact_kind == 'sd15_instruction_pipeline'
        assert store.get(source.blob_id).lease_count == 0
    finally:
        service.close()


def test_instruction_edit_rejects_controlnet_parameters_for_default_pipeline():
    request = SD15EditRequest(
        mode='instruction',
        source_blob_id='img_source',
        prompt='make it winter',
        instruction='make it winter',
        edit_adapter_id='instruction-pipeline',
        image_guidance_scale=1.0,
        conditioning_scale=0.8,
        width=512,
        height=512,
    )

    with pytest.raises(DiffusionInputError, match='conditioning_scale'):
        request.validate()


@pytest.mark.parametrize(
    ('prompt', 'image_guidance_scale', 'message'),
    [
        ('describe a new target', 1.0, 'prompt must match instruction'),
        ('make it winter', 4.1, 'image_guidance_scale'),
    ],
)
def test_instruction_edit_rejects_ambiguous_prompt_or_excessive_guidance(
    prompt,
    image_guidance_scale,
    message,
):
    request = SD15EditRequest(
        mode='instruction',
        source_blob_id='img_source',
        prompt=prompt,
        instruction='make it winter',
        edit_adapter_id='instruction-pipeline',
        image_guidance_scale=image_guidance_scale,
        width=512,
        height=512,
    )

    with pytest.raises(DiffusionInputError, match=message):
        request.validate()


@pytest.mark.parametrize('scale', [-0.1, 2.1, float('nan')])
def test_reference_edit_rejects_invalid_ip_adapter_scale(scale):
    request = SD15EditRequest(
        mode='reference',
        source_blob_id='img_reference',
        prompt='same character',
        edit_adapter_id='ip-adapter',
        ip_adapter_scale=scale,
        width=512,
        height=512,
    )

    with pytest.raises(DiffusionInputError, match='ip_adapter_scale'):
        request.validate()


def test_reference_edit_rejects_unvalidated_profile_before_queueing(tmp_path):
    factory = _EngineFactory()
    service = DiffusionService(
        inspector=_Inspector(),
        engine_factory=factory,
    )
    service.register_artifact(str(tmp_path), artifact_id='sd-test')
    config = SD15EngineConfig(
        device='cpu',
        dtype='float32',
        enable_attention_slicing=False,
        enable_qkv_fusion=True,
    )
    service.load('sd-test', config)
    source = service._blob_store.put_upload(
        _encoded_image(size=(16, 16)),
        purpose='input_image',
        owner_scope='local',
    )
    service._inspector = _Inspector('sd15_ip_adapter', loadable=True)
    service.register_artifact(str(tmp_path / 'ip-adapter'), artifact_id='ip-adapter')
    request = SD15EditRequest(
        mode='reference',
        source_blob_id=source.blob_id,
        prompt='same character',
        edit_adapter_id='ip-adapter',
        ip_adapter_scale=0.6,
        width=512,
        height=512,
    )
    try:
        with pytest.raises(DiffusionUnsupportedError, match='non-quantized'):
            service.submit_edit(request)
        assert service.snapshot()['jobs'] == 0
        assert service.snapshot()['active_job'] is None
    finally:
        service.close()


def test_output_blob_can_be_reused_for_edit_with_the_same_owner(tmp_path):
    store = MemoryImageBlobStore()
    source = store.put_png(
        _encoded_image(size=(16, 16)),
        purpose='output',
        owner_scope='inference-local',
        width=16,
        height=16,
    )
    service, _factory = _loaded_service(tmp_path, blob_store=store)
    request = SD15EditRequest(
        mode='img2img',
        source_blob_id=source.blob_id,
        prompt='continue editing the generated image',
        width=512,
        height=512,
        steps=2,
    )
    try:
        with pytest.raises(DiffusionNotFoundError):
            service.validate_edit(request, owner_scope='local')

        submitted = service.submit_edit(
            request,
            owner_scope='inference-local',
        )
        completed = _wait_for_state(service, submitted['job_id'], 'completed')

        assert completed['owner_scope'] == 'inference-local'
        assert completed['blob']['owner_scope'] == 'inference-local'
        assert completed['blob']['parent_blob_ids'] == [source.blob_id]
        assert store.get(source.blob_id).reference_count == 1
    finally:
        service.close()


def test_generation_preserves_explicit_owner_scope(tmp_path):
    service, _factory = _loaded_service(tmp_path)
    try:
        submitted = service.submit_generation(
            SD15GenerationRequest(prompt='owned output', steps=1),
            owner_scope='inference-local',
        )
        completed = _wait_for_state(service, submitted['job_id'], 'completed')

        assert completed['owner_scope'] == 'inference-local'
        assert completed['blob']['owner_scope'] == 'inference-local'
    finally:
        service.close()
