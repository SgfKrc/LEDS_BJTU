"""Contract tests for the read-only MODEL-TOOLS P0 trio."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.model_tools.cli import main
from scripts.model_tools.gguf import inspect_gguf, verify_gguf
from scripts.model_tools.gguf_convert import GGUFConvertError, execute_conversion, plan_conversion
from scripts.model_tools.maintenance import clean_models, model_disk_usage
from scripts.model_tools.sd15_batch import run_prompt_batch, run_sampler_matrix
from scripts.model_tools.sweep import sweep_models
from scripts.model_tools.sync_status import build_inventory, compare_inventories, validate_inventory
from scripts.model_tools.llm_smoke_matrix import fixed_prompts, run_smoke_matrix
from scripts.model_tools.llm_smoke_worker import execute_request, validate_output
from scripts.model_tools.lora import MAX_HEADER_BYTES, inspect_lora


def _write_hf_fixture(root: Path, architecture: str = "Qwen2ForCausalLM") -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        '{"architectures":["' + architecture + '"],"model_type":"fixture"}',
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights" * 128)
    return root


def _write_converter_fixture(root: Path, architecture: str = "Qwen2ForCausalLM") -> Path:
    converter = root / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True, exist_ok=True)
    converter.write_text("# fixture\n", encoding="utf-8")
    conversion = root / "conversion"
    conversion.mkdir()
    (conversion / "fixture.py").write_text(
        'from somewhere import ModelBase\n@ModelBase.register("' + architecture + '")\nclass Fixture: pass\n',
        encoding="utf-8",
    )
    return converter


def _write_copy_converter(root: Path, *, fail_code: int = 0, race_target: Path | None = None) -> Path:
    converter = _write_converter_fixture(root)
    if fail_code:
        converter.write_text(f"raise SystemExit({fail_code})\n", encoding="utf-8")
        return converter
    race = repr(str(race_target)) if race_target is not None else "None"
    converter.write_text(
        "import os, shutil, sys\n"
        "args = sys.argv[1:]\n"
        "output = args[args.index('--outfile') + 1]\n"
        "shutil.copyfile(os.environ['QLH_FAKE_GGUF'], output)\n"
        f"race = {race}\n"
        "open(race, 'wb').write(b'race-owner') if race else None\n",
        encoding="utf-8",
    )
    return converter


def _write_copy_quantizer(path: Path) -> Path:
    path.write_text(
        "import shutil, sys\n"
        "shutil.copyfile(sys.argv[1], sys.argv[2])\n",
        encoding="utf-8",
    )
    return path


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I I", 4, value)


def _write_gguf(path: Path, *, payload: bytes = b"\x00" * 16) -> str:
    metadata = b"".join([
        _kv_string("general.architecture", "test"),
        _kv_string("general.name", "tiny-test"),
        _kv_u32("general.alignment", 32),
        _kv_u32("test.context_length", 2048),
        _kv_u32("test.block_count", 2),
    ])
    tensor_prefix = _string("blk.0.weight") + struct.pack("<I Q I", 1, 4, 0)
    header_prefix = b"GGUF" + struct.pack("<I Q Q", 3, 1, 5) + metadata + tensor_prefix
    data_offset = (len(header_prefix) + 31) // 32 * 32
    tensor = tensor_prefix + struct.pack("<Q", 0)
    header = b"GGUF" + struct.pack("<I Q Q", 3, 1, 5) + metadata + tensor
    content = header + (b"\x00" * (data_offset - len(header))) + payload
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_safetensors_header(path: Path, header: dict, payload: bytes = b"\x00" * 32) -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_inspect_reads_metadata_and_tensor_descriptors(tmp_path: Path):
    target = tmp_path / "tiny.gguf"
    _write_gguf(target)

    report = inspect_gguf(target)

    assert report["valid"] is True
    assert report["version"] == 3
    assert report["derived"] == {
        "architecture": "test",
        "name": "tiny-test",
        "context_length": 2048,
        "block_count": 2,
        "vocab_size": None,
        "tokenizer_model": None,
        "tensor_types": {"F32": 1},
    }
    assert report["tensors"] == [{
        "name": "blk.0.weight",
        "shape": [4],
        "type": "F32",
        "type_code": 0,
        "offset": 0,
        "byte_size": 16,
    }]


def test_verify_checks_sidecar_and_detects_tampering(tmp_path: Path):
    target = tmp_path / "tiny.gguf"
    digest = _write_gguf(target)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")

    good = verify_gguf(target)
    assert good["valid"] is True
    assert good["sha256_checked"] is True
    assert good["sha256"] == digest

    target.write_bytes(target.read_bytes()[:-1] + b"X")
    bad = verify_gguf(target)
    assert bad["valid"] is False
    assert any("sha256 mismatch" in error for error in bad["errors"])


def test_verify_rejects_truncated_tensor_data(tmp_path: Path):
    target = tmp_path / "truncated.gguf"
    _write_gguf(target, payload=b"\x00" * 8)

    report = verify_gguf(target)

    assert report["valid"] is False
    assert any("tensor data truncated" in error for error in report["errors"])


def test_models_sweep_reports_integrity_and_orphan_candidates_without_writes(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    good = root / "good.gguf"
    digest = _write_gguf(good)
    (root / "good.gguf.sha256").write_text(f"{digest}  good.gguf\n", encoding="utf-8")
    (root / "download.part").write_bytes(b"partial")
    (root / "orphan.sha256").write_text("a" * 64 + "  missing.gguf\n", encoding="utf-8")
    before = {item: item.stat().st_mtime_ns for item in root.iterdir()}

    report = sweep_models(root)

    assert report["valid"] is True
    assert report["read_only"] is True
    assert report["gguf"][0]["valid"] is True
    assert "download.part" in report["orphan_files"]
    assert "orphan.sha256" in report["orphan_files"]
    assert before == {item: item.stat().st_mtime_ns for item in root.iterdir()}


def test_cli_json_output_and_exit_code(tmp_path: Path, capsys):
    target = tmp_path / "tiny.gguf"
    _write_gguf(target)

    assert main(["gguf_inspect", str(target), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"tensor_count": 1' in output


def test_sd15_lora_inspect_reads_only_header_and_redacts_training_text(tmp_path: Path, capsys):
    target = tmp_path / "portrait_lora.safetensors"
    private_trigger = "private trigger phrase"
    private_dataset = "C:/private/training-images"
    _write_safetensors_header(target, {
        "__metadata__": {
            "ss_network_dim": "64",
            "ss_learning_rate": "0.0001",
            "ss_tag_frequency": json.dumps({private_dataset: {private_trigger: 9}}),
            "ss_dataset_dirs": json.dumps({private_dataset: {"img_count": 12}}),
            "ss_custom_comment": "private training comment",
        },
        "lora_unet_down_blocks_0.lora_down.weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
        },
        "lora_unet_down_blocks_0.lora_up.weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [32, 64],
        },
    }, payload=b"x" * 64)
    before = target.read_bytes()

    report = inspect_lora(target)

    assert report["valid"] is True
    assert report["read_only"] is True
    assert report["weights_loaded"] is False
    assert report["cuda_used"] is False
    assert report["input_kind"] == "safetensors_file"
    assert report["tensor_summary"] == {
        "tensor_count": 2,
        "lora_down_tensor_count": 1,
        "lora_up_tensor_count": 1,
        "alpha_tensor_count": 0,
        "lora_detected": True,
        "components": ["unet"],
    }
    fields = report["metadata"]["fields"]
    assert fields["ss_network_dim"] == {"kind": "numeric", "value": "64"}
    assert fields["ss_tag_frequency"]["numeric_entry_count"] == 1
    assert fields["ss_dataset_dirs"]["numeric_value_total"] == 12
    rendered = json.dumps(report)
    assert private_trigger not in rendered
    assert private_dataset not in rendered
    assert "private training comment" not in rendered
    assert str(tmp_path) not in rendered
    assert target.read_bytes() == before

    assert main(["sd15_lora_inspect", str(target), "--json"]) == 0
    cli_output = capsys.readouterr().out
    assert private_trigger not in cli_output
    assert str(tmp_path) not in cli_output


@pytest.mark.parametrize("header, payload, expected_code", [
    ({"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 99]}}, b"x", "tensor_out_of_range"),
    ({"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}, "other": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"xxxx", "overlapping_tensors"),
])
def test_sd15_lora_inspect_rejects_invalid_tensor_layout(tmp_path: Path, header: dict, payload: bytes, expected_code: str):
    target = tmp_path / "invalid.safetensors"
    _write_safetensors_header(target, header, payload=payload)

    report = inspect_lora(target)

    assert report["valid"] is False
    assert report["errors"][0]["code"] == expected_code


def test_sd15_lora_inspect_rejects_unsupported_dtype_and_size_mismatch(tmp_path: Path):
    target = tmp_path / "bad_dtype.safetensors"
    _write_safetensors_header(target, {
        "tensor": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 1]},
    }, payload=b"x")
    unsupported = inspect_lora(target)
    assert unsupported["valid"] is False
    assert unsupported["errors"][0]["code"] == "invalid_tensor_descriptor"

    _write_safetensors_header(target, {
        "tensor": {"dtype": "F16", "shape": [2], "data_offsets": [0, 2]},
    }, payload=b"xx")
    mismatch = inspect_lora(target)
    assert mismatch["valid"] is False
    assert mismatch["errors"][0]["code"] == "tensor_size_mismatch"


def test_sd15_lora_inspect_rejects_large_headers_and_path_escape(tmp_path: Path):
    target = tmp_path / "large.safetensors"
    target.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1) + b"{}")

    too_large = inspect_lora(target)

    assert too_large["valid"] is False
    assert too_large["errors"][0]["code"] == "header_too_large"

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    _write_safetensors_header(outside, {})
    escaped = inspect_lora(outside, root=root)
    assert escaped["valid"] is False
    assert escaped["errors"][0]["code"] == "path_outside_root"


def test_sd15_lora_inspect_accepts_no_metadata_and_rejects_truncated_prefix(tmp_path: Path):
    empty_metadata = tmp_path / "no_metadata.safetensors"
    _write_safetensors_header(empty_metadata, {
        "lora_te_text_model.lora_down.weight": {
            "dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8],
        },
        "lora_te_text_model.lora_up.weight": {
            "dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16],
        },
    }, payload=b"x" * 16)

    report = inspect_lora(empty_metadata)

    assert report["valid"] is True
    assert report["metadata"] == {"ss_field_count": 0, "fields": {}}
    assert report["tensor_summary"]["lora_detected"] is True

    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(b"bad")
    bad = inspect_lora(truncated)
    assert bad["valid"] is False
    assert bad["errors"][0]["code"] == "truncated_prefix"


def test_model_disk_usage_groups_top_level_assets_and_is_read_only(tmp_path: Path):
    root = tmp_path / "models"
    (root / "qwen").mkdir(parents=True)
    (root / "qwen" / "config.json").write_text("{}", encoding="utf-8")
    (root / "qwen" / "weights.safetensors").write_bytes(b"weights")
    (root / "single.gguf").write_bytes(b"gguf")

    report = model_disk_usage(root)

    assert report["valid"] is True
    assert report["read_only"] is True
    assert report["totals"]["file_count"] == 3
    assert {row["path"] for row in report["entries"]} == {"qwen", "single.gguf"}
    assert report["entries"][0]["logical_size_bytes"] > 0


def test_models_clean_defaults_to_dry_run_and_requires_confirmation(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    partial = root / "download.part"
    partial.write_bytes(b"partial")
    old = partial.stat().st_mtime - 48 * 3600
    import os
    os.utime(partial, (old, old))

    dry_run = clean_models(root, min_age_hours=24)
    assert dry_run["valid"] is True
    assert dry_run["read_only"] is True
    assert dry_run["deleted"] == []
    assert partial.exists()

    refused = clean_models(root, apply=True, min_age_hours=24)
    assert refused["valid"] is False
    assert partial.exists()


def test_models_clean_confirmed_apply_removes_only_stale_candidates(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    stale = root / "old.tmp"
    fresh = root / "active.part"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"active")
    import os
    old = stale.stat().st_mtime - 48 * 3600
    os.utime(stale, (old, old))

    report = clean_models(root, apply=True, confirmation="CLEAN", min_age_hours=24)

    assert report["valid"] is True
    assert report["deleted"] == ["old.tmp"]
    assert not stale.exists()
    assert fresh.exists()


def test_models_clean_detects_duplicates_only_with_explicit_opt_in(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "a.gguf").write_bytes(b"same-model")
    (root / "b.gguf").write_bytes(b"same-model")

    without_opt_in = clean_models(root, min_age_hours=0)
    assert without_opt_in["candidates"] == []

    with_opt_in = clean_models(root, min_age_hours=0, include_duplicates=True)
    assert len(with_opt_in["candidates"]) == 1
    assert with_opt_in["candidates"][0]["kind"] == "duplicate_model"
    assert with_opt_in["candidates"][0]["safe_to_delete"] is False

    untouched = clean_models(root, apply=True, confirmation="CLEAN", min_age_hours=0)
    assert untouched["deleted"] == []
    assert (root / "a.gguf").exists() and (root / "b.gguf").exists()

    applied = clean_models(
        root,
        apply=True,
        confirmation="CLEAN",
        min_age_hours=0,
        include_duplicates=True,
    )
    assert applied["valid"] is True
    assert applied["deleted"] == []
    assert (root / "a.gguf").exists() and (root / "b.gguf").exists()


def test_models_clean_collapses_cache_tree_to_one_candidate(tmp_path: Path):
    root = tmp_path / "models"
    cache = root / ".cache"
    cache.mkdir(parents=True)
    (cache / "orphan.sha256").write_text("0" * 64, encoding="utf-8")

    report = clean_models(root, min_age_hours=0)

    assert [(item["kind"], item["path"]) for item in report["candidates"]] == [("cache", ".cache")]
    assert report["candidates"][0]["safe_to_delete"] is False

    default_apply = clean_models(root, apply=True, confirmation="CLEAN", min_age_hours=0)
    assert default_apply["deleted"] == []
    assert cache.exists()

    cache_apply = clean_models(
        root,
        apply=True,
        confirmation="CLEAN",
        min_age_hours=0,
        include_caches=True,
    )
    assert cache_apply["deleted"] == [".cache"]
    assert not cache.exists()


def test_models_clean_zero_age_includes_fresh_cache_despite_clock_boundary(tmp_path: Path, monkeypatch):
    from scripts.model_tools import maintenance

    root = tmp_path / "models"
    cache = root / ".cache"
    cache.mkdir(parents=True)
    (cache / "orphan.sha256").write_text("0" * 64, encoding="utf-8")

    # Windows file timestamps may briefly be later than the initial scan clock.
    monkeypatch.setattr(maintenance, "_tree_mtime", lambda _path: time.time() + 1.0)

    report = clean_models(root, min_age_hours=0)

    assert [(item["kind"], item["path"]) for item in report["candidates"]] == [("cache", ".cache")]


def test_disk_tools_never_follow_nested_symlinks(tmp_path: Path):
    root = tmp_path / "models"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "outside.part"
    secret.write_bytes(b"outside")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        import pytest
        pytest.skip("directory symlinks are unavailable in this Windows environment")

    usage = model_disk_usage(root)
    clean = clean_models(root, min_age_hours=0)

    assert usage["totals"]["file_count"] == 0
    assert usage["junctions_skipped"] == ["linked"]
    assert clean["candidates"] == []
    assert clean["junctions_skipped"] == ["linked"]
    assert secret.exists()


def test_cli_disk_usage_and_clean_json(tmp_path: Path, capsys):
    root = tmp_path / "models"
    root.mkdir()
    (root / "one.gguf").write_bytes(b"one")

    assert main(["model_disk_usage", str(root), "--json"]) == 0
    assert '"totals"' in capsys.readouterr().out
    assert main(["models_clean", str(root), "--json"]) == 0
    assert '"candidates"' in capsys.readouterr().out


class _FakeImage:
    def __init__(self, color: tuple[int, int, int]):
        from PIL import Image

        self._image = Image.new("RGB", (64, 64))
        self._image.putdata([
            (
                (color[0] + x * 7 + y * 3) % 256,
                (color[1] + x * 5 + y * 11) % 256,
                (color[2] + x * 13 + y * 2) % 256,
            )
            for y in range(64)
            for x in range(64)
        ])

    def convert(self, mode: str):
        return self._image.convert(mode)


class _FakeEngine:
    def __init__(self):
        self.loaded = None
        self.unloaded = False
        self.requests = []

    def load(self, path: str):
        self.loaded = path
        return SimpleNamespace(to_dict=lambda: {"path": path})

    def generate(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            image=_FakeImage((request.seed % 255, 80, 140)),
            seed=request.seed,
            elapsed_seconds=0.125,
            metadata={"scheduler": request.scheduler or "DPMSolverMultistepScheduler", "safety_flagged": False},
        )

    def unload(self):
        self.unloaded = True


def _fake_asset_gate(monkeypatch):
    import diffusion

    monkeypatch.setattr(diffusion, "verify_asset_directory", lambda *_args, **_kwargs: {
        "valid": True,
        "integrity_scope": "fixture",
    })


def test_sd15_prompt_batch_is_bounded_deterministic_and_writes_report(tmp_path: Path, monkeypatch):
    _fake_asset_gate(monkeypatch)
    from diffusion import get_preset

    engines = []

    def factory():
        engine = _FakeEngine()
        engines.append(engine)
        return engine

    report = run_prompt_batch(
        asset_id="sd15_90s_retrovers_v1",
        model_path=tmp_path / "asset",
        output_dir=tmp_path / "out",
        preset=get_preset("sd15_retrovers_space_courier_v1"),
        prompts=["a red observatory", "a blue observatory"],
        seeds=[11, 12],
        steps=2,
        engine_factory=factory,
    )

    assert report["tool"] == "sd15_prompt_batch"
    assert report["valid"] is True
    assert report["automatic_gate"]["passed"] is True
    assert report["automatic_gate"]["outputs"] == 4
    assert Path(report["contact_sheet"]).is_file()
    assert Path(report["report_path"]).is_file()
    assert len(engines) == 1 and engines[0].unloaded is True
    assert [request.seed for request in engines[0].requests] == [11, 12, 11, 12]


def test_sd15_sampler_matrix_preserves_scheduler_and_rejects_oversized_matrix(tmp_path: Path, monkeypatch):
    _fake_asset_gate(monkeypatch)
    from diffusion import get_preset

    engines = []

    def factory():
        engine = _FakeEngine()
        engines.append(engine)
        return engine

    report = run_sampler_matrix(
        asset_id="sd15_90s_retrovers_v1",
        model_path=tmp_path / "asset",
        output_dir=tmp_path / "out",
        preset=get_preset("sd15_retrovers_space_courier_v1"),
        prompt="a test prompt",
        schedulers=["EulerDiscreteScheduler", "DDIMScheduler"],
        steps_list=[2, 3],
        seed=99,
        engine_factory=factory,
    )

    assert report["tool"] == "sd15_sampler_matrix"
    assert [request.scheduler for request in engines[0].requests] == [
        "EulerDiscreteScheduler", "EulerDiscreteScheduler", "DDIMScheduler", "DDIMScheduler",
    ]
    assert [request.steps for request in engines[0].requests] == [2, 3, 2, 3]

    import pytest
    with pytest.raises(ValueError, match="64 outputs"):
        run_sampler_matrix(
            asset_id="sd15_90s_retrovers_v1",
            model_path=tmp_path / "asset",
            output_dir=tmp_path / "too-many",
            preset=get_preset("sd15_retrovers_space_courier_v1"),
            prompt="a test prompt",
            schedulers=["EulerDiscreteScheduler"] * 65,
            steps_list=[2],
            seed=99,
            engine_factory=factory,
        )


def test_sd15_batch_asset_failure_writes_fail_closed_report(tmp_path: Path, monkeypatch):
    import diffusion
    from diffusion import get_preset

    monkeypatch.setattr(diffusion, "verify_asset_directory", lambda *_args, **_kwargs: {
        "valid": False,
        "errors": ["fixture mismatch"],
    })
    report = run_prompt_batch(
        asset_id="sd15_90s_retrovers_v1",
        model_path=tmp_path / "invalid-asset",
        output_dir=tmp_path / "out",
        preset=get_preset("sd15_retrovers_space_courier_v1"),
        prompts=["a test prompt"],
        seeds=[1],
        steps=2,
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("engine must not start")),
    )

    assert report["valid"] is False
    assert report["status"] == "asset_invalid"
    assert Path(report["report_path"]).is_file()


def _write_model_tree(root: Path, assets: dict[str, bytes]) -> None:
    for asset_id, content in assets.items():
        target = root / asset_id / "model.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def test_sync_inventory_is_stable_and_never_exposes_absolute_paths(tmp_path: Path):
    root = tmp_path / "models"
    _write_model_tree(root, {"z-model": b"z", "a-model": b"a"})
    sd = root / "sd-asset"
    sd.mkdir()
    (sd / ".qlh-sd-asset.json").write_text("{}", encoding="utf-8")
    (sd / "weights.safetensors").write_bytes(b"weights")

    report = build_inventory(root)
    serialized = str(report)

    assert report["valid"] is True
    assert report["read_only"] is True
    assert report["hash_mode"] == "structure"
    assert [item["asset_id"] for item in report["assets"]] == ["a-model", "sd-asset", "z-model"]
    assert report["assets"][1]["kind"] == "diffusion"
    assert report["assets"][1]["file_count"] == 2
    assert all(item["content_digest"] is None for item in report["assets"])
    assert str(tmp_path) not in serialized
    assert "models" not in report


def test_sync_structure_mode_is_fast_and_full_hash_detects_same_size_changes(tmp_path: Path):
    local_root = tmp_path / "local"
    peer_root = tmp_path / "peer"
    _write_model_tree(local_root, {"same-model": b"AAAA"})
    _write_model_tree(peer_root, {"same-model": b"BBBB"})

    fast = compare_inventories(build_inventory(local_root), build_inventory(peer_root))
    complete = compare_inventories(
        build_inventory(local_root, full_hash=True),
        build_inventory(peer_root, full_hash=True),
    )

    assert fast["valid"] is True and fast["in_sync"] is True
    assert complete["valid"] is True and complete["in_sync"] is False
    assert complete["mismatched"] == [{"asset_id": "same-model", "changed_fields": ["content_digest"]}]


def test_sync_compare_separates_missing_extra_and_mismatched_assets(tmp_path: Path):
    local_root = tmp_path / "local"
    peer_root = tmp_path / "peer"
    _write_model_tree(local_root, {"matched": b"1", "missing": b"2", "changed": b"333"})
    _write_model_tree(peer_root, {"matched": b"1", "extra": b"2", "changed": b"4444"})

    report = compare_inventories(build_inventory(local_root), build_inventory(peer_root))

    assert report["valid"] is True
    assert report["in_sync"] is False
    assert report["missing_on_peer"] == ["missing"]
    assert report["extra_on_peer"] == ["extra"]
    assert report["matched_count"] == 1
    assert report["mismatched"] == [{
        "asset_id": "changed",
        "changed_fields": ["logical_size_bytes", "structure_digest"],
    }]


def test_sync_inventory_validation_rejects_traversal_duplicates_and_hash_mode_mixing(tmp_path: Path):
    root = tmp_path / "models"
    _write_model_tree(root, {"model": b"weights"})
    inventory = build_inventory(root)
    bad = {**inventory, "assets": [inventory["assets"][0], inventory["assets"][0]]}
    bad["assets"][0] = {**bad["assets"][0], "asset_id": "../outside"}
    bad["assets"][1] = {**bad["assets"][1], "asset_id": "../outside"}

    normalized, errors = validate_inventory(bad)
    mixed = compare_inventories(inventory, build_inventory(root, full_hash=True))

    assert normalized is None
    assert any("relative path component" in error for error in errors)
    assert any("duplicate asset_id" in error for error in errors)
    assert mixed["valid"] is False
    assert mixed["errors"] == ["local and peer hash_mode must match"]


def test_sync_inventory_does_not_follow_nested_symlinks(tmp_path: Path):
    import pytest

    root = tmp_path / "models"
    model = root / "model"
    outside = tmp_path / "outside"
    model.mkdir(parents=True)
    outside.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (outside / "secret.safetensors").write_bytes(b"secret")
    link = model / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this Windows environment")

    report = build_inventory(root, full_hash=True)

    assert report["valid"] is True
    assert report["assets"][0]["file_count"] == 1
    assert report["assets"][0]["logical_size_bytes"] == 2
    assert report["warnings"] == ["junction/reparse point not traversed: model/linked"]
    assert "secret" not in str(report)


def test_sync_cli_exit_codes_and_atomic_inventory_output(tmp_path: Path, capsys):
    local_root = tmp_path / "local"
    peer_root = tmp_path / "peer"
    output = tmp_path / "reports" / "inventory.json"
    _write_model_tree(local_root, {"model": b"same"})
    _write_model_tree(peer_root, {"model": b"same"})

    assert main(["models_sync_status", "inventory", str(local_root), "--output", str(output), "--json"]) == 0
    capsys.readouterr()
    assert output.is_file()
    assert not list(output.parent.glob(".*.tmp"))
    assert main([
        "models_sync_status", "compare",
        "--local-root", str(local_root), "--peer-root", str(peer_root), "--json",
    ]) == 0
    capsys.readouterr()

    (peer_root / "model" / "model.safetensors").write_bytes(b"different-size")
    assert main([
        "models_sync_status", "compare",
        "--local-root", str(local_root), "--peer-root", str(peer_root), "--json",
    ]) == 1
    difference = capsys.readouterr().out
    assert '"in_sync": false' in difference

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert main([
        "models_sync_status", "compare",
        "--local-inventory", str(malformed), "--peer-inventory", str(output), "--json",
    ]) == 2
    invalid = capsys.readouterr().out
    assert '"valid": false' in invalid
    assert str(tmp_path) not in invalid

    assert main([
        "models_sync_status", "inventory", str(local_root),
        "--output", str(local_root / "inventory.json"), "--json",
    ]) == 2
    refused = capsys.readouterr().out
    assert "outside model roots" in refused


def test_llm_smoke_prompts_have_fixed_json_and_context_gates():
    prompts = fixed_prompts()
    assert [item["id"] for item in prompts] == ["zh_basic", "en_basic", "json_object", "context_marker"]
    assert len(prompts[-1]["text"]) > 500
    assert validate_output("json_object", '{"ok": true, "kind": "smoke"}') == {
        "non_empty": True,
        "language_valid": None,
        "json_valid": True,
        "context_marker_found": None,
        "passed": True,
    }
    assert validate_output("context_marker", "QLH-SMOKE-MARKER-7F3A") ["passed"] is True
    assert validate_output("json_object", "not json")["passed"] is False
    assert validate_output("zh_basic", "这是中文回答")["passed"] is True
    assert validate_output("zh_basic", "English only")["passed"] is False
    assert validate_output("en_basic", "An English answer.")["passed"] is True
    assert validate_output("json_object", '{"ok":true,"kind":"other"}')["passed"] is False


def test_llm_smoke_matrix_is_bounded_and_isolates_unit_failures(monkeypatch):
    units = [
        {"model_id": "ok", "name": "OK", "format": "gguf", "engine": "llama_cpp", "path": "C:/models/ok.gguf", "available": True, "recommended_vram_gb": 1.0},
        {"model_id": "missing", "name": "Missing", "format": "gguf", "engine": "llama_cpp", "path": "C:/models/missing.gguf", "available": False, "recommended_vram_gb": 1.0},
        {"model_id": "bad", "name": "Bad", "format": "safetensors", "engine": "pytorch", "path": "C:/models/bad", "available": True, "recommended_vram_gb": 1.0},
    ]
    for unit in units:
        unit["asset_size_bytes"] = 1
    monkeypatch.setattr("scripts.model_tools.llm_smoke_matrix.discover_units", lambda *_args, **_kwargs: units)

    def runner(unit, prompts, **_kwargs):
        if unit["model_id"] == "bad":
            return {"status": "failed", "jobs": [], "error": {"code": "load_failed", "message": "fixture failure"}}
        return {
            "status": "passed",
            "load_ms": 3,
            "jobs": [{"prompt_id": prompt["id"], "status": "passed"} for prompt in prompts],
            "error": None,
        }

    monkeypatch.setattr("scripts.model_tools.llm_smoke_matrix._resource_rejection", lambda *_args, **_kwargs: None)
    report = run_smoke_matrix(max_models=3, max_new_tokens=4, timeout_seconds=2, worker_runner=runner, allow_cpu=True)

    assert report["valid"] is True
    assert report["summary"] == {
        "units_discovered": 3,
        "units_total": 3,
        "selection_truncated": False,
        "units_executed": 2,
        "units_passed": 1,
        "units_failed": 1,
        "units_skipped": 1,
        "jobs_passed": 4,
        "jobs_failed": 0,
        "execution_gate_passed": False,
        "coverage_complete": False,
        "gate_passed": False,
    }
    assert report["models"][1]["status"] == "skipped"
    assert report["models"][2]["error"]["code"] == "load_failed"


def test_llm_smoke_worker_isolates_prompt_failure_without_output_text():
    class FakeManager:
        def load_model(self, **_kwargs):
            return None

        def chat(self, messages, **_kwargs):
            if messages[-1]["content"].startswith("fail"):
                raise RuntimeError("prompt fixture failure")
            return {"content": '{"ok": true, "kind": "smoke"}'}

        def unload_model(self):
            return None

    request = {
        "schema_version": 1,
        "operation": "llm_smoke_worker",
        "model_id": "fixture",
        "format": "gguf",
        "engine": "llama_cpp",
        "model_path": str(Path.cwd() / "tests"),
        "prompts": [
            {"id": "json_object", "text": "ok"},
            {"id": "en_basic", "text": "fail"},
        ],
        "max_new_tokens": 4,
    }
    report = execute_request(request, manager_factory=FakeManager)

    assert report["status"] == "failed"
    assert [job["status"] for job in report["jobs"]] == ["passed", "failed"]
    assert all("content" not in job for job in report["jobs"])
    assert report["jobs"][1]["error"]["code"] == "generation_failed"


def test_llm_smoke_worker_redacts_model_path_from_errors(tmp_path: Path):
    model = tmp_path / "private-model.gguf"
    model.write_bytes(b"fixture")

    class FailingManager:
        def load_model(self, **_kwargs):
            raise RuntimeError(f"cannot load {model}")

    report = execute_request({
        "schema_version": 1,
        "operation": "llm_smoke_worker",
        "model_id": "fixture",
        "format": "gguf",
        "engine": "llama_cpp",
        "model_path": str(model),
        "prompts": [{"id": "zh_basic", "text": "test"}],
    }, manager_factory=FailingManager)

    assert report["status"] == "failed"
    assert report["error"]["message"] == "cannot load <model-path>"
    assert str(tmp_path) not in str(report)

    cache_error = execute_request({
        "schema_version": 1,
        "operation": "llm_smoke_worker",
        "model_id": "fixture",
        "format": "gguf",
        "engine": "llama_cpp",
        "model_path": str(model),
        "prompts": [{"id": "zh_basic", "text": "test"}],
    }, manager_factory=lambda: (_ for _ in ()).throw(RuntimeError("cache failed at C:\\Users\\private\\cache.bin")))
    assert "C:\\" not in cache_error["error"]["message"]
    assert "<path>" in cache_error["error"]["message"]


def test_llm_smoke_worker_rejects_invalid_engine_and_prompt_contract(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    base = {
        "schema_version": 1,
        "operation": "llm_smoke_worker",
        "model_id": "fixture",
        "format": "gguf",
        "engine": "pytorch",
        "model_path": str(model),
        "prompts": [{"id": "zh_basic", "text": "test"}],
        "max_new_tokens": 4,
    }

    mismatch = execute_request(base, manager_factory=lambda: None)
    bad_prompt = execute_request({**base, "engine": "llama_cpp", "prompts": [{"id": "unknown", "text": "test"}]}, manager_factory=lambda: None)

    assert mismatch["status"] == "invalid_request"
    assert mismatch["error"]["message"] == "model format and engine do not match"
    assert bad_prompt["status"] == "invalid_request"
    assert bad_prompt["error"]["message"] == "invalid prompt item"

    duplicate = execute_request({
        **base,
        "engine": "llama_cpp",
        "prompts": [{"id": "zh_basic", "text": "one"}, {"id": "zh_basic", "text": "two"}],
    }, manager_factory=lambda: None)
    assert duplicate["error"]["message"] == "duplicate prompt_id"


def test_llm_smoke_cli_rejects_unknown_model_id(capsys):
    assert main(["llm_smoke_matrix", "--model-id", "not-registered", "--json"]) == 2
    report = capsys.readouterr().out
    assert '"valid": false' in report
    assert "unknown model_id" in report


def test_llm_smoke_cli_exit_codes_for_partial_pass_and_failed_gate(monkeypatch, capsys):
    partial = {
        "tool": "llm_smoke_matrix",
        "valid": True,
        "models": [],
        "summary": {"gate_passed": True, "coverage_complete": False},
        "errors": [],
    }
    monkeypatch.setattr("scripts.model_tools.cli.run_smoke_matrix", lambda **_kwargs: partial)
    assert main(["llm_smoke_matrix"]) == 0
    assert "PASS (partial coverage)" in capsys.readouterr().out

    failed = {**partial, "summary": {"gate_passed": False, "coverage_complete": True}}
    monkeypatch.setattr("scripts.model_tools.cli.run_smoke_matrix", lambda **_kwargs: failed)
    assert main(["llm_smoke_matrix", "--json"]) == 1
    assert '"gate_passed": false' in capsys.readouterr().out


def test_llm_smoke_require_complete_fails_partial_coverage(monkeypatch):
    units = [
        {"model_id": "ok", "name": "OK", "format": "gguf", "engine": "llama_cpp", "path": "C:/ok.gguf", "available": True, "recommended_vram_gb": 1.0, "asset_size_bytes": 1},
        {"model_id": "missing", "name": "Missing", "format": "gguf", "engine": "llama_cpp", "path": "C:/missing.gguf", "available": False, "recommended_vram_gb": 1.0, "asset_size_bytes": 0},
    ]
    monkeypatch.setattr("scripts.model_tools.llm_smoke_matrix.discover_units", lambda *_args, **_kwargs: units)
    monkeypatch.setattr("scripts.model_tools.llm_smoke_matrix._resource_rejection", lambda *_args, **_kwargs: None)

    def runner(_unit, prompts, **_kwargs):
        return {"status": "passed", "jobs": [{"status": "passed"} for _ in prompts], "error": None}

    report = run_smoke_matrix(require_complete=True, worker_runner=runner)

    assert report["summary"]["execution_gate_passed"] is True
    assert report["summary"]["coverage_complete"] is False
    assert report["summary"]["gate_passed"] is False
    assert report["errors"] == ["coverage is incomplete"]


def test_llm_smoke_cli_refuses_output_inside_model_root(capsys):
    target = Path.cwd() / "models" / "llm-smoke.json"
    assert main(["llm_smoke_matrix", "--output", str(target), "--json"]) == 2
    assert "outside model roots" in capsys.readouterr().out
    assert not target.exists()


def test_llm_smoke_worker_uses_ephemeral_offline_cache(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs["env"])
        assert Path(kwargs["env"]["HF_HOME"]).is_dir()
        return SimpleNamespace(stdout='{"operation":"llm_smoke_worker","status":"passed","jobs":[]}\n')

    monkeypatch.setattr("scripts.model_tools.llm_smoke_matrix.subprocess.run", fake_run)
    from scripts.model_tools.llm_smoke_matrix import _run_worker

    result = _run_worker(
        {"model_id": "fixture", "format": "gguf", "engine": "llama_cpp", "path": "C:/fixture.gguf"},
        fixed_prompts(),
        quant="int4",
        max_new_tokens=4,
        timeout_seconds=2,
        allow_cpu=False,
    )

    assert result["status"] == "passed"
    assert observed["HF_HUB_OFFLINE"] == "1"
    assert observed["TRANSFORMERS_OFFLINE"] == "1"
    assert not Path(observed["HF_HOME"]).exists()


def test_gguf_convert_f16_dry_run_is_read_only_and_path_redacted(tmp_path: Path):
    source = _write_hf_fixture(tmp_path / "private" / "source-model")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    (tmp_path / "publish").mkdir()
    target = tmp_path / "publish" / "fixture-F16.gguf"
    before = sorted((item.relative_to(tmp_path), item.stat().st_size) for item in tmp_path.rglob("*") if item.is_file())

    report = plan_conversion(source=source, target=target, outtype="F16", converter=converter)

    after = sorted((item.relative_to(tmp_path), item.stat().st_size) for item in tmp_path.rglob("*") if item.is_file())
    assert report["valid"] is True
    assert report["request_valid"] is True
    assert report["read_only"] is True
    assert report["writes_performed"] is False
    assert report["apply_supported"] is True
    assert report["output_type"] == "f16"
    assert report["toolchain"]["architecture_supported"] is True
    assert report["toolchain"]["quantizer"]["required"] is False
    assert report["plan"]["stages"] == ["convert_hf_to_gguf", "inspect_gguf", "verify_gguf", "atomic_publish"]
    assert report["plan"]["commands"][0] == ["python", "<converter>", "<source>", "--outtype", "f16", "--outfile", "<target>"]
    assert str(tmp_path) not in str(report)
    assert not target.exists()
    assert before == after


def test_gguf_convert_quantized_plan_requires_quantizer(tmp_path: Path):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    target = tmp_path / "fixture-Q4_K_M.gguf"

    blocked = plan_conversion(
        source=source,
        target=target,
        outtype="Q4_K_M",
        converter=converter,
        quantizer=tmp_path / "missing-quantizer",
    )
    quantizer = tmp_path / "llama-quantize.exe"
    quantizer.write_bytes(b"fixture")
    ready = plan_conversion(source=source, target=target, outtype="q4-k-m", converter=converter, quantizer=quantizer)

    assert blocked["valid"] is False
    assert [error["code"] for error in blocked["errors"]] == ["quantizer_missing"]
    assert ready["valid"] is True
    assert ready["output_type"] == "Q4_K_M"
    assert ready["plan"]["stages"][:2] == ["convert_hf_to_gguf", "llama_quantize"]
    assert ready["plan"]["commands"][0][4] == "f16"
    assert ready["plan"]["commands"][1][-1] == "Q4_K_M"


def test_gguf_convert_fails_closed_for_target_source_and_architecture(tmp_path: Path):
    source = _write_hf_fixture(tmp_path / "source", architecture="UnsupportedFixtureModel")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    existing = tmp_path / "existing.gguf"
    existing.write_bytes(b"do-not-overwrite")
    orphan_sidecar_target = tmp_path / "orphan-sidecar.gguf"
    orphan_sidecar = orphan_sidecar_target.with_name(orphan_sidecar_target.name + ".sha256")
    orphan_sidecar.write_text("0" * 64 + "  orphan-sidecar.gguf\n", encoding="ascii")

    existing_report = plan_conversion(source=source, target=existing, outtype="f16", converter=converter)
    nested_report = plan_conversion(source=source, target=source / "nested.gguf", outtype="f16", converter=converter)
    architecture_report = plan_conversion(source=source, target=tmp_path / "out.gguf", outtype="f16", converter=converter)
    sidecar_report = plan_conversion(source=source, target=orphan_sidecar_target, outtype="f16", converter=converter)

    assert "target_exists" in {error["code"] for error in existing_report["errors"]}
    assert existing.read_bytes() == b"do-not-overwrite"
    assert "target_inside_source" in {error["code"] for error in nested_report["errors"]}
    assert "unsupported_architecture" in {error["code"] for error in architecture_report["errors"]}
    assert "target_sidecar_exists" in {error["code"] for error in sidecar_report["errors"]}
    assert orphan_sidecar.is_file()


def test_gguf_convert_rejects_missing_index_shards(tmp_path: Path):
    source = _write_hf_fixture(tmp_path / "source")
    (source / "model.safetensors.index.json").write_text(
        '{"weight_map":{"fixture.weight":"missing-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    converter = _write_converter_fixture(tmp_path / "toolchain")

    report = plan_conversion(source=source, target=tmp_path / "out.gguf", outtype="f16", converter=converter)

    assert report["source"]["weight_index_valid"] is False
    assert "missing_weight_shard" in {error["code"] for error in report["errors"]}


def test_gguf_convert_space_gate_and_invalid_source_contract(tmp_path: Path, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    monkeypatch.setattr(
        "scripts.model_tools.gguf_convert.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=999, free=1),
    )

    report = plan_conversion(source=source, target=tmp_path / "out.gguf", outtype="f16", converter=converter)

    assert "insufficient_space" in {error["code"] for error in report["errors"]}
    with pytest.raises(GGUFConvertError, match="exactly one"):
        plan_conversion(source=source, model_id="qwen-1_8b")
    with pytest.raises(GGUFConvertError, match="unsupported output type"):
        plan_conversion(source=source, outtype="not-a-quant")


def test_gguf_convert_cli_exit_codes_and_report_output(tmp_path: Path, capsys):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    output = tmp_path / "reports" / "plan.json"
    target = tmp_path / "out.gguf"

    assert main([
        "gguf_convert", "--source", str(source), "--target", str(target),
        "--outtype", "f16", "--converter", str(converter), "--output", str(output), "--json",
    ]) == 0
    stdout = capsys.readouterr().out
    assert '"writes_performed": false' in stdout
    assert output.is_file()
    assert not target.exists()
    assert str(tmp_path) not in output.read_text(encoding="utf-8")

    assert main([
        "gguf_convert", "--source", str(source), "--target", str(target),
        "--outtype", "Q4_K_M", "--converter", str(converter),
        "--quantizer", str(tmp_path / "missing"), "--json",
    ]) == 1
    assert '"code": "quantizer_missing"' in capsys.readouterr().out

    assert main(["gguf_convert", "--model-id", "not-registered", "--json"]) == 2
    assert '"request_valid": false' in capsys.readouterr().out


def test_gguf_convert_execute_f16_publishes_valid_artifact(tmp_path: Path, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_copy_converter(tmp_path / "toolchain")
    fixture = tmp_path / "fixture.gguf"
    digest = _write_gguf(fixture)
    publish = tmp_path / "publish"
    publish.mkdir()
    target = publish / "converted-F16.gguf"
    monkeypatch.setenv("QLH_FAKE_GGUF", str(fixture))

    report = execute_conversion(
        source=source,
        target=target,
        outtype="f16",
        converter=converter,
        confirmation="CONVERT",
        timeout_seconds=30,
    )

    assert report["valid"] is True
    assert report["preflight_valid"] is True
    assert report["writes_performed"] is True
    assert report["execution"]["published"] is True
    assert [stage["stage"] for stage in report["execution"]["stages"]] == [
        "convert_hf_to_gguf", "inspect_verify", "atomic_publish",
    ]
    assert report["execution"]["artifact"]["sha256"] == digest
    assert report["execution"]["artifact"]["sha256_sidecar"] is True
    assert target.with_name(target.name + ".sha256").is_file()
    assert verify_gguf(target, full_hash=True)["valid"] is True
    assert not list(publish.glob(".qlh-gguf-staging-*"))
    assert str(tmp_path) not in str(report)


def test_gguf_convert_execute_two_stage_quantization(tmp_path: Path, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_copy_converter(tmp_path / "toolchain")
    quantizer = _write_copy_quantizer(tmp_path / "llama-quantize.py")
    fixture = tmp_path / "fixture.gguf"
    _write_gguf(fixture)
    target = tmp_path / "converted-Q4_K_M.gguf"
    monkeypatch.setenv("QLH_FAKE_GGUF", str(fixture))

    report = execute_conversion(
        source=source,
        target=target,
        outtype="Q4_K_M",
        converter=converter,
        quantizer=quantizer,
        confirmation="CONVERT",
        timeout_seconds=30,
    )

    assert report["valid"] is True
    assert [stage["stage"] for stage in report["execution"]["stages"]] == [
        "convert_hf_to_gguf", "inspect_intermediate", "llama_quantize", "inspect_verify", "atomic_publish",
    ]
    assert target.is_file()
    assert target.with_name(target.name + ".sha256").is_file()
    assert not list(tmp_path.glob(".qlh-gguf-staging-*"))


def test_gguf_convert_execute_failure_cleans_staging(tmp_path: Path, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_copy_converter(tmp_path / "toolchain", fail_code=7)
    target = tmp_path / "failed-F16.gguf"
    monkeypatch.setenv("QLH_FAKE_GGUF", str(tmp_path / "unused.gguf"))

    report = execute_conversion(
        source=source,
        target=target,
        outtype="f16",
        converter=converter,
        confirmation="CONVERT",
        timeout_seconds=30,
    )

    assert report["valid"] is False
    assert report["writes_performed"] is True
    assert report["execution"]["published"] is False
    assert report["execution"]["error"]["code"] == "stage_failed"
    assert not target.exists()
    assert not list(tmp_path.glob(".qlh-gguf-staging-*"))


def test_gguf_convert_execute_never_overwrites_racing_target(tmp_path: Path, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    target = tmp_path / "racing-F16.gguf"
    converter = _write_copy_converter(tmp_path / "toolchain", race_target=target)
    fixture = tmp_path / "fixture.gguf"
    _write_gguf(fixture)
    monkeypatch.setenv("QLH_FAKE_GGUF", str(fixture))

    report = execute_conversion(
        source=source,
        target=target,
        outtype="f16",
        converter=converter,
        confirmation="CONVERT",
        timeout_seconds=30,
    )

    assert report["valid"] is False
    assert report["execution"]["error"]["code"] == "target_exists"
    assert target.read_bytes() == b"race-owner"
    assert not list(tmp_path.glob(".qlh-gguf-staging-*"))


def test_gguf_convert_execute_requires_confirmation_and_target(tmp_path: Path):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")

    with pytest.raises(GGUFConvertError, match="confirm CONVERT"):
        execute_conversion(source=source, target=tmp_path / "out.gguf", outtype="f16", converter=converter)
    with pytest.raises(GGUFConvertError, match="--target is required"):
        execute_conversion(source=source, outtype="f16", converter=converter, confirmation="CONVERT")


def test_gguf_convert_cli_apply_requires_exact_confirmation(tmp_path: Path, capsys, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    target = tmp_path / "out.gguf"

    assert main([
        "gguf_convert", "--source", str(source), "--target", str(target),
        "--outtype", "f16", "--converter", str(converter), "--apply", "--json",
    ]) == 2
    assert "confirm CONVERT" in capsys.readouterr().out
    assert not target.exists()

    fixture = tmp_path / "fixture.gguf"
    _write_gguf(fixture)
    converter = _write_copy_converter(tmp_path / "real-toolchain")
    monkeypatch.setenv("QLH_FAKE_GGUF", str(fixture))
    assert main([
        "gguf_convert", "--source", str(source), "--target", str(target),
        "--outtype", "f16", "--converter", str(converter),
        "--apply", "--confirm", "CONVERT", "--timeout-seconds", "30", "--json",
    ]) == 0
    output = capsys.readouterr().out
    assert '"published": true' in output
    assert target.is_file()
    assert target.with_name(target.name + ".sha256").is_file()


def test_gguf_convert_execute_timeout_cleans_staging(tmp_path: Path):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    converter.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    target = tmp_path / "timeout.gguf"

    report = execute_conversion(
        source=source,
        target=target,
        outtype="f16",
        converter=converter,
        confirmation="CONVERT",
        timeout_seconds=1,
    )

    assert report["valid"] is False
    assert report["execution"]["error"]["code"] == "stage_timeout"
    assert not target.exists()
    assert not target.with_name(target.name + ".sha256").exists()
    assert not list(tmp_path.glob(".qlh-gguf-staging-*"))


def test_gguf_convert_execute_cancel_cleans_staging(tmp_path: Path, monkeypatch):
    source = _write_hf_fixture(tmp_path / "source")
    converter = _write_converter_fixture(tmp_path / "toolchain")
    target = tmp_path / "cancelled.gguf"

    def cancel_stage(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("scripts.model_tools.gguf_convert._run_stage", cancel_stage)

    report = execute_conversion(
        source=source,
        target=target,
        outtype="f16",
        converter=converter,
        confirmation="CONVERT",
        timeout_seconds=30,
    )

    assert report["valid"] is False
    assert report["execution"]["error"]["code"] == "cancelled"
    assert not target.exists()
    assert not target.with_name(target.name + ".sha256").exists()
    assert not list(tmp_path.glob(".qlh-gguf-staging-*"))


# ================================================================
# P8: 模型导入向导（2026-08-16）
# ================================================================

class TestImportModelWizard:
    def test_resolve_local_directory(self, tmp_path):
        from scripts.model_tools.import_model import resolve_target
        assert resolve_target(str(tmp_path), None) == tmp_path.absolute()

    def test_resolve_repo_name_to_models_root(self, tmp_path, monkeypatch):
        from scripts.model_tools.import_model import resolve_target
        monkeypatch.chdir(tmp_path)
        assert resolve_target("SomeOrg/some-model", None).name == "some-model"

    def test_verify_files_calculates_sha_and_bytes(self, tmp_path):
        from scripts.model_tools.import_model import verify_files
        import hashlib
        target = tmp_path / "model"
        target.mkdir()
        payload = b"fake-weights"
        (target / "model.safetensors").write_bytes(payload)
        summary = verify_files([target / "model.safetensors"], None)
        assert summary["file_count"] == 1
        assert summary["total_bytes"] == len(payload)
        assert summary["sha256"] == hashlib.sha256(payload).hexdigest()

    def test_verify_files_strict_sha_mismatch_fails_closed(self, tmp_path):
        from scripts.model_tools.import_model import verify_files
        target = tmp_path / "model"
        target.mkdir()
        (target / "model.gguf").write_bytes(b"x")
        import pytest as _pytest
        with _pytest.raises(ValueError, match="SHA-256 不匹配"):
            verify_files([target / "model.gguf"], "0" * 64)

    def test_register_model_persists_to_sqlite(self, tmp_path, monkeypatch):
        from scripts.model_tools.import_model import register_model
        from local_store import get_local_experimental_models, delete_local_experimental_model
        target = tmp_path / "imported-model"
        target.mkdir()
        (target / "model.gguf").write_bytes(b"abc")
        summary = {"file_count": 1, "total_bytes": 3, "sha256": "a" * 64}
        assert register_model("imported-model", target, summary) is True
        try:
            models = get_local_experimental_models()
            assert any(m.get("model_id") == "imported-model" for m in models)
        finally:
            delete_local_experimental_model("imported-model")

    def test_cli_wizard_imports_local_dir_without_download(self, tmp_path, monkeypatch, capsys):
        import subprocess, sys
        target = tmp_path / "cli-model"
        target.mkdir()
        (target / "model.gguf").write_bytes(b"fake")
        result = subprocess.run(
            [sys.executable, "scripts/model_tools.py", "import-model", str(target),
             "--json", "--register"],
            capture_output=True, text=True, encoding="utf-8", cwd=".",
        )
        assert result.returncode == 0
        import json as _json
        payload = _json.loads(result.stdout[result.stdout.find("{"):])
        assert payload["registered"] is True
        assert payload["model_id"] == "cli-model"
        from local_store import delete_local_experimental_model
        delete_local_experimental_model("cli-model")

    def test_empty_directory_fails_closed(self, tmp_path):
        from scripts.model_tools.import_model import main
        target = tmp_path / "empty"
        target.mkdir()
        assert main([str(target), "--skip-download"]) == 2

    def test_safetensors_import_writes_manifest_and_type(self, tmp_path):
        from scripts.model_tools.import_model import main
        target = tmp_path / "hf-model"
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")
        assert main([str(target), "--skip-download"]) == 0
        manifest = target / "model.manifest.json"
        assert manifest.is_file()
        payload = manifest.read_text(encoding="utf-8")
        assert '"model_type": "safetensors"' in payload
        assert '"path": "config.json"' in payload

    def test_explicit_missing_gguf_fails_closed(self, tmp_path):
        from scripts.model_tools.import_model import main
        target = tmp_path / "hf-model"
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")
        assert main([str(target), "--skip-download", "--gguf-path", str(tmp_path / "missing.gguf")]) == 2

    def test_proxy_resolution_prefers_qlh_setting(self):
        from proxy_config import resolve_http_proxy
        assert resolve_http_proxy(env={"QLH_HTTP_PROXY": "http://127.0.0.1:7897", "HTTP_PROXY": "http://bad:1"}) == "http://127.0.0.1:7897"
        assert resolve_http_proxy("https://proxy.example:8443", env={"QLH_HTTP_PROXY": "http://bad:1"}) == "https://proxy.example:8443"

    def test_remote_staging_is_removed_when_download_fails(self, tmp_path, monkeypatch):
        import scripts.model_tools.import_model as wizard
        target = tmp_path / "remote-model"

        def broken_download(_source, staging, **_kwargs):
            (staging / "partial.gguf").write_bytes(b"partial")
            raise RuntimeError("network failed")

        monkeypatch.setattr(wizard, "download_model", broken_download)
        assert wizard.main(["Org/remote-model", "--target", str(target)]) == 2
        assert not target.exists()
        assert not list(tmp_path.glob(".remote-model.qlh-import-*"))
