import json
import struct
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.artifacts import DiffusionArtifactInspector


def _write_safetensors_header(path: Path, tensors: dict) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def test_diffusers_directory_is_recognized_without_loading_weights(tmp_path):
    root = tmp_path / "sd15"
    (root / "unet").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "text_encoder").mkdir()
    (root / "tokenizer").mkdir()
    (root / "scheduler").mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8"
    )
    _write_safetensors_header(
        root / "unet" / "diffusion_pytorch_model.safetensors",
        {"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
    )

    artifact = DiffusionArtifactInspector().inspect(str(root), compute_hash=True)

    assert artifact.artifact_kind == "sd15_pipeline"
    assert artifact.loadable is True
    assert artifact.precision == "fp16"
    assert len(artifact.sha256) == 64
    assert artifact.missing_components == []


def test_inpaint_directory_has_a_dedicated_non_base_artifact_kind(tmp_path):
    root = tmp_path / "sd15-inpaint"
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        (root / component).mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionInpaintPipeline"}),
        encoding="utf-8",
    )
    _write_safetensors_header(
        root / "unet" / "diffusion_pytorch_model.safetensors",
        {"conv_in.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
    )

    artifact = DiffusionArtifactInspector().inspect(str(root), compute_hash=True)

    assert artifact.artifact_kind == "sd15_inpaint_pipeline"
    assert artifact.loadable is True
    assert artifact.precision == "fp16"


def test_controlnet_single_file_is_not_treated_as_a_full_checkpoint(tmp_path):
    path = tmp_path / "control_v11e_sd15_ip2p_fp16.safetensors"
    _write_safetensors_header(
        path,
        {
            "control_model.input_hint_block.0.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
    )

    artifact = DiffusionArtifactInspector().inspect(str(path))

    assert artifact.artifact_kind == "controlnet"
    assert artifact.loadable is False
    assert any("辅助组件" in warning for warning in artifact.warnings)


def test_unknown_safetensors_fails_closed(tmp_path):
    path = tmp_path / "adapter.safetensors"
    _write_safetensors_header(
        path,
        {"some_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
    )

    artifact = DiffusionArtifactInspector().inspect(str(path))

    assert artifact.artifact_kind == "unknown"
    assert artifact.loadable is False
    assert artifact.warnings


def test_invalid_diffusers_directory_reports_missing_components(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8"
    )

    artifact = DiffusionArtifactInspector().inspect(str(root))

    assert artifact.artifact_kind == "unknown"
    assert artifact.loadable is False
    assert set(artifact.missing_components) == {
        "unet",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
    }


def test_sd15_ip_adapter_directory_requires_adapter_and_image_encoder(tmp_path):
    root = tmp_path / "ip-adapter"
    models = root / "models"
    image_encoder = models / "image_encoder"
    image_encoder.mkdir(parents=True)
    _write_safetensors_header(
        models / "ip-adapter_sd15.safetensors",
        {
            "image_proj.proj.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            "ip_adapter.1.to_k_ip.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
    )
    (image_encoder / "config.json").write_text("{}", encoding="utf-8")
    _write_safetensors_header(
        image_encoder / "model.safetensors",
        {
            "vision_model.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
    )

    artifact = DiffusionArtifactInspector().inspect(str(root), compute_hash=True)

    assert artifact.artifact_kind == "sd15_ip_adapter"
    assert artifact.loadable is True
    assert artifact.precision == "fp16"
    assert len(artifact.sha256) == 64


def test_sd15_ip_adapter_without_image_encoder_fails_closed(tmp_path):
    root = tmp_path / "ip-adapter"
    root.mkdir()
    _write_safetensors_header(
        root / "ip-adapter_sd15.safetensors",
        {
            "image_proj.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            "ip_adapter.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
    )

    artifact = DiffusionArtifactInspector().inspect(str(root))

    assert artifact.artifact_kind == "sd15_ip_adapter"
    assert artifact.loadable is False
    assert "image_encoder" in artifact.missing_components
