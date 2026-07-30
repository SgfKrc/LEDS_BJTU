"""Download a pinned Stable Diffusion 1.5 Diffusers snapshot.

The model is an external asset and is ignored by Git.  Use ``--proxy
http://127.0.0.1:7897`` when the direct Hugging Face route is unavailable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned SD1.5 Diffusers assets")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-dir", default="models/sd15-original-v1")
    parser.add_argument("--proxy", default="", help="HTTP(S) proxy, e.g. http://127.0.0.1:7897")
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "缺少 huggingface_hub，请先安装 packaging/requirements-sd15.txt"
        ) from exc

    local_dir = Path(args.local_dir).expanduser()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        allow_patterns=[
            "model_index.json",
            "scheduler/*.json",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
            "feature_extractor/preprocessor_config.json",
            "README.md",
            "LICENSE*",
        ],
    )
    print(f"SD15 snapshot ready: {local_dir.resolve()}")
    print("下一步: python -m pytest tests/test_diffusion_artifacts.py tests/test_diffusion_presets.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
