"""Download a pinned, verified Stable Diffusion asset outside the installer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import (  # noqa: E402
    DiffusionAssetManager,
    LOCAL_PROXY_FALLBACK,
    get_asset_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a pinned QLH SD 1.5 asset")
    parser.add_argument(
        "--asset-id",
        default="sd15_original_v1",
        choices=(
            "sd15_original_v1",
            "sd15_90s_retrovers_v1",
            "sd15_ip_adapter_v1",
        ),
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that the model card and declared OpenRAIL license were reviewed",
    )
    parser.add_argument(
        "--proxy-fallback",
        default=LOCAL_PROXY_FALLBACK,
        help="Loopback proxy used only after the direct route fails; pass an empty value to disable",
    )
    parser.add_argument(
        "--local-dir",
        default="",
        help="Compatibility check only; catalog assets always use their pinned models/ directory",
    )
    args = parser.parse_args()

    spec = get_asset_spec(args.asset_id)
    target = spec.target_path(ROOT).resolve()
    if args.local_dir and Path(args.local_dir).expanduser().resolve() != target:
        parser.error(f"{args.asset_id} has a fixed target directory: {target}")
    if not args.accept_license:
        parser.error("--accept-license is required")

    manager = DiffusionAssetManager(root=ROOT)
    status = manager.start_download(
        args.asset_id,
        license_accepted=True,
        proxy_fallback=args.proxy_fallback,
    )
    last_line = ""
    while status["state"] not in manager.TERMINAL_STATES:
        line = (
            f"{status['state']}: {status['progress_percent']}% "
            f"({status['present_bytes']}/{status['download_bytes']} bytes)"
        )
        if line != last_line:
            print(line, flush=True)
            last_line = line
        time.sleep(1)
        status = manager.status(args.asset_id)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["state"] != "completed":
        return 1
    print(f"SD asset ready: {target}")
    if spec.artifact_kind == "sd15_pipeline":
        print(
            "Next: python scripts/quality_gate_sd15.py "
            f"--asset-id {args.asset_id}"
        )
    else:
        print("Next: run the SD15 IP-Adapter reference-image GPU gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
