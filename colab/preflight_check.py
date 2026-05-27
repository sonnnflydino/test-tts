from __future__ import annotations

import argparse
import json

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the current Colab GPU is suitable for VoxCPM fine-tuning.")
    parser.add_argument(
        "--mode",
        choices=["lora", "full"],
        default="lora",
        help="Training mode you intend to run.",
    )
    parser.add_argument(
        "--allow-low-vram",
        action="store_true",
        help="Return success even if the VRAM is below the recommended threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_vram_gb = 20 if args.mode == "lora" else 40

    payload = {
        "cuda_available": torch.cuda.is_available(),
        "mode": args.mode,
        "recommended_min_vram_gb": min_vram_gb,
    }

    if not torch.cuda.is_available():
        print(json.dumps(payload, indent=2))
        raise SystemExit("CUDA is not available in this Colab runtime.")

    props = torch.cuda.get_device_properties(0)
    payload["gpu_name"] = props.name
    payload["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
    print(json.dumps(payload, indent=2))

    if payload["gpu_vram_gb"] < min_vram_gb and not args.allow_low_vram:
        raise SystemExit(
            f"Only {payload['gpu_vram_gb']} GB VRAM detected. "
            f"Recommended minimum for {args.mode} mode is {min_vram_gb} GB."
        )


if __name__ == "__main__":
    main()
