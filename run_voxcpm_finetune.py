from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download

UPSTREAM_TRAIN_SCRIPT_URL = (
    "https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/scripts/train_voxcpm_finetune.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve the base VoxCPM2 checkpoint and launch the official fine-tuning script."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("finetune/configs/voxcpm2_north_female_lora.yaml"),
        help="Path to a LoRA or full fine-tuning config YAML.",
    )
    parser.add_argument(
        "--upstream-script",
        type=Path,
        default=Path("finetune/upstream/train_voxcpm_finetune.py"),
        help="Local path where the upstream training script should live.",
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path("finetune/.resolved/active_config.yaml"),
        help="Path for the resolved config that is actually passed to training.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
        help="Hugging Face cache used when downloading the base model snapshot.",
    )
    parser.add_argument(
        "--refresh-upstream",
        action="store_true",
        help="Re-download the official training script even if it already exists locally.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow launching training without CUDA. This is usually too slow to be practical.",
    )
    parser.add_argument(
        "--allow-low-vram",
        action="store_true",
        help="Allow training to launch even if detected GPU VRAM is below the recommended threshold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and stop before launching the training process.",
    )
    return parser.parse_args()


_NONLOCAL_BUG_OLD = (
    "    data_epoch = 0\n"
    "    train_iter = iter(train_loader)\n"
    "\n"
    "    def get_next_batch():\n"
    '        """Get next batch, handles epoch boundary and DistributedSampler."""\n'
    "        nonlocal train_iter, data_epoch\n"
    "        try:\n"
    "            return next(train_iter)\n"
    "        except StopIteration:\n"
    "            data_epoch += 1\n"
    "            # Key: set DistributedSampler epoch to ensure different data order each epoch\n"
    "            sampler = getattr(train_loader, \"sampler\", None)\n"
    "            if hasattr(sampler, \"set_epoch\"):\n"
    "                sampler.set_epoch(data_epoch)\n"
    "            train_iter = iter(train_loader)\n"
    "            return next(train_iter)\n"
)

_NONLOCAL_BUG_NEW = (
    "    # nonlocal workaround: use mutable dict so nested function can close over it\n"
    "    _iter_state = {\"data_epoch\": 0, \"train_iter\": iter(train_loader)}\n"
    "\n"
    "    def get_next_batch():\n"
    '        """Get next batch, handles epoch boundary and DistributedSampler."""\n'
    "        try:\n"
    "            return next(_iter_state[\"train_iter\"])\n"
    "        except StopIteration:\n"
    "            _iter_state[\"data_epoch\"] += 1\n"
    "            # Key: set DistributedSampler epoch to ensure different data order each epoch\n"
    "            sampler = getattr(train_loader, \"sampler\", None)\n"
    "            if hasattr(sampler, \"set_epoch\"):\n"
    "                sampler.set_epoch(_iter_state[\"data_epoch\"])\n"
    "            _iter_state[\"train_iter\"] = iter(train_loader)\n"
    "            return next(_iter_state[\"train_iter\"])\n"
)


def _fix_nonlocal_bug(script_path: Path) -> None:
    """Thay thế nonlocal pattern bằng dict-based closure để tránh SyntaxError trên Python 3.12."""
    source = script_path.read_text(encoding="utf-8")
    if "nonlocal train_iter, data_epoch" not in source:
        return  # không có bug, bỏ qua
    if _NONLOCAL_BUG_OLD not in source:
        # Bug có nhưng format khác — fallback: chỉ xóa dòng nonlocal
        print("[FIX] nonlocal pattern differs from expected; removing the nonlocal line only.")
        fixed = source.replace(
            "        nonlocal train_iter, data_epoch\n", "", 1
        )
    else:
        fixed = source.replace(_NONLOCAL_BUG_OLD, _NONLOCAL_BUG_NEW, 1)
    script_path.write_text(fixed, encoding="utf-8")
    print(f"[FIX] Patched nonlocal bug in {script_path.name}")


def ensure_upstream_script(script_path: Path, refresh: bool) -> None:
    if script_path.exists() and not refresh:
        return
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UPSTREAM_TRAIN_SCRIPT_URL, timeout=60) as response:
        script_path.write_bytes(response.read())
    _fix_nonlocal_bug(script_path)


def resolve_config_paths(config: dict, repo_root: Path, cache_dir: Path) -> dict:
    resolved = dict(config)
    pretrained_value = str(config["pretrained_path"])
    pretrained_path = Path(pretrained_value)

    if pretrained_path.exists():
        resolved["pretrained_path"] = str(pretrained_path.resolve())
    else:
        resolved["pretrained_path"] = snapshot_download(
            repo_id=pretrained_value,
            cache_dir=str(cache_dir.resolve()),
            local_files_only=False,
        )

    for key in ["train_manifest", "val_manifest", "save_path", "tensorboard"]:
        value = resolved.get(key)
        if not value:
            continue
        path_value = Path(str(value))
        if not path_value.is_absolute():
            resolved[key] = str((repo_root / path_value).resolve())
        else:
            resolved[key] = str(path_value.resolve())

    return resolved


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    resolved_config_path = (
        (repo_root / args.resolved_config).resolve()
        if not args.resolved_config.is_absolute()
        else args.resolved_config.resolve()
    )
    upstream_script = (
        (repo_root / args.upstream_script).resolve()
        if not args.upstream_script.is_absolute()
        else args.upstream_script.resolve()
    )
    cache_dir = (repo_root / args.cache_dir).resolve() if not args.cache_dir.is_absolute() else args.cache_dir.resolve()

    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available on this machine. Fine-tuning VoxCPM2 is not practical on CPU-only. "
            "Use --allow-cpu only if you explicitly want to try anyway."
        )

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    recommended_min_vram_gb = 20 if config.get("lora") else 40
    ensure_upstream_script(upstream_script, refresh=args.refresh_upstream)
    resolved = resolve_config_paths(config=config, repo_root=repo_root, cache_dir=cache_dir)

    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU VRAM: {total_vram_gb:.1f} GB")
        if total_vram_gb < recommended_min_vram_gb and not args.allow_low_vram:
            raise SystemExit(
                f"Detected only {total_vram_gb:.1f} GB VRAM, but this config recommends at least "
                f"{recommended_min_vram_gb} GB. Re-run with --allow-low-vram if you explicitly want to try anyway."
            )

    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)

    print(f"Resolved config written to: {resolved_config_path}")
    print(f"Using base model snapshot: {resolved['pretrained_path']}")
    print(f"Using upstream training script: {upstream_script}")

    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.allow_low_vram:
        # Giảm memory fragmentation trên GPU nhỏ (khuyến nghị của PyTorch khi OOM)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gc_wrapper = repo_root / "finetune" / "gc_wrapper.py"
    if args.allow_low_vram and gc_wrapper.exists():
        # Dùng wrapper để monkey-patch gradient checkpointing trước khi chạy training
        cmd = [sys.executable, str(gc_wrapper), str(upstream_script),
               "--config_path", str(resolved_config_path)]
    else:
        cmd = [sys.executable, str(upstream_script),
               "--config_path", str(resolved_config_path)]

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()
