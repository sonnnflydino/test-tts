"""
Gradient Checkpointing Wrapper
================================
Chạy upstream training script nhưng TRƯỚC ĐÓ monkey-patch VoxCPM2 để
bật gradient checkpointing — giúp train được trên GPU < 20 GB VRAM (T4, RTX 4050...).

Cách dùng (thay thế gọi trực tiếp upstream script):
    python finetune/gc_wrapper.py <upstream_script.py> --config_path <config.yaml>
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _enable_gc_on_model(model: object) -> None:
    """Thử bật gradient checkpointing trên model hoặc sub-module của nó."""
    candidates = [model]
    # Thêm base_lm và base_lm.model nếu tồn tại
    base_lm = getattr(model, "base_lm", None)
    if base_lm is not None:
        candidates.append(base_lm)
        inner = getattr(base_lm, "model", None)
        if inner is not None:
            candidates.append(inner)

    for obj in candidates:
        enable_fn = getattr(obj, "gradient_checkpointing_enable", None)
        if callable(enable_fn):
            try:
                enable_fn()
                print(
                    f"[GC_WRAPPER] gradient_checkpointing_enable() OK on {type(obj).__name__}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[GC_WRAPPER] gradient_checkpointing_enable() failed on "
                      f"{type(obj).__name__}: {exc}", flush=True)


def _patch_voxcpm2() -> None:
    """Monkey-patch VoxCPM2.from_local để bật GC ngay sau khi model được tạo."""
    try:
        voxcpm_module = importlib.import_module("voxcpm.model.voxcpm2")
    except ImportError as exc:
        print(f"[GC_WRAPPER] Cannot import voxcpm.model.voxcpm2: {exc}", flush=True)
        return

    cls = getattr(voxcpm_module, "VoxCPM2", None)
    if cls is None:
        print("[GC_WRAPPER] VoxCPM2 class not found — skip patch.", flush=True)
        return

    original_from_local = cls.from_local

    @classmethod  # type: ignore[misc]
    def patched_from_local(klass, *args, **kwargs):  # type: ignore[override]
        model = original_from_local.__func__(klass, *args, **kwargs)
        _enable_gc_on_model(model)
        return model

    cls.from_local = patched_from_local
    print("[GC_WRAPPER] VoxCPM2.from_local patched for gradient checkpointing.", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: gc_wrapper.py <training_script.py> [args...]")

    script_path = Path(sys.argv[1]).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    # argv[0] → training script path, argv[1:] → its args
    sys.argv = [str(script_path)] + sys.argv[2:]

    # Patch trước khi exec
    _patch_voxcpm2()

    # Exec training script trong __main__ namespace (giống chạy trực tiếp)
    source = script_path.read_text(encoding="utf-8")
    code = compile(source, str(script_path), "exec")
    ns: dict = {"__name__": "__main__", "__file__": str(script_path)}
    exec(code, ns)  # noqa: S102


if __name__ == "__main__":
    main()
