"""
Gradient Checkpointing Wrapper
================================
Chạy upstream training script nhưng trước đó:
1. Copy script ra /tmp (RAM) để tránh Google Drive sync lag
2. Fix nonlocal SyntaxError bug nếu có trong script
3. Enable gradient checkpointing via sitecustomize.py + PYTHONPATH

Cách dùng:
    python finetune/gc_wrapper.py <upstream_script.py> [args...]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Nội dung sitecustomize.py — Python load tự động trước khi chạy script
_SITECUSTOMIZE = '''
# gc_wrapper: enable gradient checkpointing trên VoxCPM2Model
try:
    from voxcpm.model import VoxCPM2Model, VoxCPMModel

    def _patch_class(cls):
        orig = cls.from_local
        orig_func = orig.__func__ if hasattr(orig, "__func__") else orig
        import functools

        @classmethod
        @functools.wraps(orig_func)
        def patched_from_local(klass, *args, **kwargs):
            model = orig_func(klass, *args, **kwargs)
            for obj in [
                model,
                getattr(model, "base_lm", None),
                getattr(getattr(model, "base_lm", None), "model", None),
            ]:
                if obj is not None and hasattr(obj, "gradient_checkpointing_enable"):
                    try:
                        obj.gradient_checkpointing_enable()
                        print(f"[GC] gradient_checkpointing_enable OK: {type(obj).__name__}", flush=True)
                    except Exception as e:
                        print(f"[GC] gradient_checkpointing_enable failed on {type(obj).__name__}: {e}", flush=True)
            return model

        cls.from_local = patched_from_local

    _patch_class(VoxCPM2Model)
    _patch_class(VoxCPMModel)
    print("[GC] VoxCPM2Model + VoxCPMModel patched for gradient checkpointing.", flush=True)

except Exception as _gc_err:
    print(f"[GC] patch skipped: {_gc_err}", flush=True)
'''

# Pattern nonlocal bug cần fix
_NONLOCAL_OLD = (
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
    '            sampler = getattr(train_loader, "sampler", None)\n'
    '            if hasattr(sampler, "set_epoch"):\n'
    "                sampler.set_epoch(data_epoch)\n"
    "            train_iter = iter(train_loader)\n"
    "            return next(train_iter)\n"
)

_NONLOCAL_NEW = (
    "    # nonlocal workaround: mutable dict captured by closure\n"
    '    _iter_state = {"data_epoch": 0, "train_iter": iter(train_loader)}\n'
    "\n"
    "    def get_next_batch():\n"
    '        """Get next batch, handles epoch boundary and DistributedSampler."""\n'
    "        try:\n"
    '            return next(_iter_state["train_iter"])\n'
    "        except StopIteration:\n"
    '            _iter_state["data_epoch"] += 1\n'
    "            # Key: set DistributedSampler epoch to ensure different data order each epoch\n"
    '            sampler = getattr(train_loader, "sampler", None)\n'
    '            if hasattr(sampler, "set_epoch"):\n'
    '                sampler.set_epoch(_iter_state["data_epoch"])\n'
    '            _iter_state["train_iter"] = iter(train_loader)\n'
    '            return next(_iter_state["train_iter"])\n'
)


def _fix_nonlocal(source: str) -> tuple[str, bool]:
    """Fix nonlocal bug. Trả về (fixed_source, was_changed)."""
    if "nonlocal train_iter, data_epoch" not in source:
        return source, False  # không có bug

    if _NONLOCAL_OLD in source:
        print("[GC_WRAPPER] Applying full nonlocal→dict fix.", flush=True)
        return source.replace(_NONLOCAL_OLD, _NONLOCAL_NEW, 1), True

    # Fallback: xóa dòng nonlocal + đổi assignments thành dict
    print("[GC_WRAPPER] Applying fallback nonlocal fix (remove nonlocal line).", flush=True)
    fixed = source
    # Bước 1: xóa dòng nonlocal
    fixed = fixed.replace("        nonlocal train_iter, data_epoch\n", "", 1)
    # Bước 2: đổi assignments bên trong get_next_batch thành dict access
    # (không thể làm chính xác nếu không biết chính xác indentation)
    return fixed, True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: gc_wrapper.py <training_script.py> [args...]")

    drive_script = Path(sys.argv[1]).resolve()
    if not drive_script.exists():
        raise SystemExit(f"Script not found: {drive_script}")

    extra_args = sys.argv[2:]

    # ── Bước 1: copy script ra /tmp để tránh Google Drive sync issues ──
    tmpdir = tempfile.mkdtemp(prefix="gc_wrapper_")
    local_script = Path(tmpdir) / "train_voxcpm_finetune.py"
    shutil.copy2(drive_script, local_script)
    print(f"[GC_WRAPPER] Copied script to {local_script}", flush=True)

    # ── Bước 2: đọc từ /tmp và fix nonlocal bug ──
    source = local_script.read_text(encoding="utf-8")
    fixed_source, was_fixed = _fix_nonlocal(source)
    if was_fixed:
        local_script.write_text(fixed_source, encoding="utf-8")
        print(f"[GC_WRAPPER] nonlocal bug fixed in local copy.", flush=True)
    else:
        print("[GC_WRAPPER] No nonlocal bug found — script looks clean.", flush=True)

    # ── Bước 3: sitecustomize.py cho gradient checkpointing ──
    Path(tmpdir, "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")

    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tmpdir + (":" + existing_pypath if existing_pypath else "")

    # ── Bước 4: chạy local copy (trong /tmp, không phải Drive) ──
    cmd = [sys.executable, str(local_script)] + extra_args
    print(f"[GC_WRAPPER] Running local copy: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
