"""
Gradient Checkpointing Wrapper
================================
Chạy upstream training script nhưng trước đó enable gradient checkpointing
trên VoxCPM2Model bằng cách inject sitecustomize.py vào PYTHONPATH.

sitecustomize.py được Python load tự động trước khi chạy bất kỳ script nào —
không cần exec/compile file training, tránh hoàn toàn vấn đề SyntaxError.

Cách dùng:
    python finetune/gc_wrapper.py <upstream_script.py> [args...]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Nội dung sitecustomize.py — sẽ được Python load tự động khi khởi động
_SITECUSTOMIZE = '''
# gc_wrapper: enable gradient checkpointing trên VoxCPM2Model
try:
    from voxcpm.model import VoxCPM2Model, VoxCPMModel

    def _patch_class(cls):
        orig = cls.from_local
        # Lấy underlying function (classmethod wrapper)
        orig_func = orig.__func__ if hasattr(orig, "__func__") else orig

        import functools

        @classmethod
        @functools.wraps(orig_func)
        def patched_from_local(klass, *args, **kwargs):
            model = orig_func(klass, *args, **kwargs)
            # Thử enable GC trên model và các sub-module
            for attr_path in [
                [model],
                [getattr(model, "base_lm", None)],
                [getattr(getattr(model, "base_lm", None), "model", None)],
            ]:
                obj = attr_path[0]
                if obj is not None and hasattr(obj, "gradient_checkpointing_enable"):
                    try:
                        obj.gradient_checkpointing_enable()
                        print(
                            f"[GC] gradient_checkpointing_enable OK: {type(obj).__name__}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GC] gradient_checkpointing_enable failed on "
                              f"{type(obj).__name__}: {e}", flush=True)
            return model

        cls.from_local = patched_from_local

    _patch_class(VoxCPM2Model)
    _patch_class(VoxCPMModel)
    print("[GC] VoxCPM2Model + VoxCPMModel patched for gradient checkpointing.", flush=True)

except Exception as _gc_err:
    print(f"[GC] patch skipped: {_gc_err}", flush=True)
'''


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: gc_wrapper.py <training_script.py> [args...]")

    script_path = Path(sys.argv[1]).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    extra_args = sys.argv[2:]

    # Ghi sitecustomize.py vào thư mục tạm, thêm vào đầu PYTHONPATH
    tmpdir = tempfile.mkdtemp(prefix="gc_wrapper_")
    Path(tmpdir, "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    print(f"[GC_WRAPPER] sitecustomize.py written to {tmpdir}", flush=True)

    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tmpdir + (":" + existing_pypath if existing_pypath else "")

    # Chạy training script như subprocess bình thường — không exec/compile
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"[GC_WRAPPER] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
