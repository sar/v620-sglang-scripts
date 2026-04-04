# Loaded via site-packages/zzz_sglang_rocm_rope.pth ("import sglang_rocm_rope") and/or
# sitecustomize.py (see run-sglang-rocm-docker.sh).
#
# On HIP, routes selected MultiPlatformOp subclasses to forward_native so MI300-oriented
# sgl_kernel ops are skipped on gfx1030/V620 (they often segfault). Controlled by
# SGLANG_ROCM_NATIVE_ROPE (historical name; gates the whole HIP native op shim).
#
# Module prefixes: extend if the next crash moves to another sgl_kernel elementwise op.

import os
import sys

_HIP_NATIVE_MODULE_PREFIXES = (
    "sglang.srt.layers.rotary_embedding",
    "sglang.srt.layers.activation",
)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )


def _use_forward_native(cls: type) -> bool:
    mod = getattr(cls, "__module__", "") or ""
    if not any(mod.startswith(p) for p in _HIP_NATIVE_MODULE_PREFIXES):
        return False
    if cls.__name__ == "DualChunkRotaryEmbedding":
        return False
    return True


def _apply_hip_native_multiform_patch() -> None:
    if not _truthy("SGLANG_ROCM_NATIVE_ROPE"):
        return
    try:
        import sglang.srt.layers.utils.multi_platform as mp
    except ImportError as e:
        if _truthy("SGLANG_ROCM_NATIVE_ROPE_DEBUG"):
            print(
                f"[sglang_rocm_rope] skip (import multi_platform): {e}",
                file=sys.stderr,
            )
        return

    if not mp._is_hip:
        if _truthy("SGLANG_ROCM_NATIVE_ROPE_DEBUG"):
            print("[sglang_rocm_rope] skip (_is_hip is False)", file=sys.stderr)
        return

    _orig = mp.MultiPlatformOp.forward
    if getattr(_orig, "__sglang_rocm_rope__", False):
        return

    def forward(self, *args, **kwargs):
        if _use_forward_native(type(self)):
            return self.forward_native(*args, **kwargs)
        return _orig(self, *args, **kwargs)

    forward.__sglang_rocm_rope__ = True  # type: ignore[attr-defined]
    mp.MultiPlatformOp.forward = forward  # type: ignore[method-assign]

    if _truthy("SGLANG_ROCM_NATIVE_ROPE_DEBUG"):
        print(
            "[sglang_rocm_rope] HIP: MultiPlatformOp.forward -> forward_native for "
            f"classes in: {', '.join(_HIP_NATIVE_MODULE_PREFIXES)} "
            "(except DualChunkRotaryEmbedding)",
            file=sys.stderr,
        )


_apply_hip_native_multiform_patch()
