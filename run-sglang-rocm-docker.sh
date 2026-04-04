#!/usr/bin/env bash
# Run SGLang in the official ROCm Docker image with a working amd-aiter import.
# Intended for AMD GPUs (e.g. V620 / gfx1030). Hides CUDA devices inside the container.
#
# If logs show a frame like "sglang_rocm_rope.py" in forward, the HIP shim is active; each
# fix may expose the next broken sgl_kernel op on consumer RDNA. For production on V620,
# consider vLLM (ROCm) for TP or llama.cpp where gfx1030 is better tested; full PP+TP parity
# varies by project—check current vLLM pipeline-parallel docs for your model class.
#
# gfx1030 / RDNA2: ROCm AITER does not whitelist gfx1030 in chip_info.py. It honors
# GPU_ARCHS: if the value contains ';', the last segment is used as the arch.
# Default native;gfx1100 is a best-effort alias (RDNA3) so imports proceed; kernels
# may still fail at runtime on V620—in that case use another stack (e.g. vLLM/llama.cpp).
# Override: AITER_GPU_ARCHS='native;gfx1101' ./run-sglang-rocm-docker.sh
#
# Usage:
#   ./pull-sglang-rocm-docker.sh   # optional: prefetch the Docker image
#   ./run-sglang-rocm-docker.sh
#   SGLANG_MODEL=Qwen/Qwen2.5-0.5B-Instruct ./run-sglang-rocm-docker.sh
#   # Qwen3.5 text checkpoints need a newer Transformers than the image ships:
#   UPGRADE_TRANSFORMERS=1 SGLANG_MODEL=principled-intelligence/Qwen3.5-0.8B-text-only ./run-sglang-rocm-docker.sh
#   SKIP_AITER_REINSTALL=1 ./run-sglang-rocm-docker.sh   # if image already fixed (e.g. docker commit)
#
# Optional: append extra launch_server flags
#   SGLANG_EXTRA_ARGS="--max-total-tokens 8192" ./run-sglang-rocm-docker.sh
#
# HIP/CUDA graph capture often segfaults on gfx1030 with MI300-oriented wheels (e.g. during
# sgl_kernel rotary_embedding capture). Default is to pass --disable-cuda-graph.
# Re-enable: SGLANG_DISABLE_CUDA_GRAPH=0 ./run-sglang-rocm-docker.sh
#
# HIP shim: zzz_sglang_rocm_rope.pth + sitecustomize load sglang_rocm_rope.py (native forward for
# rotary + activation MultiPlatformOp layers on HIP; reduces sgl_kernel segfaults on gfx1030).
# Debug: SGLANG_ROCM_NATIVE_ROPE_DEBUG=1 ./run-sglang-rocm-docker.sh
# Disable: SGLANG_ROCM_NATIVE_ROPE=0 ./run-sglang-rocm-docker.sh
# Default attention backend is triton on ROCm (not aiter); override with SGLANG_ATTENTION_BACKEND.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${SGLANG_ROCM_IMAGE:-lmsysorg/sglang:v0.5.9-rocm720-mi30x}"
# Default: Qwen3 small LM (model_type qwen3). Qwen3.5 text uses qwen3_5_text — see UPGRADE_TRANSFORMERS.
MODEL_PATH="${SGLANG_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${SGLANG_PORT:-30000}"
DTYPE="${SGLANG_DTYPE:-float16}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
SHM_SIZE="${SGLANG_SHM_SIZE:-16g}"
SKIP_AITER_REINSTALL="${SKIP_AITER_REINSTALL:-0}"
EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
# Passed into the container as GPU_ARCHS (see ROCm/aiter aiter/jit/utils/chip_info.py).
AITER_GPU_ARCHS="${AITER_GPU_ARCHS:-"native;gfx1100"}"
UPGRADE_TRANSFORMERS="${UPGRADE_TRANSFORMERS:-0}"
TRANSFORMERS_PIP_SPEC="${TRANSFORMERS_PIP_SPEC:-}"
# 1 = pass --disable-cuda-graph (recommended for V620 / consumer ROCm)
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
# 1 = use PyTorch native RoPE on HIP (see sglang_rocm_rope.py)
SGLANG_ROCM_NATIVE_ROPE="${SGLANG_ROCM_NATIVE_ROPE:-1}"
# ROCm default in SGLang is aiter; triton is safer on gfx1030 / non-MI3x. Override: SGLANG_ATTENTION_BACKEND=aiter
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
# Apex fused RoPE: avoid Aiter path (see container log warning)
USE_ROCM_AITER_ROPE_BACKEND="${USE_ROCM_AITER_ROPE_BACKEND:-0}"

# -it breaks when stdin is not a TTY (e.g. CI); use -i only then.
if [[ -t 0 ]] && [[ -t 1 ]]; then
  DOCKER_TTY=(-it)
else
  DOCKER_TTY=(-i)
fi

docker run --rm "${DOCKER_TTY[@]}" \
  --network=host \
  --ipc=host \
  "--shm-size=${SHM_SIZE}" \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt=seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}/sglang_rocm_launch.py:/sglang_rocm_launch.py:ro" \
  -v "${SCRIPT_DIR}/sglang_rocm_rope.py:/sglang_rocm_rope.py:ro" \
  -e HF_HOME=/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES= \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" \
  -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}" \
  -e SGLANG_USE_AITER=0 \
  -e GPU_ARCHS="${AITER_GPU_ARCHS}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e PORT="${PORT}" \
  -e DTYPE="${DTYPE}" \
  -e SKIP_AITER_REINSTALL="${SKIP_AITER_REINSTALL}" \
  -e UPGRADE_TRANSFORMERS="${UPGRADE_TRANSFORMERS}" \
  -e TRANSFORMERS_PIP_SPEC="${TRANSFORMERS_PIP_SPEC}" \
  -e SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH}" \
  -e SGLANG_ROCM_NATIVE_ROPE="${SGLANG_ROCM_NATIVE_ROPE}" \
  -e SGLANG_ROCM_NATIVE_ROPE_DEBUG="${SGLANG_ROCM_NATIVE_ROPE_DEBUG:-}" \
  -e "SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND}" \
  -e "USE_ROCM_AITER_ROPE_BACKEND=${USE_ROCM_AITER_ROPE_BACKEND}" \
  -e EXTRA_ARGS="${EXTRA_ARGS}" \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
    if [[ "${SKIP_AITER_REINSTALL}" != "1" ]]; then
      pip uninstall -y amd-aiter aiter 2>/dev/null || true
      cd /sgl-workspace/aiter
      git submodule update --init --recursive
      pip install --no-cache-dir -e .
    fi
    if [[ -n "${TRANSFORMERS_PIP_SPEC}" ]]; then
      pip install --no-cache-dir ${TRANSFORMERS_PIP_SPEC}
    elif [[ "${UPGRADE_TRANSFORMERS}" == "1" ]]; then
      pip install --upgrade --no-cache-dir transformers
    fi
    # After pip: install RoPE hook (.pth runs during site init; sitecustomize is backup).
    if [[ -f /sglang_rocm_rope.py ]]; then
      _SP="$(python3 -c "import site; print(next(p for p in site.getsitepackages() if p.endswith(\"site-packages\")))")"
      cp /sglang_rocm_rope.py "${_SP}/sglang_rocm_rope.py"
      printf "%s\n" "import sglang_rocm_rope" > "${_SP}/zzz_sglang_rocm_rope.pth"
      printf "%s\n" "import sglang_rocm_rope" > "${_SP}/sitecustomize.py"
    fi
    GRAPH_ARGS=
    if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-1}" != "0" ]]; then
      GRAPH_ARGS=--disable-cuda-graph
    fi
    ATTN_ARGS=
    if [[ -n "${SGLANG_ATTENTION_BACKEND:-}" ]]; then
      ATTN_ARGS="--attention-backend ${SGLANG_ATTENTION_BACKEND}"
    fi
    # shellcheck disable=SC2086
    exec python3 /sglang_rocm_launch.py \
      --model-path "${MODEL_PATH}" \
      --dtype "${DTYPE}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      ${GRAPH_ARGS} \
      ${ATTN_ARGS} \
      ${EXTRA_ARGS}
  '
