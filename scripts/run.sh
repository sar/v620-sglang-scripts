docker run --rm -it --network=host --ipc=host --shm-size=16g \
  --group-add=video --cap-add=SYS_PTRACE --security-opt=seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e MODEL_PATH="${SGLANG_MODEL:-Qwen/Qwen3-0.6B}" \
  -e PORT="${SGLANG_PORT:-30000}" \
  -e DTYPE="${SGLANG_DTYPE:-float16}" \
  -e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}" \
  -e EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}" \
  sglang-rocm-offline:latest \
  bash -lc 'exec python3 /sglang_rocm_launch.py \
    --model-path "$MODEL_PATH" \
    --dtype "$DTYPE" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --disable-cuda-graph \
    --attention-backend triton \
    $EXTRA_ARGS'