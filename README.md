# SGLang on AMD ROCm (Docker)

Scripts to run [SGLang](https://github.com/sgl-project/sglang) in the official ROCm image, with workarounds for **consumer AMD GPUs** (e.g. gfx1030 / Radeon Pro V620) where MI300-oriented `sgl_kernel` builds can segfault.

## Disk space (important)

Plan for **on the order of ~150 GB free** for a comfortable first-time setup:

- **Docker image** `lmsysorg/sglang:v0.5.9-rocm720-mi30x`: large multi-layer pull and local store.
- **Docker build/extract overhead** during pull and run.
- **Hugging Face cache** (`~/.cache/huggingface`): model weights (default small model is modest; larger models add many GB).
- **pip / editable installs** inside the container each run (e.g. `amd-aiter`) use extra space under Docker’s storage.

Exact numbers depend on image updates, model choice, and Docker’s storage driver. If space is tight, prune unused images (`docker system prune`) and use a smaller `SGLANG_MODEL`.

## Prerequisites

- Linux host with **Docker**
- **AMD GPU** with a working **ROCm** stack on the host (enough that `/dev/kfd` and `/dev/dri` work for the container)
- All scripts in **this directory** kept together (`run-sglang-rocm-docker.sh` bind-mounts sibling files)

## Files

| File | Role |
|------|------|
| `pull-sglang-rocm-docker.sh` | Optional: `docker pull` only (same image as run script). |
| `run-sglang-rocm-docker.sh` | Main entry: `docker run`, aiter fix, HIP shim install, start server. |
| `sglang_rocm_launch.py` | Wrapper that starts `sglang.launch_server`. |
| `sglang_rocm_rope.py` | HIP shim: forces PyTorch-native paths for some layers (see script header). |

## Running

1. Optional — prefetch the image (avoids a long wait on first `run`):

   ```bash
   ./pull-sglang-rocm-docker.sh
   ```

   Same as `docker pull lmsysorg/sglang:v0.5.9-rocm720-mi30x` unless you set `SGLANG_ROCM_IMAGE`.

2. Start the server (default model `Qwen/Qwen3-0.6B`, port **30000**, `network=host`):

   ```bash
   ./run-sglang-rocm-docker.sh
   ```

3. Wait until logs show the HTTP server listening and (after warmup) something like “ready to roll”.

### Common environment overrides

```bash
# Another model
SGLANG_MODEL=Qwen/Qwen2.5-0.5B-Instruct ./run-sglang-rocm-docker.sh

# Different GPU index
HIP_VISIBLE_DEVICES=0 ./run-sglang-rocm-docker.sh

# Different port (still host networking)
SGLANG_PORT=30000 ./run-sglang-rocm-docker.sh

# Skip editable aiter reinstall if you committed a fixed image
SKIP_AITER_REINSTALL=1 ./run-sglang-rocm-docker.sh

# Custom image (must match what you pulled)
SGLANG_ROCM_IMAGE=lmsysorg/sglang:v0.5.9-rocm720-mi30x ./run-sglang-rocm-docker.sh
```

Hugging Face downloads go to **`$HOME/.cache/huggingface`** on the host (mounted into the container).

## Using the API

Base URL with defaults: **`http://127.0.0.1:30000`** (host networking).

**List models (OpenAI-compatible):**

```bash
curl -s http://127.0.0.1:30000/v1/models | jq .
```

**Chat completions** (use the `id` from `/v1/models` as `model`):

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }' | jq .
```

**SGLang `/generate`:**

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 32}}' | jq .
```

**Health / info:**

```bash
curl -s http://127.0.0.1:30000/model_info
```

## Sharing these scripts

There are **no hard-coded user paths**: the run script uses **`$HOME/.cache/huggingface`** and the directory containing the scripts. Recipients keep all four files together and run from that folder.

## Limitations

- Tuned for **ROCm MI-class** wheels; **gfx1030 / RDNA** may need the bundled HIP shim and is not officially guaranteed by upstream.
- For production on marginal GPUs, consider **vLLM (ROCm)** or **llama.cpp** as alternatives.
