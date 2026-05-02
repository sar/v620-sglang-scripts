FROM lmsysorg/sglang:v0.5.9-rocm720-mi30x

# Copy helper scripts into the image
COPY sglang_rocm_rope.py /sglang_rocm_rope.py
COPY sglang_rocm_launch.py /sglang_rocm_launch.py

# 1. Build & install AITER from source (internet required ONLY during build)
WORKDIR /sgl-workspace/aiter
RUN git submodule update --init --recursive && \
    pip install --no-cache-dir -e . && \
    # Clean up git metadata to save image size
    rm -rf .git

# 2. Install the ROCm RoPE hook into Python site-packages
RUN SP=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp /sglang_rocm_rope.py "$SP/" && \
    echo "import sglang_rocm_rope" > "$SP/zzz_sglang_rocm_rope.pth" && \
    echo "import sglang_rocm_rope" > "$SP/sitecustomize.py"

# 3. Set safe defaults for consumer ROCm / gfx1030 (disables runtime rebuilds)
ENV CUDA_VISIBLE_DEVICES= \
    NVIDIA_VISIBLE_DEVICES=void \
    SGLANG_USE_AITER=0 \
    GPU_ARCHS="native;gfx1100" \
    SGLANG_DISABLE_CUDA_GRAPH=1 \
    SGLANG_ROCM_NATIVE_ROPE=1 \
    SGLANG_ATTENTION_BACKEND=triton \
    USE_ROCM_AITER_ROPE_BACKEND=0 \
    SKIP_AITER_REINSTALL=1 \
    UPGRADE_TRANSFORMERS=0

WORKDIR /sgl-workspace