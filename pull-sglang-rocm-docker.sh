#!/usr/bin/env bash
# Pull the SGLang ROCm Docker image only (large download). Run once before first
# ./run-sglang-rocm-docker.sh so startup does not block on pull.
#
# Same image as run-sglang-rocm-docker.sh:
#   SGLANG_ROCM_IMAGE=my/image:tag ./pull-sglang-rocm-docker.sh
#
# Model weights still download on first server start (into $HOME/.cache/huggingface)
# unless you prefetch with huggingface-cli.

set -euo pipefail

IMAGE="${SGLANG_ROCM_IMAGE:-lmsysorg/sglang:v0.5.9-rocm720-mi30x}"

echo "Pulling ${IMAGE} ..."
docker pull "${IMAGE}"
echo "Done."
