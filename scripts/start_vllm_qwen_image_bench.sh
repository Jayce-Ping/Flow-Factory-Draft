#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for the Qwen-Image-Bench judge ("Q-Judger").
#   - Weights:        Qwen/Qwen-Image-Bench  (a fine-tuned ~27B Qwen3-VL)
#   - Served model id: Qwen-Image-Bench       (must equal the YAML `vlm_model` key)
# Training YAML: set api_base_url=http://<host>:<port>/v1 and vlm_model=<--served-model-name>.
#
# Usage (multi-GPU; tensor-parallel-size defaults to len(CUDA_VISIBLE_DEVICES)):
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   ./scripts/start_vllm_qwen_image_bench.sh --max-model-len 32768
#
# The judge emits a long <think> ... </think> section before the JSON scores, so
# keep --max-model-len well above (image tokens + max_tokens). The reward defaults
# to max_tokens=4096; 32768 model length is a safe starting point.
#
# Optional environment variables:
#   VLLM_BIN               vLLM entrypoint (default: vllm from PATH)
#   MODEL_PATH             HF id or local path (default: Qwen/Qwen-Image-Bench)
#   SERVED_MODEL_NAME      OpenAI "model" id (default: Qwen-Image-Bench); must match YAML vlm_model
#   PORT                   listen port (default: 8000)
#   HOST                   bind address (default: 0.0.0.0)
#   TENSOR_PARALLEL_SIZE   If unset: number of entries in CUDA_VISIBLE_DEVICES, else 1.
#   GPU_MEMORY_UTILIZATION (default: 0.9)
#   Any extra arguments are forwarded to `vllm serve` (e.g. --max-model-len 32768).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image-Bench}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen-Image-Bench}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

case "${TENSOR_PARALLEL_SIZE-unset}" in
  unset)
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      TENSOR_PARALLEL_SIZE="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
      TENSOR_PARALLEL_SIZE=1
    fi
    ;;
esac

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --limit-mm-per-prompt '{"image": 1}' \
  --trust-remote-code \
  "$@"
