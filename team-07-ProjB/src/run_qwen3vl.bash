CUDA_VISIBLE_DEVICES=6 vllm serve \
    Qwen/Qwen3-VL-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768 \
    --port 8001
