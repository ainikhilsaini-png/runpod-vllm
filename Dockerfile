# FROM vllm/vllm-openai:v0.19.0

# # Install supporting tools
# RUN pip install --no-cache-dir hf_transfer runai-model-streamer

# ENV HF_HUB_ENABLE_HF_TRANSFER=1
# ENV VLLM_USE_V1=1
# ENV PYTHONUNBUFFERED=1
# ENV TOKENIZERS_PARALLELISM=false

# # Pre-download weights during build (baked image, ~27 GB)
# # Comment this out if you want faster iterative builds and runtime download
# RUN huggingface-cli download Qwen/Qwen3.6-27B-FP8 \
#     --local-dir /models/qwen3.6-27b-fp8 \
#     --local-dir-use-symlinks False \
#     --max-workers 16

# EXPOSE 8000

# CMD ["--model", "/models/qwen3.6-27b-fp8", \
#      "--served-model-name", "qwen3.6-27b", \
#      "--host", "0.0.0.0", \
#      "--port", "8000", \
#      "--tensor-parallel-size", "1", \
#      "--max-model-len", "131072", \
#      "--max-num-seqs", "16", \
#      "--gpu-memory-utilization", "0.92", \
#      "--kv-cache-dtype", "fp8_e4m3", \
#      "--load-format", "runai_streamer", \
#      "--language-model-only", \
#      "--enable-prefix-caching", \
#      "--enable-chunked-prefill", \
#      "--enforce-eager", \
#      "--reasoning-parser", "qwen3", \
#      "--enable-auto-tool-choice", \
#      "--tool-call-parser", "qwen3_coder", \
#      "--speculative-config", "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}"]

FROM vllm/vllm-openai:v0.19.0

RUN pip install --no-cache-dir runpod aiohttp hf_transfer runai-model-streamer

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV VLLM_USE_V1=1
ENV PYTHONUNBUFFERED=1

# Bake weights
RUN huggingface-cli download Qwen/Qwen3.6-27B-FP8 \
    --local-dir /models/qwen3.6-27b-fp8 \
    --local-dir-use-symlinks False \
    --max-workers 16

# Copy handler
COPY src/handler.py /handler.py

# Default config (can be overridden via env at deploy time)
ENV MODEL_NAME=/models/qwen3.6-27b-fp8
ENV SERVED_MODEL_NAME=qwen3.6-27b
ENV MAX_MODEL_LEN=131072
ENV KV_CACHE_DTYPE=fp8_e4m3
ENV NUM_SPECULATIVE_TOKENS=2

# Runpod worker needs its own entrypoint, override vLLM's
ENTRYPOINT []
CMD ["python", "/handler.py"]
