# Qwen3.6-27B-FP8 Runpod Serverless Worker

vLLM 0.19+ based serverless worker for Qwen/Qwen3.6-27B-FP8.

## Hardware requirement
- 1× NVIDIA H100 80GB or H200 141GB
- 60 GB container disk

## Features enabled
- FP8 weights + FP8 KV cache
- MTP speculative decoding (qwen3_next_mtp, 2 token)
- Tool calling (qwen3_coder parser)
- Reasoning parser (qwen3)
- Prefix caching
- Chunked prefill
- 131K context window
