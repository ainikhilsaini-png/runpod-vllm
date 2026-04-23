"""
Runpod Serverless handler for Qwen3.6-27B-FP8.
Starts vLLM's OpenAI server as a subprocess and proxies Runpod requests to it.
"""
import os
import subprocess
import time
import json
import asyncio
import aiohttp
import runpod

# Start vLLM subprocess once when the worker boots
VLLM_PROC = None
VLLM_READY = False
VLLM_BASE_URL = "http://127.0.0.1:8000"


def build_vllm_command():
    """Build the vLLM serve command from env vars."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", os.environ.get("MODEL_NAME", "/models/qwen3.6-27b-fp8"),
        "--served-model-name", os.environ.get("SERVED_MODEL_NAME", "qwen3.6-27b"),
        "--host", "127.0.0.1",
        "--port", "8000",
        "--tensor-parallel-size", os.environ.get("TENSOR_PARALLEL_SIZE", "1"),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "131072"),
        "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "16"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"),
        "--kv-cache-dtype", os.environ.get("KV_CACHE_DTYPE", "fp8_e4m3"),
        "--reasoning-parser", os.environ.get("REASONING_PARSER", "qwen3"),
        "--tool-call-parser", os.environ.get("TOOL_CALL_PARSER", "qwen3_coder"),
    ]

    # Optional boolean flags
    if os.environ.get("ENABLE_PREFIX_CACHING", "1") == "1":
        cmd.append("--enable-prefix-caching")
    if os.environ.get("ENABLE_CHUNKED_PREFILL", "1") == "1":
        cmd.append("--enable-chunked-prefill")
    if os.environ.get("ENFORCE_EAGER", "1") == "1":
        cmd.append("--enforce-eager")
    if os.environ.get("ENABLE_AUTO_TOOL_CHOICE", "1") == "1":
        cmd.append("--enable-auto-tool-choice")
    if os.environ.get("LANGUAGE_MODEL_ONLY", "1") == "1":
        cmd.append("--language-model-only")

    # Speculative decoding
    num_spec = os.environ.get("NUM_SPECULATIVE_TOKENS", "2")
    if int(num_spec) > 0:
        spec_config = json.dumps({
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": int(num_spec)
        })
        cmd.extend(["--speculative-config", spec_config])

    # Load format
    load_format = os.environ.get("LOAD_FORMAT", "runai_streamer")
    cmd.extend(["--load-format", load_format])

    return cmd


def start_vllm():
    """Start vLLM as a subprocess."""
    global VLLM_PROC
    if VLLM_PROC is not None:
        return
    cmd = build_vllm_command()
    print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
    VLLM_PROC = subprocess.Popen(cmd)


async def wait_for_vllm(timeout=600):
    """Poll vLLM's health endpoint until it's ready."""
    global VLLM_READY
    if VLLM_READY:
        return True

    async with aiohttp.ClientSession() as session:
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with session.get(f"{VLLM_BASE_URL}/health", timeout=2) as resp:
                    if resp.status == 200:
                        VLLM_READY = True
                        print("vLLM is ready", flush=True)
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def handler(event):
    """
    Runpod handler. Proxies requests to the local vLLM server.

    Expected input format:
    {
      "input": {
        "openai_route": "/v1/chat/completions",
        "openai_input": { ... OpenAI chat completion request ... }
      }
    }
    """
    # Ensure vLLM is up
    if not VLLM_READY:
        ok = await wait_for_vllm()
        if not ok:
            return {"error": "vLLM failed to start in time"}

    job_input = event.get("input", {})
    route = job_input.get("openai_route", "/v1/chat/completions")
    payload = job_input.get("openai_input", {})
    stream = payload.get("stream", False)

    url = f"{VLLM_BASE_URL}{route}"

    async with aiohttp.ClientSession() as session:
        if route == "/v1/models" or (not payload):
            # Simple GET
            async with session.get(url) as resp:
                return await resp.json()

        if stream:
            # Streaming response — yield chunks
            async with session.post(url, json=payload) as resp:
                async for line in resp.content:
                    if line:
                        yield line.decode("utf-8")
        else:
            # Non-streaming
            async with session.post(url, json=payload) as resp:
                return await resp.json()


# Start vLLM as soon as the handler module is imported (worker boot)
start_vllm()

# Register with Runpod
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,  # Allows streaming responses
})
