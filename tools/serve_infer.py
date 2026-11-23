"""Start the alloylm inference API server for a given model path.

The server exposes the OpenAI-compatible endpoints documented in README.md:

  POST /v1/chat/completions
  POST /v1/chat/interactive
  POST /generate
  GET  /v1/models
  GET  /health

Examples:

  python tools/serve_infer.py --model-path /preset-models --port 8000
  python tools/serve_infer.py --model-path Qwen/Qwen3-0.6B --max-prefill-length 1024 \
      --memory-usage 0.5
"""

import argparse
import asyncio
import signal

import torch

from alloylm.engine.infer_engine.engine import (
    AlloyLMModelConfig,
    InferEngineConfig,
    SPMDInfer,
    SPMDInferConfig,
)
from alloylm.engine.spmd import SPMDActor, SPMDActorConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.chat_template import (
    QWEN_THINKING_PATTERN,
    QWEN_TOOL_PATTERN,
    Qwen3ChatTemplate,
)
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an alloylm model behind the OpenAI-compatible API.")
    parser.add_argument("--model-path", required=True, help="local path or HF id of the model to serve")
    parser.add_argument("--port", type=int, default=8000, help="port of the API server")
    parser.add_argument("--model-name", default="ALLOYLM", help="model id reported by /v1/models")
    parser.add_argument("--max-prefill-length", type=int, default=16 * 1024)
    parser.add_argument("--memory-usage", type=float, default=0.6, help="fraction of GPU memory for the KV cache")
    parser.add_argument("--shard-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--tool-pattern", default=QWEN_TOOL_PATTERN, help="regex for tool-call parsing (default: Qwen)"
    )
    parser.add_argument(
        "--thinking-pattern", default=QWEN_THINKING_PATTERN, help="regex for thinking-block parsing (default: Qwen)"
    )
    parser.add_argument(
        "--chat-template",
        default=Qwen3ChatTemplate,
        help="Jinja template overriding the tokenizer one (default: Qwen3 official)",
    )
    return parser.parse_args()


def build_engine(args: argparse.Namespace) -> SPMDActor:
    shard_dtype = getattr(torch, args.shard_dtype)
    return SPMDActor.create_spmd_actor(
        SPMDInfer,
        args=(
            SPMDInferConfig(
                llm_config=AlloyLMModelConfig(
                    path=args.model_path,
                    model_cls=FSDPQwen2ForCausalLM,
                    fsdp_config=FSDPConfig(
                        train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
                        infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
                        shard_dtype=shard_dtype,
                        lm_head_dtype=torch.bfloat16,
                    ),
                ),
                infer_engine_config=InferEngineConfig(
                    model_name=args.model_name,
                    memory_usage=args.memory_usage,
                    max_prefill_length=args.max_prefill_length,
                    port=args.port,
                    tool_pattern=args.tool_pattern,
                    thinking_pattern=args.thinking_pattern,
                    chat_template=args.chat_template,
                ),
            ),
        ),
        spmd_config=SPMDActorConfig(world_size=1),
    )


async def serve(engine: SPMDActor, args: argparse.Namespace) -> None:
    await engine.launch()
    url = engine._workers[0].url
    print(f"[serve_infer] model={args.model_path} ready at {url}", flush=True)
    print(
        f"[serve_infer] try: curl {url}/chat/completions -H 'Content-Type: application/json' "
        f'-d \'{{"model": "{args.model_name}", "messages": [{{"role": "user", "content": "hi"}}]}}\'',
        flush=True,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        print("[serve_infer] shutting down", flush=True)
        await engine.stop()


def main() -> None:
    args = parse_args()
    engine = build_engine(args)
    try:
        asyncio.run(serve(engine, args))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
