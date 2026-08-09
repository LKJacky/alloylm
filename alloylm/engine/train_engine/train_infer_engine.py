import asyncio
import gc
import os
import uuid

import aiofiles
import httpx
import jinja2
import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.engine import InferEngine, InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.spmd import SPMDActor, SPMDActorConfig
from alloylm.engine.train_engine.utils import (
    get_logger,
)

from .hack_client import HighConcurrentClient as AsyncClient
from .hack_client import HighConcurrentClientInteractive as AsyncClientInteractive
from .train_engine import RLInput, TrainEngine, TrainEngineConfig

logger = get_logger()


class TrainInferEngineConfig(BaseModel):
    train_config: TrainEngineConfig
    infer_config: InferEngineConfig

    @property
    def work_dir(self):
        return self.train_config.work_dir

    @property
    def num_workers(self):
        return self.train_config.num_workers


class TrainInferEngine:
    def __init__(
        self,
        model_config: AlloyLMModelConfig,
        engine_config: TrainInferEngineConfig,
    ):
        self.model = model_config.build()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )
        self.args = engine_config  # TODO rename args to config

        self.infer_engine = InferEngine(
            self.model,
            tokenizer=self.tokenizer,
            engine_config=self.args.infer_config,
        )
        self.train_engine = TrainEngine(
            self.model,
            self.tokenizer,
            config=self.args.train_config,
        )

    async def lazy_init(self):
        pass

    async def launch_server(self):
        self.train_engine.offload_optimizer()
        gc.collect()
        torch.cuda.empty_cache()
        self.infer_engine.gather_context.__enter__()
        await self.infer_engine.launch()
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)

    async def stop_serve(self):
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
        await self.infer_engine.stop()
        self.infer_engine.gather_context.__exit__(None, None, None)
        self.train_engine.activate_optimizer()
        gc.collect()
        torch.cuda.empty_cache()

    def get_server_ip(self):
        return self.infer_engine.url

    def step(self, batch, step):
        return self.train_engine.step(batch, step)

    def resume(self, folder):
        return self.train_engine.resume(folder)

    def checkpoint(self, folder):
        return self.train_engine.checkpoint(folder)


# spmd


class SpmdTrainInferEngine:
    def __init__(self, model_config: AlloyLMModelConfig, engine_config: TrainInferEngineConfig):
        self.config = engine_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )
        get_logger().info(f"launch {engine_config.train_config.num_workers} model actors")
        self.actor = SPMDActor.create_spmd_actor(
            TrainInferEngine,
            args=(model_config, engine_config),
            spmd_config=SPMDActorConfig(
                world_size=engine_config.train_config.num_workers,
                num_gpus=1,
                num_cpus=10,
                memory=16 * 1024**3,
            ),
        )

        self.activate_server_event = asyncio.Event()
        AsyncClient.tokenizer = self.tokenizer
        jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )
        AsyncClient.chat_template = jinja_env.from_string(self.tokenizer.get_chat_template())
        AsyncClient.server_activate_event = self.activate_server_event

        AsyncClientInteractive.chat_template = AsyncClient.chat_template
        AsyncClientInteractive.server_activate_event = self.activate_server_event
        AsyncClientInteractive.tokenizer = self.tokenizer

    async def lazy_init(self):
        await self.actor.lazy_init()

    async def train_wrapper(self, batch: list[RLInput], step):
        if step == 0 and batch:
            input_text = self.tokenizer.decode(batch[0]["input_ids"], skip_special_tokens=False)
            labels_text = self.tokenizer.decode(
                [token_id for token_id in batch[0]["labels"] if token_id != -100],
                skip_special_tokens=False,
            )
            async with aiofiles.open(
                os.path.join(self.config.work_dir, "train_sample.txt"), "w", encoding="utf-8"
            ) as f:
                await f.write(f"Input Text:\n{input_text}\n\n")
                await f.write(f"Labels Text:\n{labels_text}\n")
        return await self.train(batch, step)

    async def train(self, batch: list[RLInput], step):
        train_data = []
        for rl_data in batch:
            train_data.append(
                {
                    **rl_data,
                    "id": uuid.uuid4().hex,
                    "num_tokens": len(rl_data["input_ids"]),
                }
            )

        results = await self.actor.step(train_data, step)
        return results[0]

    async def serve(self):
        await self.actor.launch_server()
        self.activate_server_event.set()
        AsyncClient.generated_tokens = 0
        AsyncClientInteractive.generated_tokens = 0
        AsyncClient.prefill_overhead = 0
        AsyncClientInteractive.prefill_overhead = 0

    async def stop_serve(self):
        self.activate_server_event.clear()
        url = await self.get_server_ip()
        while AsyncClient.running_count > 0:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{url.replace('/v1', '')}/abort_request", json={"abort_all": True}, timeout=3600
                )
                assert response.status_code == 200, f"Abort request failed with status code {response.status_code}"
            if AsyncClient.running_count > 0:
                await asyncio.sleep(0.1)
                logger.info(f"Waiting for {AsyncClient.running_count} requests to be aborted...")
        generated_tokens = AsyncClient.generated_tokens + AsyncClientInteractive.generated_tokens
        total_prefill_overhead = AsyncClient.prefill_overhead + AsyncClientInteractive.prefill_overhead
        logger.info(
            f"Generated {generated_tokens // 1024} k tokens, prefill overhead: {total_prefill_overhead // 1024} k tokens. Approximate Efficiency: {(generated_tokens / (total_prefill_overhead / 4 + generated_tokens + 1)):.4f}"
        )
        await self.actor.stop_serve()
        AsyncClient.step()
        return AsyncClient.generated_tokens + AsyncClientInteractive.generated_tokens

    async def resume(self, folder):
        await self.actor.resume(folder)

    async def checkpoint(self, folder):
        await self.actor.checkpoint(folder)

    async def get_server_ip(self):
        return (await self.actor.get_server_ip())[0]

    def __del__(self):
        if hasattr(self, "actor"):
            self.actor.shutdown()
            del self.actor
        gc.collect()
