import asyncio
import gc
import logging
import os
import uuid
from collections.abc import Callable

import ray
import torch
from pydantic import BaseModel
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.engine import InferEngine, InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.spmd import SPMDActor, SPMDActorConfig
from alloylm.engine.train_engine.dataset import SFTData
from alloylm.engine.train_engine.utils import (
    get_logger,
)

from ..infer_engine.infer_bank import InferBank
from .train_engine import RLInput, TrainEngine, TrainEngineConfig
from .utils import engine_logger_name

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
        self.logger = get_logger(
            engine_logger_name(),
            path=os.path.join(
                engine_config.train_config.work_dir,
                f"{engine_logger_name()}.log",
            ),
            output_to_stdout=True,
            log_level=logging.WARNING,
            force_recreate=True,
        )
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
        self.logger.info(f"TrainInferEngine initialized. with {self.model}")

    async def lazy_init(self):
        pass

    # infer
    async def launch_server(self):
        self.train_engine.offload_optimizer()
        gc.collect()
        torch.cuda.empty_cache()
        self.infer_engine.gather_context.__enter__()
        await self.infer_engine.launch()
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)

    async def pause_server(self):
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
        infer_tokens = await self.infer_engine.pause()
        self.infer_engine.gather_context.__exit__(None, None, None)
        self.train_engine.activate_optimizer()
        gc.collect()
        torch.cuda.empty_cache()
        return infer_tokens

    async def stop_server(self):
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
        infer_tokens = await self.infer_engine.stop()
        self.infer_engine.gather_context.__exit__(None, None, None)
        self.train_engine.activate_optimizer()
        gc.collect()
        torch.cuda.empty_cache()
        return infer_tokens

    def get_server_ip(self):
        return self.infer_engine.url

    def fetch_infer_info(self):
        return self.infer_engine.fetch_infer_info()

    # train

    def step(self, batch, step):
        for i, item in enumerate(batch):
            infer_info = item.pop("infer_info")
            if isinstance(infer_info, ray.ObjectRef):
                infer_info = ray.get(infer_info)
            batch[i] = {**item, **infer_info, "num_tokens": len(infer_info["input_ids"])}
        if dist.get_rank() == 0 and step == 0 and batch:
            input_text = self.tokenizer.decode(batch[-1]["input_ids"], skip_special_tokens=False)
            labels_text = self.tokenizer.decode(
                [token_id for token_id in batch[-1]["labels"] if token_id != -100],
                skip_special_tokens=False,
            )
            with open(os.path.join(self.args.work_dir, "train_sample.txt"), "w", encoding="utf-8") as f:
                f.write(f"Input Text:\n{input_text}\n\n")
                f.write(f"Labels Text:\n{labels_text}\n")

        return self.train_engine.step(batch, step)

    # checkpoint

    def resume(self, folder):
        return self.train_engine.resume(folder)

    def checkpoint(self, folder):
        return self.train_engine.checkpoint(folder)

    # sft
    def set_sft_data(
        self,
        batch: list[SFTData],
        jsonl_sources: list,
        chat_template: Callable,
    ):
        self.train_engine.set_sft_data(batch, jsonl_sources, chat_template)

    def step_sft(self, num_micro_steps: int) -> dict[str, float]:
        return self.train_engine.step_sft(num_micro_steps)


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
        self.logger = get_logger()
        self.logger.info(f"launch {engine_config.train_config.num_workers} model actors")
        self.actor: TrainInferEngine = SPMDActor.create_spmd_actor(
            TrainInferEngine,
            args=(model_config, engine_config),
            spmd_config=SPMDActorConfig(
                world_size=engine_config.train_config.num_workers,
                num_gpus=1,
                num_cpus=10,
                memory=16 * 1024**3,
            ),
        )

        self.infer_bank = InferBank()
        self.url = None

    async def lazy_init(self):
        await self.actor.lazy_init()

    async def train_wrapper(self, batch: list[RLInput], step):
        train_data = []
        for rl_data in batch:
            infer_info = self.infer_bank.retrieve_infer_info(rl_data["messages"])
            if infer_info:
                train_data.append(
                    {
                        "infer_info": infer_info,
                        "advantages": rl_data["advantages"],
                        "id": uuid.uuid4().hex,
                    }
                )
            else:
                get_logger().warning(f"Skipping training data due to missing inference information: {rl_data}")
        batch = train_data
        return await self.train(batch, step)

    async def train(self, batch: list[RLInput], step):
        results = await self.actor.step(batch, step)
        return results[0]

    async def serve(self):
        await self.actor.launch_server()
        self.url = await self.actor.get_server_ip()
        return self.url

    async def pause_serve(self):
        infer_tokens = await self.actor.pause_server()  # rank 3
        infer_tokens = [sum(rank_tokens[i] for rank_tokens in infer_tokens) for i in range(len(infer_tokens[0]))]
        infer_infos = await self.actor.fetch_infer_info()
        for infer_info_dict in infer_infos:
            for key, value in infer_info_dict.items():
                self.infer_bank.add_raw(key, value)
        return infer_tokens

    async def stop_serve(self):
        await self.actor.stop_server()

    async def resume(self, folder):
        await self.actor.resume(folder)

    async def checkpoint(self, folder):
        await self.actor.checkpoint(folder)

    async def get_server_ip(self):
        return (await self.actor.get_server_ip())[0]

    async def shutdown(self):
        self.actor.shutdown()
        del self.actor
        self.infer_bank.bank.clear()
        gc.collect()
        torch.cuda.empty_cache()

    async def set_sft_data(
        self,
        batch: list[SFTData],
        jsonl_sources: list,
        chat_template: Callable,
    ):
        await self.actor.set_sft_data(batch, jsonl_sources, chat_template)

    async def step_sft(self, num_micro_steps: int) -> dict[str, float]:
        # The SPMD actor returns one result per worker rank; return rank 0's,
        # mirroring self.train() above (all ranks share the globally-reduced loss).
        results = await self.actor.step_sft(num_micro_steps)
        return results[0]
