import asyncio
import gc
import os
import shutil
import tempfile
import time
import unittest
from unittest import IsolatedAsyncioTestCase

import torch
from huggingface_hub import constants
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.engine import (
    AlloyLMModelConfig,
    InferEngineConfig,
    SPMDInfer,
    SPMDInferConfig,
)
from alloylm.engine.spmd import SPMDActor, SPMDActorConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM

constants.HF_HUB_OFFLINE = True
os.environ["HF_HUB_OFFLINE"] = "1"


class TmpFolder:
    def __init__(self, folder_name=None):
        self.folder_name = folder_name
        self.tmp_folder = None

    def __enter__(self):
        if self.folder_name:
            self.tmp_folder = os.path.join("work_dirs/tests/", self.folder_name)
            os.makedirs(self.tmp_folder, exist_ok=True)
        else:
            self.tmp_folder = tempfile.mkdtemp()
        return self.tmp_folder

    def __exit__(self, exc_type, exc_val, exc_tb):
        while os.path.exists(self.tmp_folder):
            try:
                shutil.rmtree(self.tmp_folder)
            except Exception:  # noqa
                pass


class FakeChatTemplate:
    def render(self, messages, add_generation_prompt=False, **kwargs):
        text = [f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" for msg in messages]
        text = "\n".join(text)
        if add_generation_prompt:
            text += "\n<|im_start|>assistant\n"
        return text


class LaunchTestServer:
    def __init__(
        self,
        port=8000,
        name="ALLOYLM",
        max_prefill_length=1024,
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        shard_dtype=torch.bfloat16,
        tool_pattern=None,
        chat_template=None,
    ):
        self.engine = SPMDActor.create_spmd_actor(
            SPMDInfer,
            args=(
                SPMDInferConfig(
                    llm_config=AlloyLMModelConfig(
                        path=model_path,
                        model_cls=FSDPQwen2ForCausalLM,
                        fsdp_config=FSDPConfig(
                            train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
                            infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
                            shard_dtype=shard_dtype,
                            lm_head_dtype=torch.bfloat16,
                        ),
                    ),
                    infer_engine_config=InferEngineConfig(
                        model_name=name,
                        memory_usage=0.6,
                        max_prefill_length=max_prefill_length,
                        port=port,
                        tool_pattern=tool_pattern,
                        chat_template=chat_template,
                    ),
                ),
            ),
            spmd_config=SPMDActorConfig(world_size=1),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    async def __aenter__(self):
        await self.engine.launch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.engine.stop()

    def __del__(self):
        try:
            self.engine.shutdown()
            del self.engine
            gc.collect()
            torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa
            pass


def collect_garbage():
    gc.collect()
    torch.cuda.empty_cache()


def check_cuda_leak():
    freed = False
    for i in range(5):
        collect_garbage()
        used_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        if used_memory > 0.5:
            print(f"GPU memory leak detected: {used_memory:.2f} GB used, retrying...")
            time.sleep(5)
        else:
            freed = True
            break
    if not freed:
        raise RuntimeError(f"GPU memory leak detected: {used_memory:.2f} GB used after cleanup attempts.")


@unittest.skipIf(torch.cuda.memory_allocated() / (1024 * 1024 * 1024) > 0.1, "GPU memory is not clean before test")
class CudaAsyncTestCase(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        collect_garbage()

    async def asyncSetUp(self):
        asyncio.get_running_loop().set_debug(False)

    @classmethod
    def tearDownClass(cls):
        collect_garbage()
        check_cuda_leak()
        if dist.is_initialized():
            dist.destroy_process_group()
