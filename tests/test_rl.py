import copy
import os
import shutil
import unittest

import torch
from torch import distributed as dist

from alloylm.algorithm.base import InferArgs, TaskData
from alloylm.algorithm.rl.rl_config import UnifiedConfig, create_trainer
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.train_infer_engine import (
    AsyncClient as RLAsyncClient,
)
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl import common
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.impl.math import GSM8KDatasetConfig
from alloylm.test_utils import CudaAsyncTestCase

default_config = UnifiedConfig(
    llm_config=AlloyLMModelConfig(
        path="Qwen/Qwen2.5-0.5B-Instruct",
        model_cls=FSDPQwen2ForCausalLM,
        fsdp_config=FSDPConfig(
            train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
            infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
        ),
    ),
    max_length_rollout=1024,
    max_length_train=1024,
    # data
    train_datasets=[
        GSM8KDatasetConfig(
            name="gsm8k_1",
            split="train",
            infer_args=InferArgs(sample_args={"top_p": 1.0, "temperature": 1.0, "max_tokens": 512}),
        ),
        GSM8KDatasetConfig(
            name="gsm8k_2",
            split="train",
            infer_args=InferArgs(sample_args={"top_p": 1.0, "temperature": 1.0, "max_tokens": 512}),
        ),
    ],
    train_sample_ratios=[1.0, 1.0],
    eval_datasets=[
        GSM8KDatasetConfig(
            name="gsm8k_1",
            split="test",
            infer_args=InferArgs(sample_args={"top_p": 1.0, "temperature": 1.0, "max_tokens": 512}),
        ),
        GSM8KDatasetConfig(
            name="gsm8k_2",
            split="test",
            infer_args=InferArgs(sample_args={"top_p": 1.0, "temperature": 1.0, "max_tokens": 512}),
        ),
    ],
    eval_sample_ratios=[0.001, 0.001],
    # rl algo
    roll_out_bs=32,
    num_rl_group=4,
    num_optimize_per_step=1,
    filter_group="none",
    async_rollout="none",
    # pipeline
    total_training_steps=8,
    checkpoint_interval=4,
    max_checkpoints=1,
    eval_interval=4,
    auto_resume=False,
    # infra
    work_dir="work_dirs/tests/rl/",
    num_workers=1,
    max_concurrency_per_node=128,
    cache_max_entry_count=0.2,
    max_prefill_length=1024,
    sp_size=1,
)


class RLTest(CudaAsyncTestCase):
    default_config = default_config

    @classmethod
    def setUpClass(cls):
        cls._original_async_client = common.AsyncClient
        common.AsyncClient = RLAsyncClient

    @classmethod
    def tearDownClass(cls):
        common.AsyncClient = cls._original_async_client


@unittest.skipUnless(os.environ.get("ENABLE_LONG_RUNNING_TESTS", "0") == "1", "Skipping long-runing test")
class TestRLSystem(RLTest):
    default_config = default_config

    async def test_sync_no_filter(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        config.work_dir = "work_dirs/tests/rl"
        trainer = create_trainer(config)
        await trainer.lazy_init()
        await trainer.fit()

    async def test_async_no_filter(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        config.work_dir = "work_dirs/tests/rl"
        config.async_rollout = "task"
        trainer = create_trainer(config)
        await trainer.lazy_init()
        await trainer.fit()
        del trainer

    async def test_async_no_filter_entropy(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        config.work_dir = "work_dirs/tests/rl"
        config.async_rollout = "task"
        config.filter_group = "resubmit"
        for dataset in config.train_datasets:
            dataset.infer_args.sample_args["extra_body"] = {"max_entropy": 3}

        trainer = create_trainer(config)
        await trainer.lazy_init()
        await trainer.fit()


class TestRLSystemQuick(RLTest):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    async def test_rl(self):
        function_used = {"loss_func_used": False, "data_post_process_used": False, "step_data_process_used": False}

        def loss_func(
            logits: torch.Tensor,
            labels: torch.Tensor,
            old_logprobs: torch.Tensor,
            advantages: torch.Tensor,
            loss_weight: torch.Tensor,
            policy_loss_cfg: dict,
            function_used=function_used,
        ) -> torch.Tensor:
            function_used["loss_func_used"] = True

            assert (labels >= 0).all(), "Labels must be non-negative for loss computation"
            log_probs = logits.log_softmax(dim=-1)
            gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            log_diff = torch.clamp(gathered - old_logprobs, -20, 20)
            ratio = log_diff.exp()
            pg_loss1 = -ratio * advantages
            pg_loss2 = (
                -ratio.clamp(1 - policy_loss_cfg["cliprange_low"], 1 + policy_loss_cfg["cliprange_high"]) * advantages
            )
            clip_pg_loss1 = torch.max(pg_loss1, pg_loss2)
            # Dual-clip PPO
            pg_loss3 = -advantages * policy_loss_cfg["cliprange_c"]
            clip_pg_loss2 = torch.min(pg_loss3, clip_pg_loss1)
            pg_losses = torch.where(advantages < 0, clip_pg_loss2, clip_pg_loss1)
            pg_loss = torch.sum(pg_losses) * loss_weight
            with torch.no_grad():
                entropy = -(logits.softmax(dim=-1) * log_probs).sum(dim=-1)
            return pg_loss, entropy

        def data_post_process(batch: list[TaskData], function_used=function_used):
            function_used["data_post_process_used"] = True
            batch = [x for x in batch if x.others["rl_data"]["advantages"] != 0]
            return batch

        def step_data_process(batch, function_used=function_used):
            function_used["step_data_process_used"] = True
            return batch

        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(default_config)
        config.total_training_steps = 2
        config.checkpoint_interval = 1
        config.roll_out_bs = 2
        config.num_rl_group = 8
        config.filter_group = "none"
        config.loss_func = loss_func
        config.data_post_process_func = data_post_process
        config.step_data_process_func = step_data_process
        trainer = create_trainer(config)
        await trainer.lazy_init()
        await trainer.fit()
        for key, used in function_used.items():
            self.assertTrue(used, f"{key} was not used during training")

        resumed = await trainer.resume()
        self.assertTrue(resumed, "Failed to resume from checkpoint")

    async def test_rl_with_fake_data(self):
        try:
            os.environ["USE_FAKE_DATA"] = "1"
            await self.test_rl()
        finally:
            os.environ.pop("USE_FAKE_DATA", None)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    async def test_rl_gpu2(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(default_config)
        config.total_training_steps = 2
        config.checkpoint_interval = 1
        config.roll_out_bs = 2
        config.num_rl_group = 8
        config.filter_group = "none"
        config.num_workers = 2
        config.sp_size = 2
        config.work_dir = "work_dirs/tests/rl"
        trainer = create_trainer(config)
        await trainer.lazy_init()
        await trainer.fit()
