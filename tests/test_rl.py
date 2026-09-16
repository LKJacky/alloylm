import copy
import os
import shutil
import unittest

import torch
from torch import distributed as dist

from alloylm.algorithm.base import InferArgs, TaskData
from alloylm.algorithm.rl.rl_config import (
    InferEngineConfig,
    RLAlgorithmConfig,
    RLTrainer,
    TrainEngineConfig,
    TrainInferEngineConfig,
)
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.count_to_n import CountToNDatasetConfig
from alloylm.impl.engines.qwen.chat_template import (
    QWEN_THINKING_PATTERN,
    QWEN_TOOL_PATTERN,
    Qwen3ChatTemplate,
)
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.impl.math import GSM8KDatasetConfig
from alloylm.test_utils import CudaAsyncTestCase, collect_garbage

TRAIN_INFER_ARGS = InferArgs(
    sample_args={
        "top_p": 1.0,
        "temperature": 1.0,
        "max_tokens": 512,
        "extra_body": {"for_training": True},
    }
)
EVAL_INFER_ARGS = InferArgs(sample_args={"top_p": 1.0, "temperature": 1.0, "max_tokens": 512})

default_config = RLAlgorithmConfig(
    llm_config=AlloyLMModelConfig(
        path="Qwen/Qwen3-0.6B",
        model_cls=FSDPQwen2ForCausalLM,
        fsdp_config=FSDPConfig(
            train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
            infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
            lm_head_dtype=torch.bfloat16,
            shard_dtype=torch.bfloat16,
        ),
    ),
    engine_config=TrainInferEngineConfig(
        train_config=TrainEngineConfig(
            max_length=4096,
            work_dir="work_dirs/tests/rl/",
            num_workers=1,
            total_training_steps=5,
            lr=8e-6,
            wd=0,
            scheduler_type="cosine",
            warmup_ratio=0.03,
            chunk_loss_size=512,
        ),
        infer_config=InferEngineConfig(
            model_name="ALLOYLM",
            max_prefill_length=8192,
            chat_template=Qwen3ChatTemplate,
            tool_pattern=QWEN_TOOL_PATTERN,
            thinking_pattern=QWEN_THINKING_PATTERN,
            memory_usage=0.7,
        ),
    ),
    datasets=[
        GSM8KDatasetConfig(name="gsm8k_1", split="train", infer_args=TRAIN_INFER_ARGS),
        GSM8KDatasetConfig(name="gsm8k_2", split="train", infer_args=TRAIN_INFER_ARGS),
    ],
    train_sample_ratios=[1.0, 1.0],
    eval_datasets=[
        GSM8KDatasetConfig(name="gsm8k_1", split="test", infer_args=EVAL_INFER_ARGS),
        GSM8KDatasetConfig(name="gsm8k_2", split="test", infer_args=EVAL_INFER_ARGS),
    ],
    eval_sample_ratio=[0.001, 0.001],
    eval_interval=100,
    roll_out_bs=4,
    num_rl_group=4,
    max_length=4096,
    total_training_steps=5,
    checkpoint_interval=100,
    max_checkpoints=1,
    auto_resume=False,
    async_rollout="task",
    filter_group="resubmit",
    data_post_process_func=None,
    work_dir="work_dirs/tests/rl/",
    max_concurrency=64,
)


class RLTest(CudaAsyncTestCase):
    default_config = default_config


@unittest.skipUnless(os.environ.get("ENABLE_LONG_RUNNING_TESTS", "0") == "1", "Skipping long-runing test")
class TestRLSystem(RLTest):
    default_config = default_config

    async def test_sync_no_filter(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        config.work_dir = "work_dirs/tests/rl"
        trainer = RLTrainer(config)
        await trainer.lazy_init()
        await trainer.fit()

    async def test_async_no_filter_entropy(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        config.work_dir = "work_dirs/tests/rl"
        config.async_rollout = "task"
        config.filter_group = "resubmit"
        for dataset in config.datasets:
            dataset.infer_args.sample_args["extra_body"] = {"max_entropy": 3}

        trainer = RLTrainer(config)
        await trainer.lazy_init()
        await trainer.fit()


class TestRLSystemQuick(RLTest):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    async def test_simple_rl(self):
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
        config.engine_config.train_config.total_training_steps = config.total_training_steps = 2
        config.checkpoint_interval = 1
        config.roll_out_bs = 2
        config.num_rl_group = 8
        config.filter_group = "none"
        config.engine_config.train_config.loss_func = loss_func
        config.data_post_process_func = data_post_process
        config.engine_config.train_config.step_data_process_func = step_data_process
        trainer = RLTrainer(config)
        await trainer.lazy_init()
        await trainer.fit()
        for key, used in function_used.items():
            self.assertTrue(used, f"{key} was not used during training")

        resumed = await trainer.resume()
        self.assertTrue(resumed, "Failed to resume from checkpoint")

        del trainer
        collect_garbage()

    async def test_async_rl(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        config.work_dir = "work_dirs/tests/rl"
        config.engine_config.train_config.total_training_steps = config.total_training_steps = 4
        config.roll_out_bs = 4
        config.async_rollout = "task"
        trainer = RLTrainer(config)
        await trainer.lazy_init()
        await trainer.fit()
        del trainer

    async def test_agentic_async_rl(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(self.default_config)
        train_infer_args, eval_infer_args = copy.deepcopy((TRAIN_INFER_ARGS, EVAL_INFER_ARGS))
        for infer_args in (train_infer_args, eval_infer_args):
            infer_args.interactive_mode = True
            chat_template_kwargs = infer_args.sample_args.setdefault("extra_body", {}).setdefault(
                "chat_template_kwargs", {}
            )
            chat_template_kwargs["thinking"] = True

        config.datasets = [CountToNDatasetConfig(max_target=32, infer_args=train_infer_args)]
        config.train_sample_ratios = [1.0]
        config.eval_datasets = [CountToNDatasetConfig(max_target=32, infer_args=eval_infer_args)]
        config.eval_sample_ratio = [0.1]
        config.work_dir = "work_dirs/tests/rl"
        config.engine_config.train_config.total_training_steps = config.total_training_steps = 4
        config.roll_out_bs = 2
        config.num_rl_group = 2
        config.async_rollout = "task"
        trainer = RLTrainer(config)
        await trainer.lazy_init()
        await trainer.fit()
        await trainer.model_engine.shutdown()
        del trainer

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires at least 2 GPUs")
    async def test_rl_gpu2(self):
        shutil.rmtree("work_dirs/tests/rl", ignore_errors=True)
        config = copy.deepcopy(default_config)
        config.llm_config.fsdp_config.train_mesh["mesh_shape"] = (1, 2)
        config.total_training_steps = config.engine_config.train_config.total_training_steps = 2
        config.checkpoint_interval = 1
        config.roll_out_bs = 2
        config.num_rl_group = 8
        config.filter_group = "none"
        config.engine_config.train_config.num_workers = 2
        config.work_dir = "work_dirs/tests/rl"
        trainer = RLTrainer(config)
        await trainer.lazy_init()
        await trainer.fit()
