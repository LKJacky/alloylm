import datetime
import os

import torch

from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.rl.rl_config import UnifiedConfig, create_trainer
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.impl.math import GSM8KDatasetConfig

INFER_LENGTH = 4096
NUM_WORKERS = int(os.environ.get("GPU", "4"))
SP_SIZE = 2 if NUM_WORKERS % 2 == 0 else 1


def get_trainer():
    trainer = create_trainer(
        UnifiedConfig(
            llm_config=AlloyLMModelConfig(
                path="Qwen/Qwen2.5-0.5B-Instruct",
                model_cls=FSDPQwen2ForCausalLM,
                fsdp_config=FSDPConfig(
                    train_mesh={
                        "mesh_shape": (NUM_WORKERS // SP_SIZE, SP_SIZE),
                        "mesh_dim_names": ["dp", "sp"],
                        "device_type": "cuda",
                    },
                    infer_mesh={
                        "mesh_shape": (NUM_WORKERS, 1),
                        "mesh_dim_names": ["dp", "tp"],
                        "device_type": "cuda",
                    },
                    lm_head_dtype=torch.bfloat16,
                    shard_dtype=torch.bfloat16,
                ),
            ),
            max_length_rollout=2048 + INFER_LENGTH,
            max_length_train=8192,
            # data
            train_datasets=[
                GSM8KDatasetConfig(
                    name="gsm8k_1",
                    split="train",
                    infer_args=InferArgs(
                        sample_args={
                            "top_p": 1.0,
                            "temperature": 1.0,
                            "max_tokens": INFER_LENGTH,
                            "extra_body": {
                                "for_training": True,
                            },
                        }
                    ),
                )
            ],
            train_sample_ratios=[1.0],
            eval_datasets=[
                GSM8KDatasetConfig(
                    name="gsm8k_1",
                    split="test",
                    infer_args=InferArgs(sample_args={"top_p": 0.6, "temperature": 1.0, "max_tokens": INFER_LENGTH}),
                )
            ],
            eval_sample_ratios=[1.0],
            # rl algo
            roll_out_bs=32,
            num_rl_group=8,
            num_optimize_per_step=1,
            filter_group="discard",
            async_rollout="group",
            # pipeline
            total_training_steps=2000,
            checkpoint_interval=25,
            max_checkpoints=1,
            eval_interval=25,
            auto_resume=True,
            # infra
            work_dir=f"work_dirs/tests/debug_gsm8k/{datetime.datetime.now(tz=datetime.UTC).strftime('%Y-%m-%d')}_{os.environ.get('WORKER_NAME', 'default')}",
            num_workers=NUM_WORKERS,
            max_concurrency_per_node=512,
            cache_max_entry_count=0.4,
            max_prefill_length=8192,
            sp_size=SP_SIZE,
            sampler_batch_size=16,
        )
    )
    return trainer
