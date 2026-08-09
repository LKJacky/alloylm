import datetime
import os

from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.rl.rl_config import UnifiedConfig, create_trainer
from alloylm.engine.train_engine.hack_client import (
    HighConcurrentClient as AsyncClient,
)
from alloylm.impl import common
from alloylm.impl.math import GSM8KDatasetConfig

common.AsyncClient = AsyncClient


INFER_LENGTH = 4096


def get_trainer():
    trainer = create_trainer(
        UnifiedConfig(
            llm="Qwen/Qwen2.5-0.5B-Instruct",
            max_length_rollout=2048 + INFER_LENGTH,
            max_length_train=8192,
            # data
            train_datasets=[
                GSM8KDatasetConfig(
                    name="gsm8k_1",
                    split="train",
                    infer_args=InferArgs(sample_args={"top_p": 1.0, "temperature": 1.0, "max_tokens": INFER_LENGTH}),
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
            num_workers=int(os.environ.get("GPU", "4")),
            max_concurrency_per_node=512,
            cache_max_entry_count=0.8,
            max_prefill_length=8192,
            sp_size=min(int(os.environ.get("GPU", "4")), 2),
        )
    )
    return trainer
