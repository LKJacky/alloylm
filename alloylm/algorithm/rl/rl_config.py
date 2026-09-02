import os
from typing import Any

from pydantic import BaseModel

from alloylm.algorithm.base import DatasetConfig
from alloylm.algorithm.rl import rl_algo
from alloylm.algorithm.rl.rl_algo import RLAlgorithmConfig, RLTrainer
from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.train_engine import TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import (
    TrainInferEngineConfig,
)

DEFAULT_MODEL_NAME = "ALLOYLM"


class UnifiedConfig(BaseModel):
    llm_config: AlloyLMModelConfig
    tokenizer: str | None = None
    max_length_rollout: int = 32 * 1024
    max_length_train: int = 34 * 1024

    # data
    train_datasets: list[DatasetConfig | Any] = []
    train_sample_ratios: list[float] = []
    eval_datasets: list[DatasetConfig | Any] = []
    eval_sample_ratios: list[float | int] = []
    # rl algo
    lr: float = 1e-6
    roll_out_bs: int = 128
    num_rl_group: int = 8
    num_optimize_per_step: int = 1
    filter_group: str = "discard"
    async_rollout: str = "group"
    clip_higth: float = 0.2
    clip_low: float = 0.2
    loss_func: object = None
    data_post_process_func: object = None  # batch level
    step_data_process_func: object = None  # step level

    # pipeline
    total_training_steps: int = 4000
    checkpoint_interval: int = 10
    max_checkpoints: int = 1
    eval_interval: int = 10
    auto_resume: bool = True
    # infra
    work_dir: str = "work_dirs/debug"
    num_workers: int = 1
    max_concurrency_per_node: int = 128
    cache_max_entry_count: float = 0.8
    recompute_ratio: float = 0.0
    train_engine_version: float = float(os.environ.get("TRAIN_ENGINE_VERSION", "0.0"))
    chunk_loss_size: int = 512
    sp_size: int = 1
    freeze_routers: bool = True
    max_prefill_length: int = 8 * 1024


def create_trainer(
    config: UnifiedConfig,
):

    rl_config = RLAlgorithmConfig(
        llm_config=config.llm_config,
        engine_config=TrainInferEngineConfig(
            train_config=TrainEngineConfig(
                max_length=config.max_length_train,
                work_dir=config.work_dir,
                num_workers=config.num_workers,
                total_training_steps=config.total_training_steps,
                num_optimize_per_step=config.num_optimize_per_step,
                lr=config.lr,
                h_clip=config.clip_higth,
                l_clip=config.clip_low,
                loss_func=config.loss_func,
                chunk_loss_size=config.chunk_loss_size,
                sp_size=config.sp_size,
                step_data_process_func=config.step_data_process_func,
            ),
            infer_config=InferEngineConfig(
                model_name=DEFAULT_MODEL_NAME,
                max_prefill_length=config.max_prefill_length,
                memory_usage=config.cache_max_entry_count,
            ),
        ),
        datasets=config.train_datasets,
        train_sample_ratios=config.train_sample_ratios,
        eval_datasets=config.eval_datasets,
        eval_sample_ratio=config.eval_sample_ratios,
        eval_interval=config.eval_interval,
        roll_out_bs=config.roll_out_bs,
        num_rl_group=config.num_rl_group,
        max_length=config.max_length_rollout,
        total_training_steps=config.total_training_steps,
        filter_group=config.filter_group,
        async_rollout=config.async_rollout,
        work_dir=config.work_dir,
        max_concurrency=config.max_concurrency_per_node * config.num_workers,
        data_post_process_func=config.data_post_process_func,
        checkpoint_interval=config.checkpoint_interval,
        max_checkpoints=config.max_checkpoints,
        auto_resume=config.auto_resume,
    )

    from alloylm.engine.train_engine.train_infer_engine import AsyncClient

    rl_algo.AsyncClient = AsyncClient
    return RLTrainer(
        rl_config,
    )
