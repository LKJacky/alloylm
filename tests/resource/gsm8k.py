import datetime
import os

from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.rl.rl_algo import RLAlgorithmConfig
from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.train_engine import TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import TrainInferEngineConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.impl.math import GSM8KDatasetConfig

INFER_LENGTH = 8192
NUM_WORKERS = int(os.environ.get("GPU", "4"))
SP_SIZE = min(2, NUM_WORKERS)
WORK_DIR = f"work_dirs/tests/debug_gsm8k/{datetime.datetime.now(tz=datetime.UTC).strftime('%Y-%m-%d')}_{os.environ.get('WORKER_NAME', 'default')}"

config = RLAlgorithmConfig(
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
        ),
    ),
    engine_config=TrainInferEngineConfig(  # default value for sft
        train_config=TrainEngineConfig(
            max_length=INFER_LENGTH,
            work_dir=str(WORK_DIR),
            num_workers=NUM_WORKERS,
            total_training_steps=2000,
            num_optimize_per_step=1,
            lr=1e-6,
            wd=0.1,
            scheduler_type="constant",
            warmup_ratio=0.03,
            gpu_peak_tflops=148,  # for H20
            chunk_loss_size=512,
        ),
        infer_config=InferEngineConfig(
            model_name="ALLOYLM",
            max_prefill_length=INFER_LENGTH - 1024,
            # chat_template=Qwen3ChatTemplate,
            # tool_pattern=QWEN_TOOL_PATTERN,
            # thinking_pattern=QWEN_THINKING_PATTERN,
            memory_usage=0.8,
            sampler_batch_size=64,
        ),
    ),
    datasets=[
        GSM8KDatasetConfig(
            name="gsm8k_1",
            split="train",
            infer_args=InferArgs(
                sample_args={
                    "top_p": 1.0,
                    "temperature": 1.0,
                    "max_tokens": INFER_LENGTH - 1024,
                    "extra_body": {
                        "for_training": True,
                        "max_entropy": 4.0,
                        # "chat_template_kwargs": {"thinking": True},
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
            infer_args=InferArgs(
                sample_args={
                    "top_p": 0.6,
                    "temperature": 1.0,
                    "max_tokens": INFER_LENGTH - 1024,
                    # "extra_body": {"chat_template_kwargs": {"thinking": True}},
                }
            ),
        )
    ],
    eval_sample_ratio=[1.0],
    # rl algo
    roll_out_bs=128,
    num_rl_group=8,
    max_length=INFER_LENGTH,
    filter_group="discard",
    async_rollout="group",
    # pipeline
    total_training_steps=2000,
    checkpoint_interval=100,
    max_checkpoints=1,
    eval_interval=100,
    auto_resume=True,
    # infra
    work_dir=WORK_DIR,
    max_concurrency=128 * NUM_WORKERS,
)
