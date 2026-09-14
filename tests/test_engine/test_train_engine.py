import asyncio
import gc
import math
import shutil
import tempfile
import unittest

import torch
from openai import AsyncClient

from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.dataset import task_collate_fn
from alloylm.engine.train_engine.train_engine import TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import (
    RLInput,
    SpmdTrainInferEngine,
    TrainInferEngineConfig,
)
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.test_utils import CudaAsyncTestCase

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


class TaskCollateFnTest(unittest.TestCase):
    def test_sequence_parallel_shards_are_aligned(self):
        batch = [
            [
                {
                    "id": "a",
                    "input_ids": [1, 2, 3],
                    "labels": [-100, 2, 3],
                    "num_tokens": 3,
                    "advantages": 1.0,
                    "log_probs": [0.1, 0.2, 0.3],
                    "train_entropy": [0.4, 0.5, 0.6],
                },
                {
                    "id": "b",
                    "input_ids": [4, 5],
                    "labels": [4, 5],
                    "num_tokens": 2,
                    "advantages": -1.0,
                    "log_probs": [0.7, 0.8],
                    "train_entropy": [0.9, 1.0],
                },
            ]
        ]

        shards = [task_collate_fn(batch, sp_size=2, sp_rank=rank) for rank in range(2)]

        for key, expected in {
            "input_ids": [1, 2, 3, 4, 5, 0],
            "labels": [2, 3, -100, 5, -100, -100],
            "position_ids": [0, 1, 2, 0, 1, 0],
            "advantages": [1.0, 1.0, 1.0, -1.0, -1.0, 0.0],
        }.items():
            actual = torch.cat([shard[key] for shard in shards], dim=1).flatten()
            self.assertTrue(torch.equal(actual, torch.tensor(expected, dtype=actual.dtype)))

        self.assertTrue(torch.equal(shards[0]["seq_lens"], torch.tensor([3, 2, 1])))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TrainInferEngineTest(CudaAsyncTestCase):
    async def test_infer_then_train(self):
        work_dir = tempfile.mkdtemp()
        engine = None
        client = None

        try:
            fsdp_config = FSDPConfig(
                train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
                infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
                shard_dtype=torch.bfloat16,
                lm_head_dtype=torch.bfloat16,
            )
            engine = SpmdTrainInferEngine(
                model_config=AlloyLMModelConfig(
                    path=MODEL_PATH,
                    model_cls=FSDPQwen2ForCausalLM,
                    fsdp_config=fsdp_config,
                ),
                engine_config=TrainInferEngineConfig(
                    train_config=TrainEngineConfig(
                        max_length=1024,
                        work_dir=work_dir,
                        num_workers=1,
                        total_training_steps=4,
                    ),
                    infer_config=InferEngineConfig(model_name="ALLOYLM", memory_usage=0.2),
                ),
            )
            await engine.lazy_init()

            questions = (
                (
                    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                    "How many clips did Natalia sell altogether in April and May?"
                ),
                (
                    "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. "
                    "How much did she earn?"
                ),
            )
            system_prompt = "Answer below question and response your final answer in \\boxed"

            # Match the RL test's workload: two prompts, eight stochastic rollouts each,
            # long generations, and repeated infer/train transitions.
            for step in range(4):
                await engine.serve()
                client = AsyncClient(api_key="EMPTY", base_url=await engine.get_server_ip())
                prompts = [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                    for question in questions
                    for _ in range(8)
                ]
                responses = await asyncio.gather(
                    *(
                        client.chat.completions.create(
                            model="ALLOYLM",
                            messages=prompt,
                            max_completion_tokens=512,
                            temperature=1.0,
                            top_p=1.0,
                            extra_body={"for_training": True},
                        )
                        for prompt in prompts
                    )
                )
                self.assertTrue(all(response.usage.completion_tokens > 0 for response in responses))

                train_batch = []
                for index, (prompt, response) in enumerate(zip(prompts, responses)):
                    messages = prompt + [{"role": "assistant", "content": response.choices[0].message.content}]
                    train_batch.append(
                        RLInput(
                            messages=messages,
                            advantages=-1.0 if index % 8 < 4 else 1.0,
                        )
                    )
                await client.close()
                client = None
                await engine.pause_serve()

                result = await engine.train_wrapper(train_batch, step)

                completion_tokens = sum(response.usage.completion_tokens for response in responses)
                self.assertEqual(result["logprob_diff/num_tokens"], completion_tokens)
                self.assertTrue(math.isfinite(result["logprob_diff/avg_diff"]))
        finally:
            if client is not None:
                await client.close()
            if engine is not None:
                await engine.stop_serve()
                await engine.shutdown()
                del engine
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(work_dir, ignore_errors=True)
