import asyncio
import gc
import math
import shutil
import tempfile
import unittest

import torch

from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.train_engine import TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import (
    AsyncClient,
    SpmdTrainInferEngine,
    TrainInferEngineConfig,
)
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.test_utils import CudaAsyncTestCase

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TrainInferEngineTest(CudaAsyncTestCase):
    async def test_infer_then_train(self):
        work_dir = tempfile.mkdtemp()
        engine = None
        client = None
        serving = False

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
                        max_length=128,
                        work_dir=work_dir,
                        num_workers=1,
                        total_training_steps=10,
                    ),
                    infer_config=InferEngineConfig(model_name="ALLOYLM", memory_usage=0.1),
                ),
            )
            await engine.lazy_init()

            for step in range(10):
                await engine.serve()
                serving = True
                client = AsyncClient(api_key="EMPTY", base_url=await engine.get_server_ip())
                prompts = [[{"role": "user", "content": "What is 2 + 2?"}] for _ in range(2)]
                responses = await asyncio.gather(
                    *(
                        client.chat.completions.create(
                            model="ALLOYLM",
                            messages=prompt,
                            max_completion_tokens=8,
                            temperature=0,
                            extra_body={"top_k": 1, "ignore_eos": True},
                        )
                        for prompt in prompts
                    )
                )
                self.assertTrue(all(response.usage.completion_tokens > 0 for response in responses))

                rollouts = []
                for prompt, response in zip(prompts, responses):
                    messages = prompt + [{"role": "assistant", "content": response.choices[0].message.content}]
                    rollouts.append(AsyncClient.retrieve_collected_tokens(messages))
                await client.close()
                client = None
                await engine.stop_serve()
                serving = False

                result = await engine.train(
                    [
                        {
                            "input_ids": rollout["input_ids"],
                            "labels": rollout["labels"],
                            "inference_logprobs": rollout["log_probs"],
                            "advantages": 1.0,
                        }
                        for rollout in rollouts
                    ],
                    step=step,
                )

                completion_tokens = sum(response.usage.completion_tokens for response in responses)
                self.assertEqual(result["logprob_diff/num_tokens"], completion_tokens)
                self.assertTrue(math.isfinite(result["logprob_diff/avg_diff"]))
        finally:
            if client is not None:
                await client.close()
            if serving and engine is not None:
                await engine.stop_serve()
            if engine is not None and hasattr(engine, "actor"):
                engine.actor.shutdown()
                del engine.actor
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(work_dir, ignore_errors=True)
