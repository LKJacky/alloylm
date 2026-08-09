import asyncio

import torch
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.scheduler import InferItem, SchedulerServer
from alloylm.engine.infer_engine.utils import GeneConfig
from alloylm.engine.spmd import init_dist
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM
from alloylm.test_utils import CudaAsyncTestCase, collect_garbage


class TestScheduler(CudaAsyncTestCase):
    async def test_scheduler(self):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        init_dist()
        model = FSDPQwen2ForCausalLM.from_pretrained(
            model_path,
            fsdp_config=FSDPConfig(
                train_mesh={"device_type": "cuda", "mesh_shape": (1, 1), "mesh_dim_names": ["fsdp", "sp"]},
                infer_mesh={"device_type": "cuda", "mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"]},
                lm_head_dtype=torch.bfloat16,
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cache = model.create_cache(memory_usage=0.6)
        scheduler = SchedulerServer(
            model=model,
            cache=cache,
            max_prefill_length=1024,
            real_vocab_size=model.get_real_vocab_size(tokenizer),
        )

        qa_pairs = [
            ("What is the capital of France? Answer with one word.", "paris"),
            ("What is the capital of China? Answer with one word.", "beijing"),
        ]
        sessions = []
        for session_id, (question, _) in enumerate(qa_pairs):
            input_ids = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                .input_ids.flatten()
                .tolist()
            )
            sessions.append(
                InferItem(
                    session_id=session_id,
                    input_ids=input_ids,
                    gene_config=GeneConfig(
                        top_k=1,
                        stop_token=[tokenizer.eos_token_id],
                        total_max_length=32,
                        release_at_once=True,
                    ),
                )
            )

        try:
            await scheduler.launch()
            baseline_cache_usage = cache.cache_usage()

            for session in sessions:
                await scheduler.wait_queue.put(session)

            await asyncio.wait_for(
                asyncio.gather(*(session.finished_event.wait() for session in sessions)),
                timeout=120,
            )

            for session, (_, expected_answer) in zip(sessions, qa_pairs):
                result = session._result
                output = tokenizer.decode(result["item"]["tokens"], skip_special_tokens=True)
                self.assertIn(result["reason"], {"stop", "length"})
                self.assertEqual(result["item"]["finish_reason"], result["reason"])
                self.assertIn(expected_answer, output.lower())
                self.assertGreater(result["item"]["usage"]["output_tokens"], 0)
                self.assertFalse(session.device_session.on_device())

            self.assertEqual(cache.cache_usage(), baseline_cache_usage)
        finally:
            await scheduler.stop_server()
            model.train_shard()
            model.to_empty(device="cpu")
            del scheduler, cache, tokenizer, model
            collect_garbage()
