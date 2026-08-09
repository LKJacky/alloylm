import asyncio
import json
import os
import random
import unittest
import uuid

from huggingface_hub import try_to_load_from_cache

from alloylm.engine.infer_engine.utils import get_logger
from alloylm.server.client import HighConcurrentClient as AsyncClient
from alloylm.test_utils import CudaAsyncTestCase, LaunchTestServer


def _hf_model_cached(model_id: str) -> bool:
    """Return True only if the model weights (safetensors/bin files) are in the
    HF cache."""
    index = try_to_load_from_cache(model_id, "model.safetensors.index.json")
    if index is None or index == "model.safetensors.index.json":
        # Try single-file model
        shard = try_to_load_from_cache(model_id, "model.safetensors")
        return shard is not None and shard != "model.safetensors" and os.path.isfile(shard)
    # Multi-shard model: verify the first shard file exists
    try:
        with open(index) as f:
            data = json.load(f)
        first_shard = next(iter(data.get("weight_map", {}).values()))
        path = try_to_load_from_cache(model_id, first_shard)
        return path is not None and path != first_shard and os.path.isfile(path)
    except Exception:
        return False


logger = get_logger()


class TestLaunchSystem(CudaAsyncTestCase):
    async def try_forward_interactive(self, port=8000):
        all_prompts = [
            [
                (
                    {
                        "role": "user",
                        "content": "x = 32, y = x + 1, what is y?",
                    },
                    "33",
                ),
                ({"role": "user", "content": "Based on previous x and y, x + y = ?"}, "65"),
            ],
            [
                ({"role": "user", "content": "the price of an apple is 5 yuan, how much is 10 apples?"}, "50"),
                ({"role": "user", "content": "how much is 5 apples?"}, "25"),
            ],
        ]
        client = AsyncClient(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        cur_uuid = uuid.uuid4().int
        prompt = random.choice(all_prompts)
        for p, answer in prompt:
            output = (
                await client.chat_interactive_v1(
                    prompt=p["content"],
                    request_output_len=4096,
                    stop=["<|im_end|>"],
                    top_k=1,
                    session_id=cur_uuid,
                    stream=False,
                    interactive_mode=True,
                )
            )["text"]
        self.assertTrue(
            answer.lower() in output.lower(),
            msg=f"Interactive Test failed: Prompt: {p['content']}, Expected '{answer}', got '{output}'",
        )
        await client.end_session(cur_uuid)
        await client.close()

    async def try_forward_complete(self, port=8000):
        prompts = [
            ({"role": "user", "content": "compute 5*30+5"}, "155"),
            ({"role": "user", "content": "compute 5*7=?"}, "35"),
            ({"role": "user", "content": "what is the capital of France?"}, "Paris"),
            ({"role": "user", "content": "what is the capital of Japan?"}, "Tokyo"),
        ]
        client = AsyncClient(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        prompt, answer = random.choice(prompts)
        response = await client.chat.completions.create(
            model="ALLOYLM",
            messages=[prompt],
            max_completion_tokens=4096,
            stop=["<|im_end|>"],
            top_logprobs=1,
            extra_body={"top_k": 1},
        )
        output = response.choices[0].message.content
        self.assertTrue(
            answer.lower() in output.lower(),
            msg=f"Complete Test failed: Prompt: {prompt['content']}, Expected '{answer}', got '{output}'",
        )
        await client.close()

    async def try_generate(self, tokenizer, port=8000):
        prompts = [
            ({"role": "user", "content": "compute 5*30+5"}, "155"),
            ({"role": "user", "content": "compute 5*7=?"}, "35"),
            ({"role": "user", "content": "what is the capital of France?"}, "Paris"),
            ({"role": "user", "content": "what is the capital of Japan?"}, "Tokyo"),
        ]
        client = AsyncClient(api_key="EMPTY", base_url=f"http://localhost:{port}")
        prompt, answer = random.choice(prompts)
        input_ids = tokenizer.apply_chat_template(
            conversation=[prompt],
            add_generation_prompt=True,
            return_dict=False,
        )
        output = (await client.generate(input_ids=input_ids, top_k=1, max_tokens=4096))["text"]
        self.assertTrue(
            answer.lower() in output.lower(),
            msg=f"Generate Test failed: Prompt: {prompt['content']}, Expected '{answer}', got '{output}'",
        )
        await client.close()

    async def test_system(self):
        async with LaunchTestServer() as server:
            bs = 32
            futures = []
            for i in range(bs):
                futures.append(asyncio.create_task(self.try_forward_interactive()))
                futures.append(asyncio.create_task(self.try_forward_complete()))
                futures.append(asyncio.create_task(self.try_generate(server.tokenizer)))
            await asyncio.gather(*futures)

    @unittest.skipUnless(
        os.environ.get("RUN_30B_TESTS", "0") == "1",
        "MOE tests are disabled unless RUN_30B_TESTS=1 is set in the environment",
    )
    async def test_system_moe(self):
        async with LaunchTestServer(model_path="Qwen/Qwen3-30B-A3B") as server:
            bs = 32
            futures = []
            for i in range(bs):
                futures.append(asyncio.create_task(self.try_forward_interactive()))
                futures.append(asyncio.create_task(self.try_forward_complete()))
                futures.append(asyncio.create_task(self.try_generate(server.tokenizer)))
            await asyncio.gather(*futures)

    async def test_system_chunk_prefill(self):
        async with LaunchTestServer(max_prefill_length=32) as server:
            bs = 32
            futures = []
            for i in range(bs):
                futures.append(asyncio.create_task(self.try_forward_interactive()))
                futures.append(asyncio.create_task(self.try_forward_complete()))
                futures.append(asyncio.create_task(self.try_generate(server.tokenizer)))
            await asyncio.gather(*futures)
