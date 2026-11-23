import asyncio
import json
import os
import random
import unittest
import uuid

from openai import AsyncClient

from alloylm.impl.engines.qwen.chat_template import (
    QWEN_THINKING_PATTERN,
    QWEN_TOOL_PATTERN,
    Qwen3ChatTemplate,
)
from alloylm.test_utils import CudaAsyncTestCase, LaunchTestServer
from alloylm.utils import get_logger

logger = get_logger()


class TestLaunchSystem(CudaAsyncTestCase):
    async def try_forward_interactive(self, port=8000):
        from openai import AsyncClient

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

        prompt = random.choice(all_prompts)
        messages = []
        uuid_val = uuid.uuid4().int
        for p, answer in prompt:
            messages.append(p)
            output = await client.chat.completions.create(
                model="ALLOYLM",
                messages=messages,
                max_completion_tokens=4096,
                extra_body={"top_k": 1, "session_id": uuid_val},
            )
            output_str = output.choices[0].message.content
            messages.append({"role": "assistant", "content": output_str})

            self.assertTrue(
                answer.lower() in output_str.lower(),
                msg=f"Interactive Test failed: Prompt: {p['content']}, Expected '{answer}', got '{output_str}'",
            )

        await client.chat.completions.create(
            model="ALLOYLM", messages=[], extra_body={"session_id": uuid_val}
        )  # release session
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

    @staticmethod
    def delta_reasoning(delta) -> str:
        value = getattr(delta, "reasoning_content", None)
        if value:
            return value
        return (getattr(delta, "model_extra", None) or {}).get("reasoning_content") or ""

    @staticmethod
    def merge_tool_calls(state: dict, delta_tool_calls) -> None:
        for call in delta_tool_calls or []:
            entry = state.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function is not None:
                entry["name"] += call.function.name or ""
                entry["arguments"] += call.function.arguments or ""

    async def try_forward_stream(self, port=8000, with_tools=False, thinking=False):
        """Stream a completion and check the SSE deltas reassemble into the
        answer."""
        prompts = [
            ({"role": "user", "content": "compute 5*30+5"}, "155"),
            ({"role": "user", "content": "compute 5*7=?"}, "35"),
            ({"role": "user", "content": "what is the capital of France?"}, "Paris"),
        ]
        tools = None
        if with_tools:
            prompt = {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}
            answer = "get_weather"
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        else:
            prompt, answer = random.choice(prompts)

        client = AsyncClient(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        session_id = uuid.uuid4().int

        stream = await client.chat.completions.create(
            model="ALLOYLM",
            messages=[prompt],
            max_completion_tokens=4096,
            stream=True,
            tools=tools,
            extra_body={"top_k": 1, "session_id": session_id, "chat_template_kwargs": {"thinking": thinking}},
        )

        content = ""
        reasoning = ""
        tool_calls: dict[int, dict] = {}
        finish_reasons = []
        chunks = 0
        async for event in stream:
            chunks += 1
            self.assertTrue(event.choices, msg="Streaming Test failed: chunk without choices")
            choice = event.choices[0]
            if choice.delta is not None:
                content += choice.delta.content or ""
                reasoning += self.delta_reasoning(choice.delta)
                self.merge_tool_calls(tool_calls, choice.delta.tool_calls)
            if choice.finish_reason is not None:
                finish_reasons.append(choice.finish_reason)

        self.assertGreater(chunks, 0, msg="Streaming Test failed: no chunks received")
        expected = "tool_calls" if tool_calls else "stop"
        self.assertTrue(
            finish_reasons and finish_reasons[-1] in (expected, "length"),
            msg=f"Streaming Test failed: finish_reason {finish_reasons}, expected {expected!r} or 'length'",
        )
        # the stop marker must never leak into the streamed text
        self.assertNotIn("<|im_end|>", content, msg="Streaming Test failed: stop marker leaked into content")
        self.assertNotIn("<|im_end|>", reasoning, msg="Streaming Test failed: stop marker leaked into reasoning")
        # the thinking block is split out, its markers never reach the client
        self.assertNotIn("</think>", content, msg="Streaming Test failed: thinking marker leaked into content")
        self.assertNotIn("<think>", content, msg="Streaming Test failed: thinking marker leaked into content")
        self.assertNotIn("<think>", reasoning, msg="Streaming Test failed: thinking marker leaked into reasoning")
        self.assertNotIn("</think>", reasoning, msg="Streaming Test failed: thinking marker leaked into reasoning")
        # a complete tool call is streamed as tool_calls, never as raw text
        self.assertNotIn("<tool_call>", content, msg="Streaming Test failed: raw tool call leaked into content")
        self.assertNotIn("</tool_call>", content, msg="Streaming Test failed: raw tool call leaked into content")

        if with_tools:
            self.assertTrue(content or tool_calls, msg="Streaming Test failed: neither content nor tool calls")
            for index, call in tool_calls.items():
                self.assertTrue(call["name"], msg=f"Streaming Test failed: tool call {index} has no name")
                if call["arguments"]:
                    json.loads(call["arguments"])  # must be valid JSON once reassembled
                self.assertEqual(
                    call["name"], "get_weather", msg=f"Streaming Test failed: unexpected tool {call['name']}"
                )
        else:
            self.assertIn(
                answer.lower(),
                content.lower(),
                msg=f"Streaming Test failed: Prompt: {prompt['content']}, Expected '{answer}', got '{content}'",
            )

        await client.chat.completions.create(
            model="ALLOYLM", messages=[], extra_body={"session_id": session_id}
        )  # release session
        await client.close()

    async def test_system(self):
        async with LaunchTestServer():
            bs = 32
            futures = []
            for i in range(bs):
                futures.append(asyncio.create_task(self.try_forward_interactive()))
                futures.append(asyncio.create_task(self.try_forward_complete()))
            await asyncio.gather(*futures)

    async def test_stream(self):
        async with LaunchTestServer(
            model_path="Qwen/Qwen3-0.6B",
            tool_pattern=QWEN_TOOL_PATTERN,
            thinking_pattern=QWEN_THINKING_PATTERN,
            chat_template=Qwen3ChatTemplate,
        ):
            futures = [
                asyncio.create_task(self.try_forward_stream(with_tools=index % 2 == 1, thinking=True))
                for index in range(8)
            ]
            await asyncio.gather(*futures)

    @unittest.skipUnless(
        os.environ.get("RUN_30B_TESTS", "0") == "1",
        "MOE tests are disabled unless RUN_30B_TESTS=1 is set in the environment",
    )
    async def test_system_moe(self):
        async with LaunchTestServer(model_path="Qwen/Qwen3-30B-A3B"):
            bs = 32
            futures = []
            for i in range(bs):
                futures.append(asyncio.create_task(self.try_forward_interactive()))
                futures.append(asyncio.create_task(self.try_forward_complete()))
            await asyncio.gather(*futures)

    async def test_system_chunk_prefill(self):
        async with LaunchTestServer(max_prefill_length=32):
            bs = 32
            futures = []
            for i in range(bs):
                futures.append(asyncio.create_task(self.try_forward_interactive()))
                futures.append(asyncio.create_task(self.try_forward_complete()))
            await asyncio.gather(*futures)
