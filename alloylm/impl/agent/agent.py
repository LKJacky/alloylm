import asyncio
import time

from openai import AsyncClient

from alloylm.utils import get_logger

from .env import BaseEnv


class BaseAgent:
    chat_time = 0
    exec_time = 0
    last_log_time = time.time()

    def __init__(
        self,
        client: AsyncClient,
        env: BaseEnv = None,
        max_steps: int = 300,
    ):
        self.client = client
        self.env = env
        self.max_steps = max_steps
        self.used_tokens = []
        self.finish_reason = "stop"

        self.chat_time = 0
        self.exec_time = 0

        self.messages = []

    # basic ability
    async def chat(self, messages):
        t0 = time.time()
        response = await self.client.chat.completions.create(
            messages=messages, tools=self.env.tools() if self.env is not None else None
        )
        self.used_tokens.extend(
            [response.usage.prompt_tokens - sum(self.used_tokens), response.usage.completion_tokens]
        )
        self.finish_reason = response.choices[0].finish_reason
        self.chat_time += time.time() - t0
        return response.choices[0].message

    async def execute(self, call) -> str:
        t0 = asyncio.get_event_loop().time()
        response: str = await self.env.execute_call(call)
        self.exec_time += asyncio.get_event_loop().time() - t0
        return response

    async def solve(self, messages=()):
        self.messages.extend(messages)
        for _ in range(self.max_steps):
            message = await self.chat(self.messages)
            # build message from assistant
            message_data = {"role": message.role, "content": message.content}
            if message.tool_calls:
                message_data["tool_calls"] = [call.model_dump(exclude_none=True) for call in message.tool_calls]
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                message_data["reasoning_content"] = message.reasoning_content
            self.messages.append(message_data)

            if not message.tool_calls:
                break
            else:
                for call in message.tool_calls:
                    response = await self.execute(call)
                    self.messages.append({"role": "tool", "tool_call_id": call.id, "content": response})
        return self.messages

    async def close(self):
        self.__class__.chat_time += self.chat_time
        self.__class__.exec_time += self.exec_time
        if time.time() - self.__class__.last_log_time > 60:
            get_logger().info(
                f"Agent time ratio: chat {self.__class__.chat_time:.2f}s / exec {self.__class__.exec_time:.2f}s, {self.__class__.chat_time / (self.__class__.exec_time + 1e-6):.2f}x"
            )
            self.__class__.last_log_time = time.time()
            self.__class__.chat_time = 0
            self.__class__.exec_time = 0
