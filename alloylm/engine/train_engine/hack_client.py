import asyncio
import base64
import copy
import os
from collections import defaultdict, deque
from contextlib import contextmanager

import jinja2
import ray
import requests
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from transformers import AutoTokenizer

from alloylm.algorithm.rl.utils import get_logger, report_error_once
from alloylm.server.client import HighConcurrentClient as BaseAsyncClient
from alloylm.server.client import (
    HighConcurrentClientInteractive as BaseAsyncClientInteractive,
)

logger = get_logger()


def log_once(message: str):
    if not hasattr(log_once, "logged_messages"):
        log_once.logged_messages = set()
    if message not in log_once.logged_messages:
        logger.info(message)
        log_once.logged_messages.add(message)


# /generate input
class GenerateReqInput(BaseModel):
    session_id: int | None = -1
    prompt: str | None = None
    input_ids: list[int] | None = None
    return_logprob: bool | None = None
    max_tokens: int = 128
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    stream: bool | None = False
    temperature: float = 1.0
    repetition_penalty: float | None = 1.0
    ignore_eos: bool | None = False
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    include_stop_str_in_output: bool | None = False
    return_routed_experts: bool | None = False

    max_entropy: float | None = 100.0


class OutputChecker:
    def __init__(self):
        self.url = "http://" + os.environ.get("COMPASS_VERIFIER_V2_HOST", "127.0.0.1:6000") + "/v1"
        self.model_name = None

    async def check(self, question, output: str):
        if self.model_name is None:
            self.model_name = requests.get(f"{self.url}/models").json()["data"][0]["id"]
        if len(output) < 256:
            return "appropriate"
        output = output.strip()[-1000:]

        client = BaseAsyncClient(base_url=self.url)
        try:
            prompt = """
You are a helpful assistant that checks whether the model output is appropriate given the question.
Question: {question}
A Part of Model Output: {output}

Please answer with following choices:

A: the output is gibberish, such as random characters or repeated phrases.
B: otherwise

Only output A, B, nothing else.
    """.strip()
            prompt = prompt.format(question=question, output=output)
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1,
                temperature=0.01,
                model="deepseek-chat",
            )

            finish_reason = {"A": "gibberish", "B": "appropriate"}
            answer = response.choices[0].message.content.strip().upper()
            return finish_reason.get(answer, "unknown")
        finally:
            await client.close()


class CollectTokenClient:
    collected_tokens = defaultdict(deque)
    step_index = 0

    @classmethod
    def simple_template(cls, messages):
        prompt = ""
        for message in messages:
            prompt += f"{message['role']}: {message['content']}\n"
        return prompt

    @classmethod
    def record_ids(cls, messages, input_ids, labels, log_probs, entropy, routed_experts=None):
        key = cls.simple_template(messages)
        cls.collected_tokens[key].append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "log_probs": log_probs,
                "entropy": entropy,
                "routed_experts": routed_experts,
                "step": cls.step_index,
            }
        )

    @classmethod
    def retrieve_collected_tokens(cls, messages):
        key = cls.simple_template(messages)
        if key in cls.collected_tokens and len(cls.collected_tokens[key]) > 0:
            content = cls.collected_tokens[key].popleft()
            if len(cls.collected_tokens[key]) == 0:
                cls.collected_tokens.pop(key)
            return content
        else:
            raise ValueError("No collected tokens for the given messages")

    @classmethod
    def step(cls):
        cls.step_index += 1
        for key in list(cls.collected_tokens.keys()):
            for _ in range(len(cls.collected_tokens[key])):
                item = cls.collected_tokens[key].popleft()
                if item["step"] + 10 >= cls.step_index:
                    cls.collected_tokens[key].append(item)
                elif item["routed_experts"] is not None:
                    data = base64.b64decode(item["routed_experts"])
                    routed_experts = ray.cloudpickle.loads(data)
                    ray.internal.free(routed_experts, local_only=False)


class Session:
    def __init__(self, request_kwargs: GenerateReqInput):
        self.text = ""
        self.origin_input_id_length = len(request_kwargs.input_ids)
        self.input_ids = copy.copy(request_kwargs.input_ids)
        self.logprobs = []
        self.entropy = []
        self.request_kwargs = request_kwargs
        self.completion_tokens = request_kwargs.max_tokens

    def append(self, text, input_ids, logprobs, entropy=None):
        self.text += text
        self.input_ids.extend(input_ids)
        self.logprobs.extend(logprobs)
        if entropy is not None:
            self.entropy.extend(entropy)
        self.completion_tokens -= len(input_ids)

    def get_request_kwargs(self):
        kwargs = self.request_kwargs.model_dump()
        kwargs["input_ids"] = self.input_ids
        kwargs["max_tokens"] = self.request_kwargs.max_tokens - (len(self.input_ids) - self.origin_input_id_length)
        return kwargs


class HighConcurrentClient(BaseAsyncClient, CollectTokenClient):
    # for generation api
    tokenizer: AutoTokenizer = None
    chat_template: jinja2.Template = None

    # for monitoring
    server_activate_event: asyncio.Event = None
    running_count: int = 0
    generated_tokens = 0
    prefill_overhead = 0

    return_routed_experts: bool = False

    def __init__(self, *, api_key=None, base_url=None, concurrency=4096, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, concurrency=concurrency, **kwargs)
        assert self.tokenizer is not None, "Tokenizer must be provided before initialization"

    @contextmanager
    def posting(self):
        self.__class__.running_count += 1
        try:
            yield
        finally:
            self.__class__.running_count -= 1

    async def generate_post(self, generate_kwargs: GenerateReqInput, timeout):
        with self.posting():
            async with (await self.get_session()).post(
                self._base_url.replace("/v1", "") + "/generate",
                json=generate_kwargs.model_dump(),
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Request failed with status {response.status}, {await response.text()}")
                data = await response.json()
                if data["meta_info"]["output_token_logprobs"] is None:
                    data["meta_info"]["output_token_logprobs"] = []
                return data

    async def _generate_once(self, generate_kwargs: GenerateReqInput, timeout):
        try:
            session = Session(request_kwargs=generate_kwargs)
            for i in range(100000):
                await self.server_activate_event.wait()
                if i != 0:
                    self.__class__.prefill_overhead += len(session.input_ids)

                generate_kwargs.input_ids = session.input_ids
                generate_kwargs.max_tokens = session.completion_tokens
                data = await self.generate_post(generate_kwargs, timeout)

                if data is None:
                    continue
                self.__class__.generated_tokens += len(data["output_ids"])

                session.append(
                    text=data["text"],
                    input_ids=data["output_ids"],
                    logprobs=data["meta_info"]["output_token_logprobs"],
                    entropy=data["meta_info"]["entropy"],
                )

                if data["meta_info"]["finish_reason"]["type"] == "abort":
                    continue
                else:
                    return {
                        "text": session.text,
                        "output_ids": session.input_ids[session.origin_input_id_length :],
                        "meta_info": {
                            "prompt_tokens": session.origin_input_id_length,
                            "completion_tokens": len(session.input_ids) - session.origin_input_id_length,
                            "finish_reason": data["meta_info"]["finish_reason"],
                            "output_token_logprobs": session.logprobs if generate_kwargs.return_logprob else None,
                            "routed_experts": data["meta_info"].get("routed_experts", None),
                            "entropy": session.entropy if generate_kwargs.return_logprob else None,
                        },
                    }
        except Exception as e:
            report_error_once(f"Exception in _generate_once: {e}")
            raise e

    async def _generate(self, messages, **kwargs):
        timeout = kwargs.pop("timeout", int(os.environ.get("ALLOYLM_TIMEOUT", 3600)))
        chat_template = kwargs.pop("chat_template", None) or self.chat_template

        input_str = chat_template.render(messages=messages, add_generation_prompt=True)
        input_ids = await asyncio.get_running_loop().run_in_executor(None, self.tokenizer.encode, input_str)
        extra_body = kwargs.pop("extra_body", {})
        return_logprob = kwargs.pop("logprobs", True)
        generate_kwargs = GenerateReqInput(
            # prompt=input_str,
            input_ids=input_ids,
            return_logprob=return_logprob,
            stop=kwargs.pop("stop", None),
            # sample args
            temperature=kwargs.pop("temperature", 1.0),
            top_p=kwargs.pop("top_p", 1.0),
            top_k=extra_body.pop("top_k", 0),
            max_tokens=kwargs.pop("max_tokens", kwargs.pop("max_completion_tokens", 256)),
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            include_stop_str_in_output=return_logprob,
            return_routed_experts=self.__class__.return_routed_experts,
            **extra_body,
        )
        data = await self._generate_once(generate_kwargs, timeout)
        # WARNING: for lmdeploy and qwen
        if return_logprob and data["text"].endswith("<|im_end|>"):
            data["text"] = data["text"][: -len("<|im_end|>")]

        model_name = kwargs.pop("model", None)
        if generate_kwargs.return_logprob:
            logprobs = [0] * len(input_ids) + [x[0] for x in data["meta_info"]["output_token_logprobs"]]
            entropy = [0] * len(input_ids) + data["meta_info"]["entropy"]
        else:
            logprobs = None
            entropy = None
        output = ChatCompletion(
            id="0",
            object="chat.completion",
            created=0,
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "content": data["text"],
                        "role": "assistant",
                        "input_ids": input_ids,
                        "gen_tokens": data["output_ids"],
                        "logprobs": logprobs,
                        "entropy": entropy,
                        "routed_experts": data["meta_info"].get("routed_experts", None),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": data["meta_info"]["prompt_tokens"],
                "completion_tokens": data["meta_info"]["completion_tokens"],
                "total_tokens": data["meta_info"]["prompt_tokens"] + data["meta_info"]["completion_tokens"],
            },
        )
        output.choices[0].finish_reason = data["meta_info"]["finish_reason"]["type"]

        if len(kwargs) != 0:
            print(f"Unused kwargs in _generate: {kwargs}")

        return output

    async def _create_completion(self, **kwargs):
        output = await self._generate(**kwargs)
        for choice in output.choices:
            if hasattr(choice.message, "gen_tokens"):
                choice.message.output_ids = choice.message.gen_tokens
        if kwargs.get("extra_body", {}).get("return_token_ids", True):  # default to True
            messages = kwargs["messages"] + [{"role": "assistant", "content": output.choices[0].message.content}]
            self.__class__.record_ids(
                messages,
                input_ids=output.choices[0].message.input_ids + output.choices[0].message.output_ids,
                labels=[-100] * len(output.choices[0].message.input_ids) + output.choices[0].message.output_ids,
                log_probs=output.choices[0].message.logprobs,
                entropy=output.choices[0].message.entropy,
                routed_experts=output.choices[0].message.routed_experts,
            )

        return output


class HighConcurrentClientInteractive(BaseAsyncClientInteractive, CollectTokenClient):
    # for generation api
    tokenizer: AutoTokenizer = None
    chat_template: jinja2.Template = None

    # for monitoring
    server_activate_event: asyncio.Event = None
    running_count: int = 0
    generated_tokens = 0
    prefill_overhead = 0

    def __init__(self, *, api_key=None, base_url=None, concurrency=1024 * 16, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, concurrency=concurrency, **kwargs)
        self.text = ""
        self.tokens = []
        self.labels = []
        self.log_probs = []

    async def generate(
        self,
        *args,
        **kwargs,
    ):
        await self.__class__.server_activate_event.wait()
        self.__class__.running_count += 1
        try:
            res = await super().generate(*args, **kwargs)
            self.__class__.generated_tokens += len(res["output_ids"])
            return res
        finally:
            self.__class__.running_count -= 1

    async def _create_completion(self, **kwargs):
        input_messages = kwargs.pop("messages")
        chat_template = kwargs.pop("chat_template", None) or self.chat_template
        text = chat_template.render(messages=input_messages, add_generation_prompt=True)
        assert text.startswith(self.text), f"New text must start with the old text. old: {self.text}, new: {text}"
        new_text = text[len(self.text) :]
        new_tokens = await asyncio.get_running_loop().run_in_executor(None, self.tokenizer.encode, new_text)

        self.text += new_text
        self.tokens.extend(new_tokens)
        self.labels.extend([-100] * len(new_tokens))
        self.log_probs.extend([0] * len(new_tokens))

        return_logprobs = kwargs.pop("logprobs", True)
        args = GenerateReqInput(
            session_id=self.session_id,
            # prompt=input_str,
            input_ids=new_tokens,
            return_logprob=return_logprobs,
            stop=kwargs.pop("stop", None),
            # sample args
            temperature=kwargs.pop("temperature", 1.0),
            top_p=kwargs.pop("top_p", 1.0),
            top_k=kwargs.pop("extra_body", {}).get("top_k", 0),
            max_tokens=kwargs.pop("max_tokens", kwargs.pop("max_completion_tokens", 256)),
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            include_stop_str_in_output=return_logprobs,
        )
        data = await self.generate(**args.model_dump())

        self.text += data["text"]
        self.tokens.extend(data["output_ids"])
        self.labels.extend(data["output_ids"])
        if args.return_logprob:
            self.log_probs.extend([x[0] for x in data["meta_info"]["output_token_logprobs"]])
        self.messages = input_messages + [{"role": "assistant", "content": data["text"]}]
        return ChatCompletion(
            id="0",
            object="chat.completion",
            created=0,
            model=kwargs.pop("model", None),
            choices=[
                {
                    "index": 0,
                    "message": {
                        "content": data["text"],
                        "role": "assistant",
                    },
                    "finish_reason": data["meta_info"]["finish_reason"]["type"],
                }
            ],
            usage={
                "prompt_tokens": len(new_tokens),
                "completion_tokens": len(data["output_ids"]),
                "total_tokens": len(new_tokens) + len(data["output_ids"]),
            },
        )

    async def close(self):
        self.__class__.record_ids(self.messages, input_ids=self.tokens, labels=self.labels, log_probs=self.log_probs)
        return await super().close()
