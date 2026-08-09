import asyncio
import copy
import os
import traceback
import uuid
from typing import Any, Optional

import aiohttp
import httpx
from openai import NOT_GIVEN, AsyncClient, NotGiven
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field


class SharedSession:
    _session = None
    _num_using = 0
    _session_loop = None

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()

        need_new_session = (
            cls._session is None
            or cls._session.closed
            or cls._session_loop is None
            or cls._session_loop is not loop
            or cls._session_loop.is_closed()
        )

        if need_new_session:
            # Close old session if possible
            if cls._session is not None and not cls._session.closed:
                try:
                    await cls._session.close()
                except RuntimeError:
                    # Old loop may already be closed; ignore during recovery
                    pass

            connector = aiohttp.TCPConnector(limit=0)
            cls._session = aiohttp.ClientSession(connector=connector)
            cls._session_loop = loop

        cls._num_using += 1
        return cls._session

    @classmethod
    async def release(cls):
        cls._num_using -= 1
        if cls._num_using == 0 and cls._session is not None:
            await cls._session.close()
            cls._session = None


async def cancel_and_wait(future: asyncio.Future, raise_cancelled=False):
    future.cancel()
    try:
        await future
    except asyncio.CancelledError as e:
        if raise_cancelled:
            raise e
    except Exception as e:
        traceback.print_exc()
        raise e


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


class HighConcurrentClient(AsyncClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        concurrency=1024 * 16,
        **kwargs,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create_completion
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.session = None

    async def get_session(self):
        if self.session is None:
            self.session = await SharedSession.get_session()
        return self.session

    async def _create_completion(self, **kwargs):
        try:
            if "extra_body" in kwargs:
                kwargs.update(kwargs.pop("extra_body"))

            for key in copy.copy(list(kwargs.keys())):
                if kwargs[key] == NOT_GIVEN:
                    kwargs.pop(key)

            if "timeout" not in kwargs or kwargs["timeout"] is NotGiven:
                kwargs["timeout"] = int(os.environ.get("ALLOYLM_TIMEOUT", 3600))
            timeout = kwargs["timeout"]

            async with (await self.get_session()).post(
                self._base_url + "/chat/completions", json=kwargs, timeout=timeout
            ) as response:
                assert response.status == 200, f"Request failed with status {response.status}, {await response.text()}"
                data = await response.json()
                finish_reason = data["choices"][0].get("finish_reason", "stop")
                data["choices"][0]["finish_reason"] = "stop"
                output = ChatCompletion(**data)
                output.choices[0].finish_reason = finish_reason
                return output
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            traceback.print_exc()
            raise e

    async def chat_interactive_v1(
        self,
        prompt: str | list[dict[str, Any]],
        image_url: str | list[str] | None = None,
        session_id: int = -1,
        interactive_mode: bool = False,
        stream: bool = False,
        stop: str | list[str] | None = None,
        request_output_len: Optional[int] = None,  # noqa
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        skip_special_tokens: bool | None = True,
        spaces_between_special_tokens: bool | None = True,
        cancel: bool | None = False,  # cancel a responding request
        adapter_name: str | None = None,
        seed: int | None = None,
        min_new_tokens: int | None = None,
        min_p: float = 0.0,
    ):
        class GenerateRequest(BaseModel):
            """Generate request."""

            prompt: str | list[dict[str, Any]]
            image_url: str | list[str] | None = Field(default=None, examples=[None])
            session_id: int = -1
            interactive_mode: bool = False
            stream: bool = False
            stop: str | list[str] | None = Field(default=None, examples=[None])
            request_output_len: Optional[int] = Field(default=None, examples=[None])  # noqa
            top_p: float = 0.8
            top_k: int = 40
            temperature: float = 0.8
            repetition_penalty: float = 1.0
            ignore_eos: bool = False
            skip_special_tokens: bool | None = True
            spaces_between_special_tokens: bool | None = True
            cancel: bool | None = False  # cancel a responding request
            adapter_name: str | None = Field(default=None, examples=[None])
            seed: int | None = None
            min_new_tokens: int | None = Field(default=None, examples=[None])
            min_p: float = 0.0

        request = GenerateRequest(
            prompt=prompt,
            image_url=image_url,
            session_id=session_id,
            interactive_mode=interactive_mode,
            stream=stream,
            stop=stop,
            request_output_len=request_output_len,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            ignore_eos=ignore_eos,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            cancel=cancel,
            adapter_name=adapter_name,
            seed=seed,
            min_new_tokens=min_new_tokens,
            min_p=min_p,
        )
        async with (await self.get_session()).post(
            self._base_url + "/chat/interactive", json=request.model_dump(), timeout=7200
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            result = await response.json()
        return result

    async def generate(
        self,
        session_id: int | None = -1,
        prompt: str | None = None,
        input_ids: list[int] | None = None,
        return_logprob: bool | None = None,
        max_tokens: int = 128,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        stream: bool | None = False,
        temperature: float = 1.0,
        repetition_penalty: float | None = 1.0,
        ignore_eos: bool | None = False,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        skip_special_tokens: bool | None = True,
        spaces_between_special_tokens: bool | None = True,
        include_stop_str_in_output: bool | None = False,
        return_routed_experts=False,
        **kwargs,
    ):
        args = GenerateReqInput(
            session_id=session_id,
            prompt=prompt,
            input_ids=input_ids,
            return_logprob=return_logprob,
            max_tokens=max_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            stream=stream,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            ignore_eos=ignore_eos,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            return_routed_experts=return_routed_experts,
        )
        json_body = args.model_dump()
        json_body.update(kwargs)
        async with (await self.get_session()).post(
            self._base_url.replace("/v1", "") + "/generate",
            json=json_body,
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            result = await response.json()
        return result

    async def abort_request(self):
        async with (await self.get_session()).post(
            self._base_url + "/abort_request",
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            await response.text()

    async def end_session(self, session_id):
        async with (await self.get_session()).post(
            self._base_url + "/chat/interactive",
            json=dict(prompt="", session_id=session_id, request_output_len=0, interactive_mode=False),
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            await response.text()

    async def close(self):
        await SharedSession.release()
        self.session = None


class HighConcurrentClientInteractive(HighConcurrentClient):
    def __init__(self, *, api_key=None, base_url=None, concurrency=1024 * 16, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, concurrency=concurrency, **kwargs)
        self.session_id = uuid.uuid4().int
        self.messages = []
        self.interactive_supported = None

    async def check_server_support(self) -> bool:
        """Return True if the server exposes the /chat/interactive endpoint."""
        try:
            async with (await self.get_session()).get(self._base_url + "/chat/interactive", timeout=10) as response:
                return response.status != 404
        except Exception:
            return False

    async def _create_completion(self, **kwargs):
        if self.interactive_supported is None:
            self.interactive_supported = await self.check_server_support()
        if self.interactive_supported:
            return await self._create_completion_interactive(**kwargs)
        else:
            return await super()._create_completion(**kwargs)

    async def _create_completion_interactive(self, **kwargs):
        messages = kwargs.pop("messages")
        messages = messages[len(self.messages) :]
        response = await self.chat_interactive_v1(
            prompt=messages if len(messages) > 1 else messages[0]["content"],
            session_id=self.session_id,
            interactive_mode=True,
            stream=False,
            stop=kwargs.get("stop", None),
            top_p=kwargs.get("top_p", 0.8),
            top_k=kwargs.get("extra_body", {}).get("top_k", 40),
            temperature=kwargs.get("temperature", 0.8),
            request_output_len=kwargs.get("max_tokens", 128),
        )
        self.messages.extend(messages)
        self.messages.append({"role": "assistant", "content": response["text"]})
        return ChatCompletion(
            id="0",
            object="chat.completion",
            created=0,
            model="",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "content": response["text"],
                        "role": "assistant",
                    },
                    "finish_reason": response["finish_reason"],
                }
            ],
            usage={
                "prompt_tokens": response["input_tokens"],
                "completion_tokens": response["tokens"],
                "total_tokens": response["history_tokens"] + response["tokens"] + response["input_tokens"],
            },
        )

    async def close(self):
        if self.interactive_supported:
            await self.chat_interactive_v1(
                prompt="",
                session_id=self.session_id,
                interactive_mode=False,
                request_output_len=0,
            )
        await super().close()


class HighConcurrentClient(AsyncClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        concurrency=1024 * 16,
        **kwargs,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create_completion
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.session = None

    async def _create_completion(self, **kwargs):
        try:
            if "extra_body" in kwargs:
                kwargs.update(kwargs.pop("extra_body"))

            for key in copy.copy(list(kwargs.keys())):
                if kwargs[key] == NOT_GIVEN:
                    kwargs.pop(key)

            if "timeout" not in kwargs or kwargs["timeout"] is NotGiven:
                kwargs["timeout"] = int(os.environ.get("ALLOYLM_TIMEOUT", 3600))
            timeout = kwargs["timeout"]
            async with (await self.get_session()).post(
                self._base_url + "/chat/completions", json=kwargs, timeout=timeout
            ) as response:
                assert response.status == 200, f"Request failed with status {response.status}, {await response.text()}"
                data = await response.json()
                finish_reason = data["choices"][0].get("finish_reason", "stop")
                data["choices"][0]["finish_reason"] = "stop"
                output = ChatCompletion(**data)
                output.choices[0].finish_reason = finish_reason
                return output
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            traceback.print_exc()
            raise e

    async def chat_interactive_v1(
        self,
        prompt: str | list[dict[str, Any]],
        image_url: str | list[str] | None = None,
        session_id: int = -1,
        interactive_mode: bool = False,
        stream: bool = False,
        stop: str | list[str] | None = None,
        request_output_len: Optional[int] = None,  # noqa
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
        ignore_eos: bool = False,
        skip_special_tokens: bool | None = True,
        spaces_between_special_tokens: bool | None = True,
        cancel: bool | None = False,  # cancel a responding request
        adapter_name: str | None = None,
        seed: int | None = None,
        min_new_tokens: int | None = None,
        min_p: float = 0.0,
    ):
        class GenerateRequest(BaseModel):
            """Generate request."""

            prompt: str | list[dict[str, Any]]
            image_url: str | list[str] | None = Field(default=None, examples=[None])
            session_id: int = -1
            interactive_mode: bool = False
            stream: bool = False
            stop: str | list[str] | None = Field(default=None, examples=[None])
            request_output_len: Optional[int] = Field(default=None, examples=[None])  # noqa
            top_p: float = 0.8
            top_k: int = 40
            temperature: float = 0.8
            repetition_penalty: float = 1.0
            ignore_eos: bool = False
            skip_special_tokens: bool | None = True
            spaces_between_special_tokens: bool | None = True
            cancel: bool | None = False  # cancel a responding request
            adapter_name: str | None = Field(default=None, examples=[None])
            seed: int | None = None
            min_new_tokens: int | None = Field(default=None, examples=[None])
            min_p: float = 0.0

        request = GenerateRequest(
            prompt=prompt,
            image_url=image_url,
            session_id=session_id,
            interactive_mode=interactive_mode,
            stream=stream,
            stop=stop,
            request_output_len=request_output_len,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            ignore_eos=ignore_eos,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            cancel=cancel,
            adapter_name=adapter_name,
            seed=seed,
            min_new_tokens=min_new_tokens,
            min_p=min_p,
        )
        async with (await self.get_session()).post(
            self._base_url + "/chat/interactive", json=request.model_dump(), timeout=7200
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            result = await response.json()
        return result

    async def generate(
        self,
        session_id: int | None = -1,
        prompt: str | None = None,
        input_ids: list[int] | None = None,
        return_logprob: bool | None = None,
        max_tokens: int = 128,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        stream: bool | None = False,
        temperature: float = 1.0,
        repetition_penalty: float | None = 1.0,
        ignore_eos: bool | None = False,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        skip_special_tokens: bool | None = True,
        spaces_between_special_tokens: bool | None = True,
        include_stop_str_in_output: bool | None = False,
        return_routed_experts=False,
        **kwargs,
    ):
        args = GenerateReqInput(
            session_id=session_id,
            prompt=prompt,
            input_ids=input_ids,
            return_logprob=return_logprob,
            max_tokens=max_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            stream=stream,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            ignore_eos=ignore_eos,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            return_routed_experts=return_routed_experts,
        )
        json_body = args.model_dump()
        json_body.update(kwargs)
        async with (await self.get_session()).post(
            self._base_url.replace("/v1", "") + "/generate",
            json=json_body,
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            result = await response.json()
        return result

    async def abort_request(self):
        async with (await self.get_session()).post(
            self._base_url + "/abort_request",
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            await response.text()

    async def end_session(self, session_id):
        async with (await self.get_session()).post(
            self._base_url + "/chat/interactive",
            json=dict(prompt="", session_id=session_id, request_output_len=0, interactive_mode=False),
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            await response.text()

    async def get_session(self):
        if self.session is None:
            self.session = await SharedSession.get_session()
        return self.session

    async def close(self):
        await SharedSession.release()
        self.session = None
