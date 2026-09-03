import asyncio
import json
import re
import traceback
import uuid
from queue import Queue
from typing import Any, Dict, List, Optional, Union  # noqa: UP035

import httpx
import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel, Field
from pydantic import BaseModel as _BaseModel
from transformers import AutoTokenizer

from .scheduler import InferItem, ReleaseItem, ResetItem, TaskItem
from .utils import GeneConfig, get_current_ip, get_logger


class ModelCard(_BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "owner"
    root: str | None = None


class ModelList(_BaseModel):
    object: str = "list"
    data: list[ModelCard] = []


# request


class InteractiveRequest(BaseModel):  # copied from lmdeploy for interactive chat
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

    max_entropy: float | None = 100.0


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


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str

    messages: Union[str, List[Dict[str, Any]]] = Field(examples=[[{"role": "user", "content": "hi"}]])  # noqa
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    tools: list[ChatCompletionToolUnionParam] | None = Field(default=None, examples=[None])
    # tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
    #     default="auto", examples=["none"]
    # )
    logprobs: bool | None = False
    top_logprobs: int | None = None
    n: int | None = 1
    logit_bias: Optional[Dict[str, float]] = Field(default=None, examples=[None])  # noqa
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])  # noqa

    max_completion_tokens: int | None = None
    max_tokens: int | None = None  # duplicate of max_completion_tokens

    # extra
    top_k: int | None = 40
    extra_body: dict[str, Any] | None = None

    def clean(self):
        if self.max_completion_tokens is not None and self.max_tokens is not None:
            get_logger().warn("Both max_completion_tokens and max_tokens are provided, using max_completion_tokens")
        self.max_completion_tokens = self.max_completion_tokens if self.max_completion_tokens else self.max_tokens


# request session


class Messages:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.messages = []
        self.cached_text = ""

    def render_messages(self, new_messages, for_generate=False, tools=None):
        self.messages.extend(new_messages)
        text = self._get_text(add_generation_prompt=for_generate, tools=tools)
        assert text.startswith(self.cached_text), (
            f"New text should start with cached text, but got:\nCached:\n{self.cached_text}\nNew:\n{text}"
        )
        diff_text = text[len(self.cached_text) :]
        self.cached_text = text
        return diff_text

    def _get_text(self, add_generation_prompt=True, tools=None):
        try:
            return self.chat_template.render(
                messages=self.messages, add_generation_prompt=add_generation_prompt, enable_thinking=False, tools=tools
            )
        except Exception:  # noqa
            return self.chat_template.render(
                messages=self.messages, add_generation_prompt=add_generation_prompt, tools=tools
            )


class SessionItem:
    def __init__(self, session_id, chat_template: Template):
        self.session_id = session_id
        self.forwarded_tokens = 0

        self.messages = Messages(chat_template)

    def forward_tokens(self, tokens):
        self.forwarded_tokens += tokens

    def render_messages(self, new_messages, for_generate=False, tools=None):
        return self.messages.render_messages(new_messages, for_generate=for_generate, tools=tools)


def parse_tool_calls(text: str, tool_pattern: re.Pattern) -> tuple[str, list[dict]]:
    matches = list(tool_pattern.finditer(text))
    if not matches:
        return text, []

    tool_calls = []
    try:
        for match in matches:
            payload = json.loads(match.group(1))
            name = payload["name"]
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, (dict, list)):
                raise ValueError("Invalid tool call payload")  # noqa: TRY004
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )
        content = tool_pattern.sub("", text).strip()
        return content or None, tool_calls
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return text, []


# server


class APIServer:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        queue: Queue[GeneConfig],
        chat_template: Template = None,
        port=8000,
        proxy_url=None,
        model_name="",
        tool_pattern: str | None = None,
    ):
        self.tokenizer = tokenizer
        jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )
        self.chat_template = chat_template if chat_template else jinja_env.from_string(tokenizer.get_chat_template())
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        get_logger().info(f"Using chat template:\n{self.chat_template.render(messages=test_messages)}")
        get_logger().info(
            f"Using chat template:\n{self.chat_template.render(messages=test_messages, add_generation_prompt=True, enable_thinking=False)}"
        )
        get_logger().info(
            f"Using chat template:\n{self.chat_template.render(messages=test_messages + [{'role': 'assistant', 'content': 'Hi!'}])}"
        )

        self.queue = queue

        self.port = port

        self.task = None
        self.server = None
        self.proxy_url = proxy_url
        self.model_name = model_name
        self.session_map: dict[int, SessionItem] = {}

        self.default_stop_tokens = [self.tokenizer.eos_token_id]
        get_logger().info(
            f"Use eos token {self.tokenizer.eos_token}({self.tokenizer.eos_token_id}) as default stop token"
        )
        self.tool_pattern = re.compile(tool_pattern, re.DOTALL) if tool_pattern else None

    # session management

    def get_session(self, session_id) -> SessionItem:
        assert session_id != -1
        if session_id in self.session_map:
            return self.session_map[session_id]
        else:
            self.session_map[session_id] = SessionItem(session_id, self.chat_template)
            return self.session_map[session_id]

    async def release_session(self, session_id):
        session = self.session_map.pop(session_id, None)
        if session is not None:
            await self.run_on_engine(ResetItem(session.session_id))

    # apis

    async def launch(self):
        app = FastAPI(title="OpenAI-compatible API")

        app.post("/v1/chat/completions")(self.chat_completion)
        app.post("/v1/chat/interactive")(self.chat_interactive)
        app.post("/generate")(self.generate)
        app.post("/abort_request")(self.abort_request)
        app.get("/health")(self.health)
        app.get("/v1/models")(self.available_models)

        while True:
            try:
                config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="error", timeout_keep_alive=500)
                server = uvicorn.Server(config)
                loop = asyncio.get_event_loop()
                task = loop.create_task(server.serve())

                # Wait for server to start
                while not server.started:
                    await asyncio.sleep(0.001)
                break
            except Exception as e:  # noqa
                get_logger().error(f"Failed to launch API server on port {self.port}: {e}")
                get_logger().error("Retrying in 1 second...")
                self.port += 1  # Increment port to avoid conflicts
                await asyncio.sleep(1)

        self.task = task
        self.server = server

        # connect proxy
        if self.proxy_url is not None:
            url = f"{self.proxy_url}/nodes/add"
            data = {
                "url": f"http://{get_current_ip()}:{self.port}",
                "status": {"models": [self.model_name], "role": 1},
            }
            headers = {"accept": "application/json", "Content-Type": "application/json"}
            client = httpx.AsyncClient()
            while True:
                try:
                    response = await client.post(url, headers=headers, json=data)
                    if response.status_code != 200:
                        get_logger().error(f"Service registration failed: {response.text}")
                        raise HTTPException(status_code=400, detail="Service registration failed")
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.01)  # wait for proxy launched
            await client.aclose()

        get_logger().info(f"API server launched successfully on port {self.port}")

    async def chat_completion(self, request: ChatCompletionRequest):
        request.clean()

        gene_config = GeneConfig(
            top_p=request.top_p,
            temperature=request.temperature,
            total_max_length=request.max_completion_tokens,
            top_k=request.top_k,
            stop_token=request.stop,
            max_entropy=request.extra_body.get("max_entropy", 100) if request.extra_body else 100,
            release_at_once=True,
        )

        session: SessionItem = self.get_session(uuid.uuid4().int)
        text = session.render_messages(request.messages, for_generate=True, tools=request.tools)
        response, input_ids, result = await self.run_infer(session, gene_config, text=text)
        self.session_map.pop(session.session_id, None)
        if self.tool_pattern:
            content, tool_calls = parse_tool_calls(response, tool_pattern=self.tool_pattern)
        else:
            content, tool_calls = response, []

        message = {
            "role": "assistant",
            "content": content,
            "output_ids": result["tokens"],
            "input_ids": input_ids,
            "logprobs": [0] * len(input_ids) + result["log_prob"],
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": str(session.session_id),
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else result["finish_reason"],
                }
            ],
            "usage": {
                "completion_tokens": result["usage"]["output_tokens"],
                "prompt_tokens": result["usage"]["input_tokens"],
                "total_tokens": result["usage"]["output_tokens"] + result["usage"]["input_tokens"],
            },
        }

    async def chat_interactive(self, request: InteractiveRequest):
        try:
            if request.prompt == "":  # stop session
                # reset batch
                await self.release_session(request.session_id)
                return JSONResponse(
                    {
                        "text": "",
                        "tokens": 0,
                        "input_tokens": 0,
                        "history_tokens": 0,
                        "finish_reason": "stop",
                    }
                )
            else:
                if isinstance(request.prompt, str):
                    messages = [{"role": "user", "content": request.prompt}]
                else:
                    messages = request.prompt

                gene_config = GeneConfig(
                    top_p=request.top_p,
                    temperature=request.temperature,
                    total_max_length=request.request_output_len,
                    top_k=request.top_k,
                    stop_token=request.stop,
                )
                session = self.get_session(request.session_id)
                text = session.render_messages(messages, True)
                response, input_ids, result = await self.run_infer(session, gene_config, text=text)

                session.render_messages([{"role": "assistant", "content": response}], False)

                return JSONResponse(
                    {
                        "text": response,
                        "tokens": len(result["tokens"]),
                        "input_tokens": len(input_ids),
                        "history_tokens": result["usage"]["history_tokens"],
                        "finish_reason": result["finish_reason"],
                    }
                )
        except Exception as e:  # noqa
            get_logger().error(f"Error in chat_interactive: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e) + "\n" + traceback.format_exc())

    async def generate(self, request: GenerateReqInput):
        session_id = (
            uuid.uuid4().int if request.session_id == -1 else request.session_id
        )  # release session by interactive api
        session = self.get_session(session_id)

        gene_config = GeneConfig(
            top_p=request.top_p,
            temperature=request.temperature,
            total_max_length=request.max_tokens,
            top_k=request.top_k,
            stop_token=request.stop,
            max_entropy=request.max_entropy,
        )

        response, _, result = await self.run_infer(session, gene_config, input_tokens=request.input_ids)

        if request.session_id == -1:
            await self.release_session(session_id)

        return {
            "output_ids": result["tokens"],
            "text": response,
            "meta_info": {
                "output_token_logprobs": [(x,) for x in result["log_prob"]],
                "entropy": result["entropy"],
                "finish_reason": {"type": result["finish_reason"]},
            },
        }

    async def abort_request(self, request: dict | None = None):
        if request is None:
            request = {}
        await self.run_on_engine(ReleaseItem(-1))
        return {"status": "success"}

    # apis for proxy

    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    async def available_models(self):
        """Show available models."""
        return ModelList(data=[ModelCard(id=self.model_name, root=self.model_name)])

    # stop server

    async def stop_server(self):
        if self.task is not None:
            self.server.should_exit = True
            try:
                await asyncio.wait_for(self.task, timeout=10)
            except TimeoutError:
                get_logger().error("API server did not stop within timeout, forcing shutdown")
                self.server.force_exit = True
                try:
                    await self.task
                except Exception as e:  # noqa: BLE001
                    get_logger().error(f"Error while forcing API server shutdown: {e}")
            self.task = None
            self.server = None
            get_logger().info(f"API server on {self.port} stopped successfully")

    async def wait_closed(self):
        if self.task is not None:
            await self.task
            self.task = None
            self.server = None
        else:
            print("Server is not running, nothing to wait for.")

    # protocol for infer

    async def run_infer(self, session: SessionItem, gene_config: GeneConfig, text=None, input_tokens=None):
        # update stop token

        gene_config = await self.tokenize_stop_tokens(gene_config)
        if text is not None:
            assert input_tokens is None, "text and tokens cannot be both provided"
            input_tokens = await asyncio.get_running_loop().run_in_executor(None, self.tokenizer.encode, text)
        gene_config.total_max_length += len(input_tokens) + session.forwarded_tokens

        # generate
        if gene_config.total_max_length <= 0:
            return "", [], [], "length", []
        else:
            infer_item = InferItem(session.session_id, input_tokens, gene_config)
            result = await self.run_on_engine(infer_item)

            decode_text = await asyncio.get_running_loop().run_in_executor(
                None,
                self.tokenizer.decode,
                result["tokens"][:-1] if result["finish_reason"] == "stop" else result["tokens"],
            )
            session.forward_tokens(len(input_tokens) + len(result["tokens"]))
            return decode_text, input_tokens, result

    async def tokenize_stop_tokens(self, gene_config: GeneConfig):
        gene_config.stop_token = [
            (await asyncio.get_running_loop().run_in_executor(None, self.tokenizer.encode, token))[-1]
            for token in gene_config.stop_token
        ]
        gene_config.stop_token.extend(self.default_stop_tokens)
        return gene_config

    async def run_on_engine(self, item: TaskItem):
        await self.queue.put(item)
        await item.finished_event.wait()
        return item._result["item"]
