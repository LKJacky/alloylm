from typing import Any, Optional

import aiohttp
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str | list[dict[str, Any]]
    image_url: str | list[str] | None = None
    session_id: int = -1
    interactive_mode: bool = False
    stream: bool = False
    stop: str | list[str] | None = None
    request_output_len: int | None = None
    top_p: float = 0.8
    top_k: int = 40
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    cancel: bool | None = False
    adapter_name: str | None = None
    seed: int | None = None
    min_new_tokens: int | None = None
    min_p: float = 0.0


class DeployClient:
    def __init__(self, api_server_url, api_key=None, **kwargs):
        self.api_server_url = api_server_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        connector = aiohttp.TCPConnector(limit=8192)
        self.session = aiohttp.ClientSession(headers=self.headers, connector=connector)

    async def end_session(self, session_id):
        async with self.session.post(
            self.api_server_url + "/v1/chat/interactive",
            json=dict(prompt="", session_id=session_id, request_output_len=0, interactive_mode=False),
            timeout=7200,
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            await response.text()

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
        async with self.session.post(
            self.api_server_url + "/v1/chat/interactive", json=request.model_dump(), timeout=7200
        ) as response:
            assert response.status == 200, (
                f"Request failed with status code {response.status}, {response}, {response.text()}"
            )
            result = await response.json()
        return result

    async def close(self):
        await self.session.close()
