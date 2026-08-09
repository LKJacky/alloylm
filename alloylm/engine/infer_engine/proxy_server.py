import asyncio
import importlib.util
import json
import os
import signal
import traceback

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel as PydanticBaseModel

from alloylm.engine.infer_engine.utils import get_logger

# method


_collected_error = set()


def report_error_once(message: str):
    global _collected_error
    global logger
    if message not in _collected_error:
        get_logger().critical(message)
        _collected_error.add(message)


async def simple_forward_request(request_content: dict, url):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as client:
        async with client.post(url, json=request_content, timeout=7200) as response:
            assert response.status == 200, f"Request failed with status {response.status}, {await response.text()}"
            text = await response.text()
            try:
                return json.loads(text)
            except Exception:
                return text


PROXY_METHOD_MAPPING = {"default": simple_forward_request}


# for server add
class ServerAddInfo(PydanticBaseModel):
    models: list[str]
    role: int | None = 0


class ServerAddRequest(PydanticBaseModel):
    url: str
    status: ServerAddInfo


# for  scheduler


class ProxyServer:
    def __init__(self, port, ip="0.0.0.0", method="default"):
        self.port = port
        self.ip = ip if ip is not None else os.popen("hostname -I").read().strip().split()[0]
        self.servers = {}

        self.server = None
        self.task = None
        self.method_name = method
        if method.endswith(".py"):
            spec = importlib.util.spec_from_file_location("custom_method", os.path.abspath(method))
            custom_method = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_method)
            self.method = custom_method.method
        else:
            self.method = PROXY_METHOD_MAPPING[method]

    def get_ip(self):
        return self.ip

    # route schedule

    def acquire_server_url(self, model):
        servers = self.servers.get(model, None)
        if servers:
            servers = list(servers.items())
            servers.sort(key=lambda x: x[1])  # Sort by running count
            server_url = servers[0][0]  # Get the server with the least
            self.servers[model][server_url] += 1  # Increment running count
            return server_url
        else:
            return None

    def release_server_url(self, model, url):
        if model in self.servers and url in self.servers[model]:
            self.servers[model][url] -= 1  # Decrement running count

    # execute

    # apis
    async def launch(self):
        app = FastAPI(title="OpenAI-compatible API")

        app.post("/v1/chat/completions")(self.chat_completion)
        app.post("/v1/chat/interactive")(self.chat_interactive_v1)
        app.post("/generate")(self.generate)
        app.post("/abort_request")(self.abort_request)
        app.post("/nodes/add")(self.add)
        app.post("/nodes/remove")(self.remove)
        app.get("/v1/models")(self.models)

        config = uvicorn.Config(app, host=self.ip, port=self.port, log_level="error", timeout_keep_alive=3600)
        server = uvicorn.Server(config)
        loop = asyncio.get_event_loop()
        task = loop.create_task(server.serve())

        # Wait for server to start
        while not server.started:
            await asyncio.sleep(0.01)

        self.task = task
        self.server = server

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.force_stop(signal.SIGTERM)))
            loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.force_stop(signal.SIGINT)))
        except Exception as e:
            get_logger().warning(f"Failed to set signal handlers: {e}")
        get_logger().info(f"Launched Proxy Server running on {self.ip}:{self.port} with {self.method_name} method")

    async def chat_completion(self, request: Request):
        return await self.transmit(request, post_url="v1/chat/completions")

    async def chat_interactive_v1(self, request: Request):
        if not hasattr(self, "interactive_sessions"):
            self.interactive_sessions = {}
        request_content = await request.json()
        session_id = request_content.get("session_id")
        if session_id in self.interactive_sessions:
            model_name, server_url = self.interactive_sessions[session_id]
            if request_content.get("interactive_mode", False) is False:  # release session
                self.interactive_sessions.pop(session_id)
                self.release_server_url(model_name, server_url)
        elif request_content.get("interactive_mode", False) is True:
            model_name = request_content.get("model", self.servers.keys().__iter__().__next__())
            server_url = self.acquire_server_url(model_name)
            self.interactive_sessions[session_id] = (model_name, server_url)
        else:
            server_url = None  # not in interactive mode, normal acquire

        return await self.transmit(request, post_url="v1/chat/interactive", server_url=server_url)

    async def generate(self, request: Request):
        if not hasattr(self, "interactive_sessions"):
            self.interactive_sessions = {}
        request_content = await request.json()
        session_id = request_content.get("session_id")
        if session_id != -1:
            if session_id in self.interactive_sessions:
                model_name, server_url = self.interactive_sessions[session_id]
            else:
                model_name = request_content.get("model", self.servers.keys().__iter__().__next__())
                server_url = self.acquire_server_url(model_name)
                self.interactive_sessions[session_id] = (model_name, server_url)
        else:
            server_url = None  # not in interactive mode, normal acquire

        return await self.transmit(request, post_url="generate", server_url=server_url)

    async def abort_request(self, request: Request):
        all_urls = {url for servers in self.servers.values() for url in servers.keys()}
        futures = [
            asyncio.create_task(self.transmit(request, post_url="abort_request", server_url=url)) for url in all_urls
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        return results[0]

    async def transmit(self, request: Request, post_url, server_url=None, request_content=None):
        if request_content is None:
            request_content = await request.json()

        manage_url = server_url is None
        model_name = None
        try:
            if manage_url:
                model_name = request_content.get("model", self.servers.keys().__iter__().__next__())
                server_url = self.acquire_server_url(model_name)
            if server_url is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"No available server for the requested model: {model_name}",
                )

            # Create task and wait with disconnect check
            return await self.method(request_content, server_url + "/" + post_url)
        except HTTPException as e:
            raise e
        except Exception as e:
            report_error_once(f"Error in transmit: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error in transmit: {e}\n{traceback.format_exc()}")
        finally:
            if manage_url and model_name and server_url:
                self.release_server_url(model_name, server_url)

    async def add(self, request: ServerAddRequest):
        for model in request.status.models:
            if model not in self.servers:
                self.servers[model] = {}
            if request.url not in self.servers[model]:
                self.servers[model][request.url] = 0
        get_logger().info(f"Added server {request.url} for models {request.status.models}")
        return {"status": "added"}

    async def remove(self, request: ServerAddRequest):
        exist = False
        for model in request.status.models:
            if model in self.servers and request.url in self.servers[model]:
                self.servers[model].pop(request.url)
                exist = True
        if not exist:
            get_logger().info("No such server to remove.")
            return {"status": "not exist"}
        else:
            get_logger().info(f"Remove server {request.url} for models {request.status.models}")
            return {"status": "removed"}

    async def models(self):
        return {"data": [{"id": model, "object": "model"} for model in self.servers.keys()]}

    # stop server

    async def stop_server(self):
        if self.task is not None:
            self.server.should_exit = True
            await self.task
            self.task = None
            self.server = None
            get_logger().info(f"Proxy server on {self.port} stopped successfully")

    async def wait_closed(self):
        if self.task is not None:
            await self.task
            self.task = None
            self.server = None
        else:
            print("Server is not running, nothing to wait for.")

    async def force_stop(self, signum):
        """Force stop the server."""
        print(f"Received signal {signum}, stopping server...")
        await self.stop_server()
        print("Server stopped.")
