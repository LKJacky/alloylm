# inference engine
import asyncio
import os
from asyncio import Queue
from contextlib import asynccontextmanager, contextmanager

from pydantic import BaseModel as PydanticBaseModel
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.proxy_server import ProxyServer
from alloylm.engine.infer_engine.utils import get_free_port
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.utils import get_logger

from .api_server import APIServer
from .scheduler import SchedulerServer
from .utils import GatherContext

logger = get_logger()

# inference engine


class InferEngineConfig(PydanticBaseModel):
    model_name: str
    memory_usage: float = 0.8
    chat_template: str | None = None
    max_prefill_length: int = 16 * 1024
    port: int | None = None
    proxy_url: str | None = None


class InferEngine:
    def __init__(
        self,
        model,
        tokenizer,
        engine_config: InferEngineConfig,
    ):
        self.tokenizer = tokenizer

        # scheduler server
        task_queue = Queue()

        cache = model.create_cache(memory_usage=engine_config.memory_usage)
        real_vocab_size = model.get_real_vocab_size(tokenizer)
        get_logger().info(f"Real vocab size: {real_vocab_size}")
        self.scheduler_server = SchedulerServer(
            model=model,
            cache=cache,
            max_prefill_length=engine_config.max_prefill_length,
            task_queue=task_queue,
            real_vocab_size=real_vocab_size,
        )

        if engine_config.port is None:
            engine_config.port = get_free_port()

        # proxy server
        if engine_config.proxy_url is None:
            if dist.get_world_size() > 1:
                if dist.get_rank() == 0:
                    proxy_port = engine_config.port if engine_config.port is not None else get_free_port()
                    proxy_url = f"http://{os.environ.get('MASTER_ADDR')}:{proxy_port}"
                    self.proxy_server = ProxyServer(port=proxy_port)
                else:
                    proxy_url = None
                    self.proxy_server = None
                proxy_url_list = [proxy_url]
                dist.broadcast_object_list(proxy_url_list, src=0)
                proxy_url = proxy_url_list[0]
                server_port = get_free_port()
                self.url = f"{proxy_url}/v1"
            else:
                self.proxy_server = None
                proxy_url = None
                server_port = engine_config.port
                self.url = f"http://127.0.0.1:{server_port}/v1"
        else:
            server_port = engine_config.port
            self.proxy_server = None
            proxy_url = engine_config.proxy_url
            self.url = f"http://127.0.0.1:{server_port}/v1"

        # api server
        self.api_server = APIServer(
            tokenizer,
            chat_template=engine_config.chat_template,
            queue=task_queue,
            port=server_port,
            proxy_url=proxy_url,
            model_name=engine_config.model_name,
        )

        self.gather_context = GatherContext(model)
        self.paused = False

    async def launch(self):
        queue = Queue()  # create a new queue for each launch to avoid interference from previous runs
        self.scheduler_server.wait_queue = queue
        self.api_server.queue = queue
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
        await self.scheduler_server.launch()
        if not self.paused:
            if self.proxy_server:
                await self.proxy_server.launch()
            await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
            await self.api_server.launch()
        self.paused = False

    async def stop(self):
        if not self.paused:
            await self.scheduler_server.stop_server()
        await self.api_server.stop_server()
        if self.proxy_server:
            await self.proxy_server.stop_server()
        self.scheduler_server.model.train_shard()

    async def pause(self):
        await self.scheduler_server.stop_server()
        self.paused = True

    @contextmanager
    def gather(self):
        with self.gather_context:
            yield self

    @asynccontextmanager
    async def serve(self):
        await self.launch()
        await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
        try:
            yield self
        finally:
            await asyncio.get_event_loop().run_in_executor(None, dist.barrier)
            await self.pause()

    async def wait_closed(self):
        await self.api_server.wait_closed()


class SPMDInferConfig(PydanticBaseModel):
    llm_config: AlloyLMModelConfig
    infer_engine_config: InferEngineConfig


class SPMDInfer(InferEngine):
    def __init__(self, config: SPMDInferConfig):
        self.model = config.llm_config.build()
        super().__init__(
            model=self.model,
            tokenizer=AutoTokenizer.from_pretrained(config.llm_config.path, trust_remote_code=True),
            engine_config=config.infer_engine_config,
        )
