import asyncio
import functools
import inspect
import os
import sys

import ray
from pydantic import BaseModel as PydanticBaseModel
from torch import distributed as dist

from alloylm.utils import get_free_port, init_ray
from alloylm.utils import get_logger


async def run_by_func_name(self, method, args, kwargs):
    """Run ``method(*args, **kwargs)`` on the local actor instance.

    Injected onto the wrapped actor class so the driver can dispatch arbitrary methods over Ray. Defined at module
    level so re-wrapping the same actor class is idempotent (stable function identity).
    """
    func = getattr(self, method)
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


class _RaySPMDWorker:
    def __init__(self, actor_cls, args, kwargs):
        self.actor_cls = actor_cls
        self.args = args
        self.kwargs = kwargs
        self.actor = None

    def get_rendezvous(self):
        return ray.util.get_node_ip_address(), get_free_port()

    def initialize(self, envs):
        os.environ.update(envs)
        self.actor = self.actor_cls(*self.args, **self.kwargs)

    async def run_by_func_name(self, method, args, kwargs):
        return await run_by_func_name(self.actor, method, args, kwargs)


class SPMDActorConfig(PydanticBaseModel):
    world_size: int = 1
    num_gpus: int = 1
    num_cpus: int = 1
    memory: int = 1 * 1024**3


def init_dist():
    if not dist.is_initialized():
        os.environ["RANK"] = os.environ.get("RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "0")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "9000")

        dist.init_process_group(backend="nccl", init_method="env://")


def get_init_func_with_init_dist(origin_init):
    def init_func(*args, **kwargs):
        init_dist()
        print(
            f"SPMDActor: initialized process group with rank {dist.get_rank()} and world size {dist.get_world_size()}"
        )
        return origin_init(*args, **kwargs)

    # Preserve the original ``__init__`` signature so Ray's actor argument
    # validation (which strips the first parameter assuming it is ``self``)
    # sees the real parameters instead of collapsing ``*args`` into ``**kwargs``.
    functools.update_wrapper(init_func, origin_init)

    return init_func


class SPMDActor:
    def __init__(
        self,
        actor_cls,
        args=(),
        kwargs=None,
        # resources per actor
        spmd_config: SPMDActorConfig | None = None,
    ) -> None:
        kwargs = {} if kwargs is None else kwargs
        spmd_config = SPMDActorConfig() if spmd_config is None else spmd_config
        existing = getattr(actor_cls, "run_by_func_name", None)
        assert existing is None or existing is run_by_func_name, (
            "actor_cls already has a run_by_func_name method, which is reserved for SPMDActor"
        )
        if existing is None:
            actor_cls.run_by_func_name = run_by_func_name  # add a method to the actor class
        actor_cls.__init__ = get_init_func_with_init_dist(actor_cls.__init__)  # wrap the init method to init dist

        # Serialize ``actor_cls`` by value so Ray workers don't need to import
        # its defining module. Without this, actor classes defined in a script
        # or test module (e.g. ``test_engine.test_spdm``) raise
        # ``ModuleNotFoundError`` on the workers.
        module = sys.modules.get(actor_cls.__module__)
        if module is not None:
            try:
                ray.cloudpickle.register_pickle_by_value(module)
            except Exception:  # noqa
                pass

        self._workers = []
        if spmd_config.world_size > 1 or os.environ.get("USE_RAY", "0") == "1":
            init_ray()
            worker_cls = ray.remote(_RaySPMDWorker)
            for _ in range(spmd_config.world_size):
                self._workers.append(
                    worker_cls.options(
                        num_gpus=spmd_config.num_gpus,
                        num_cpus=spmd_config.num_cpus,
                        memory=spmd_config.memory,
                    ).remote(actor_cls, args, kwargs)
                )

            master_addr, master_port = ray.get(self._workers[0].get_rendezvous.remote())
            init_refs = []
            for rank, worker in enumerate(self._workers):
                envs = {
                    "RANK": str(rank),
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": str(spmd_config.world_size),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                    "USE_RAY": "1",
                }
                init_refs.append(worker.initialize.remote(envs))
            ray.get(init_refs)
            self.use_ray = True
        else:
            os.environ.update(
                {
                    "RANK": str(0),
                    "LOCAL_RANK": str(0),
                    "WORLD_SIZE": str(1),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(get_free_port()),
                    "USE_RAY": "0",
                }
            )
            self._workers.append(actor_cls(*args, **kwargs))
            self.use_ray = False

    async def _call(self, method, *args, **kwargs):
        try:
            if self.use_ray:
                futures = [w.run_by_func_name.remote(method, args, kwargs) for w in self._workers]
                return await asyncio.gather(*futures)
            else:
                functions = [getattr(w, method) for w in self._workers]
                if inspect.iscoroutinefunction(functions[0]):
                    return await asyncio.gather(*[f(*args, **kwargs) for f in functions])
                else:
                    return [f(*args, **kwargs) for f in functions]
        except BaseException as e:
            get_logger().error(f"Error during SPMD call to method {method}: {e}")
            raise

    def __getattr__(self, name):
        async def remote(*args, **kwargs):
            return await self._call(name, *args, **kwargs)

        return remote

    def shutdown(self) -> None:
        if self.use_ray:
            for w in self._workers:
                ray.kill(w)
            if ray.is_initialized():
                ray.shutdown()

    @classmethod
    def create_spmd_actor(
        cls,
        actor_cls,
        args=(),
        kwargs=None,
        spmd_config: SPMDActorConfig | None = None,
    ) -> "SPMDActor":
        return cls(
            actor_cls=actor_cls,
            args=args,
            kwargs={} if kwargs is None else kwargs,
            spmd_config=SPMDActorConfig() if spmd_config is None else spmd_config,
        )
