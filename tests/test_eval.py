import asyncio
import os
import unittest

from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.eval.base import EvalConfig, EvalRunner, run_eval
from alloylm.impl.math import GSM8KDatasetConfig, GSM8KTask
from alloylm.test_utils import (
    CudaAsyncTestCase,
    LaunchTestServer,
    check_cuda_leak,
)


class TestTask(CudaAsyncTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = LaunchTestServer()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.engine
        check_cuda_leak()

    async def test_eval_runner(self):
        async with self.__class__.engine:
            dataset = await GSM8KDatasetConfig(
                task_cls=GSM8KTask,
                infer_args=InferArgs(
                    model_name="ALLOYLM",
                    sample_args={
                        "temperature": 0.0,
                        "extra_body": {"top_k": 1},
                    },
                ),
            ).build()
            dataset.data = [dataset.data[i] for i in range(32)]
            runner = EvalRunner(dataset, work_dir="work_dirs/tests/test_eval/")
            summary = await runner.run(asyncio.Semaphore(512))
            print("accuracy:", summary.metric)
            self.assertTrue(summary.metric > 0.4)
            self.assertGreater(summary.mean_input_tokens, 10)
            self.assertGreater(summary.mean_output_tokens, 100)

    @unittest.skipUnless(os.environ.get("ENABLE_LONG_RUNNING_TESTS", "0") == "1", "Skipping long-runing test")
    async def test_run_eval(self):
        async with self.__class__.engine:
            await run_eval(
                EvalConfig(
                    datasets=[
                        GSM8KDatasetConfig(
                            task_cls=GSM8KTask,
                            infer_args=InferArgs(
                                sample_args={
                                    "temperature": 1.0,
                                    "max_tokens": 4096,
                                    "top_p": 1.0,
                                    "extra_body": {"top_k": 1},
                                }
                            ),
                        ),
                    ],
                    work_dir="work_dirs/tests/test_eval/",
                    concurrency=512,
                    resume=False,
                )
            )


class TestSglangTask(CudaAsyncTestCase):
    class LaunchSGLangServer:
        """Launch an sglang OpenAI-compatible server as an async context
        manager.

        Mirrors :class:`LaunchTestServer` but serves the model through sglang so the
        eval pipeline can run against sglang. On enter it spawns
        ``python -m sglang.launch_server`` on a free port and waits until the
        ``/health`` endpoint is ready; on exit it terminates the process.

        Use :attr:`infer_args_kwargs` to point an :class:`InferArgs` at this server.
        """

        def __init__(
            self,
            model_path="Qwen/Qwen3-0.6B",
            served_model_name="ALLOYLM",
            host="127.0.0.1",
            port=8000,
            mem_fraction_static=0.4,
            extra_args="",
        ):
            self.model_path = model_path
            self.served_model_name = served_model_name
            self.host = host
            self.port = port
            self.mem_fraction_static = mem_fraction_static
            self.extra_args = extra_args
            self.process = None
            self.base_url = None

        async def __aenter__(self):
            import asyncio

            from sglang.utils import launch_server_cmd, wait_for_server

            command = (
                "python -m sglang.launch_server "
                f"--model-path {self.model_path} "
                f"--served-model-name {self.served_model_name} "
                f"--host {self.host} "
                f"--mem-fraction-static {self.mem_fraction_static} "
                f"{self.extra_args}"
            ).strip()
            self.process, self.port = launch_server_cmd(command, host=self.host, port=self.port)
            self.base_url = f"http://{self.host}:{self.port}"
            # wait_for_server polls synchronously; run it off the event loop.
            await asyncio.to_thread(wait_for_server, self.base_url)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            from sglang.utils import terminate_process

            if self.process is not None:
                terminate_process(self.process)
                self.process = None

        @property
        def infer_args_kwargs(self) -> dict:
            """Kwargs to build an ``InferArgs`` pointing at this server."""
            return {
                "model_url": f"{self.base_url}/v1",
                "model_name": self.served_model_name,
            }

        def __del__(self):
            try:
                if self.process is not None:
                    from sglang.utils import terminate_process

                    terminate_process(self.process)
                    self.process = None
            except Exception:  # noqa
                pass

    @unittest.skipUnless(os.environ.get("ENABLE_LONG_RUNNING_TESTS", "0") == "1", "Skipping long-runing test")
    async def test_run_eval_sglang(self):
        async with self.__class__.LaunchSGLangServer(model_path="Qwen/Qwen3-0.6B") as server:
            await run_eval(
                EvalConfig(
                    datasets=[
                        GSM8KDatasetConfig(
                            task_cls=GSM8KTask,
                            infer_args=InferArgs(
                                **server.infer_args_kwargs,
                                sample_args={
                                    "temperature": 1.0,
                                    "max_tokens": 4096,
                                    "top_p": 1.0,
                                    "extra_body": {"top_k": 1},
                                },
                            ),
                        ),
                    ],
                    work_dir="work_dirs/tests/test_eval_sglang/",
                    concurrency=512,
                    resume=False,
                )
            )
