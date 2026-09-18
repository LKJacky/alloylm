import asyncio

from alloylm.algorithm.base import InferArgs
from alloylm.engine.infer_engine.infer_bank import InferBank
from alloylm.impl.count_to_n import CountToNDatasetConfig
from alloylm.impl.engines.qwen.chat_template import (
    QWEN_THINKING_PATTERN,
    QWEN_TOOL_PATTERN,
    Qwen3ChatTemplate,
)
from alloylm.impl.math import GSM8KDatasetConfig, GSM8KTask
from alloylm.test_utils import CudaAsyncTestCase, LaunchTestServer
from alloylm.utils import get_logger, write_jsonl


class TestTask(CudaAsyncTestCase):
    async def test_gsm8k_task(self):
        async with LaunchTestServer():
            dataset = await GSM8KDatasetConfig(
                infer_args=InferArgs(
                    model_name="ALLOYLM",
                    sample_args={"temperature": 1.0, "max_tokens": 1024, "extra_body": {"top_k": 1}},
                )
            ).build()
            task_item = dataset[0]
            task_data = await GSM8KTask.run_and_eval(task_item.task_data)
            self.assertTrue(task_data.metric == 1.0)

    async def test_count_to_n_task(self):
        async with LaunchTestServer(
            model_path="Qwen/Qwen3-0.6B",
            tool_pattern=QWEN_TOOL_PATTERN,
            chat_template=Qwen3ChatTemplate,
            thinking_pattern=QWEN_THINKING_PATTERN,
        ) as server:  # qwen3 has better agentic performance Qwen2.5, hence we use it for this test
            for thinking, min_metric in (
                (False, 0.1),
                (True, 0.01),
            ):
                dataset = await CountToNDatasetConfig(
                    max_target=32,
                    infer_args=InferArgs(
                        model_name="ALLOYLM",
                        interactive_mode=True,
                        sample_args={
                            "temperature": 1.0,
                            "max_tokens": 4096,
                            "extra_body": {
                                # "top_k": 1,
                                "for_training": True,
                                "chat_template_kwargs": {"thinking": thinking},
                            },
                        },
                    ),
                ).build()
                results = await asyncio.gather(*(item.task_cls.run_and_eval(item.task_data) for item in dataset))
                metric = sum(item.metric for item in results) / len(results)
                write_jsonl("work_dirs/tests/count_to_n/data.jsonl", [x.model_dump() for x in results])

                self.assertTrue(metric > min_metric, f"Expected metric > {min_metric}, but got {metric}")
                infer_info = (await server.engine.fetch_infer_info())[0]
                for task_data in results:
                    message_hash = InferBank.hash_messages(task_data.messages)
                    self.assertIn(message_hash, infer_info)
                get_logger().info(f"metric={metric}")
