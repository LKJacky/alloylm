import asyncio

from alloylm.algorithm.base import InferArgs
from alloylm.algorithm.eval.base import EvalRunner
from alloylm.impl.count_to_n import CountToNDatasetConfig
from alloylm.impl.engines.qwen import QWEN_TOOL_PATTERN, Qwen3ChatTemplate
from alloylm.impl.math import GSM8KDatasetConfig, GSM8KTask
from alloylm.test_utils import CudaAsyncTestCase, LaunchTestServer


class TestTask(CudaAsyncTestCase):
    async def test_gsm8k_task(self):
        async with LaunchTestServer():
            dataset = await GSM8KDatasetConfig(infer_args=InferArgs(model_name="ALLOYLM")).build()
            task_item = dataset[0]
            task_data = await GSM8KTask.run_and_eval(task_item.task_data)
            self.assertTrue(task_data.metric == 1.0)

    async def test_count_to_n_task(self):
        async with LaunchTestServer(
            model_path="Qwen/Qwen3-0.6B", tool_pattern=QWEN_TOOL_PATTERN, chat_template=Qwen3ChatTemplate
        ):  # qwen3 has better agentic performance Qwen2.5, hence we use it for this test
            dataset = await CountToNDatasetConfig(
                max_target=32,
                infer_args=InferArgs(
                    model_name="ALLOYLM",
                    sample_args={"temperature": 1.0, "max_tokens": 1024, "extra_body": {"top_k": 1}},
                ),
            ).build()
            runner = EvalRunner(dataset, work_dir="work_dirs/tests/test_eval/")
            summary = await runner.run(asyncio.Semaphore(64))
            print("accuracy:", summary.metric)

            self.assertGreater(summary.metric, 0.6)
