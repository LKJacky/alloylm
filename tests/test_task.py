from alloylm.algorithm.base import InferArgs
from alloylm.impl.math import GSM8KDatasetConfig, GSM8KTask
from alloylm.test_utils import CudaAsyncTestCase, LaunchTestServer


class TestTask(CudaAsyncTestCase):
    async def test_gsm8k_task(self):
        async with LaunchTestServer():
            dataset = await GSM8KDatasetConfig(infer_args=InferArgs(model_name="ALLOYLM")).build()
            task_item = dataset[0]
            task_data = await GSM8KTask.run_and_eval(task_item.task_data)
            self.assertTrue(task_data.metric == 1.0)
