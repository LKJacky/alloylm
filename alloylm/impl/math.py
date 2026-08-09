from datasets import load_dataset
from math_verify import parse, verify

from alloylm.algorithm.base import Dataset, DatasetConfig, TaskData, TaskItem

from .common import OneStepTask, extract_boxed


class GSM8KDataset(Dataset):
    def __init__(self, config: "GSM8KDatasetConfig"):
        super().__init__(config)
        self.data = load_dataset("openai/gsm8k", "main", split=config.split)

    def __getitem__(self, index):
        data = TaskData(
            id=f"{self.config.name}_{index}",
            messages=[
                {"role": "system", "content": "Answer below question and response your final answer in \\boxed"},
                {"role": "user", "content": self.data[index]["question"]},
            ],
            others={"answer": self.data[index]["answer"].split("#### ")[-1].strip()},
            infer_args=self.config.infer_args,
        )
        return TaskItem(
            task_data=data,
            task_cls=self.config.task_cls,
        )

    def __len__(self):
        return len(self.data)


class GSM8KTask(OneStepTask):
    @classmethod
    async def eval(cls, task_data: TaskData) -> TaskData:
        def verify_math(x: str, gt: str):
            return verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                parse(f"\\boxed{{{x}}}", parsing_timeout=5),
                timeout_seconds=5,
            )

        assert task_data.messages[-1]["role"] == "assistant"
        answer = extract_boxed(task_data.messages[-1]["content"])
        gt = task_data.others["answer"]
        task_data.metric = 1.0 if verify_math(answer, gt) else 0.0
        return task_data


class GSM8KDatasetConfig(DatasetConfig):
    dataset_cls: object = GSM8KDataset
    name: str = "gsm8k"
    task_cls: object = GSM8KTask
    split: str = "test"

    def model_post_init(self, context):
        self.name += "_" + self.split
