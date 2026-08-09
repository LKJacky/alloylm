import numpy as np
from pydantic import BaseModel


class InferArgs(BaseModel):
    model_url: str = "http://127.0.0.1:8000/v1"
    model_name: str = "ALLOYLM"

    sample_args: dict = {
        "temperature": 1.0,
        "max_tokens": 2048,
        "top_p": 1.0,
        "extra_body": {
            "top_k": 1,
        },
    }
    others: dict = {}


class DatasetSummary(BaseModel):
    dataset_name: str
    task_data: list["TaskData"]
    metric: float = -1.0

    num_overlong: int = -1
    mean_input_tokens: int = -1
    mean_output_tokens: int = -1

    origin_num: int = 0
    result_num: int = 0

    # others
    others: dict = {}

    def model_post_init(self, context):
        self.metric = np.mean([d.metric for d in self.task_data])
        if len(self.task_data) > 0:
            self.mean_input_tokens = int(sum([d.input_tokens for d in self.task_data]) / len(self.task_data))
            self.mean_output_tokens = int(sum([d.output_tokens for d in self.task_data]) / len(self.task_data))
        self.num_overlong = len([d for d in self.task_data if d.finish_reason != "stop"])

        self.result_num = len(self.task_data)


# Task


class Task:
    @classmethod
    async def infer(cls, task_data: "TaskData") -> "TaskData":
        raise NotImplementedError()

    @classmethod
    async def eval(cls, TaskData: "TaskData") -> "TaskData":
        raise NotImplementedError()

    @classmethod
    async def run_and_eval(cls, task_data: "TaskData") -> "TaskData":
        task_data = await cls.infer(task_data)
        task_data = await cls.eval(task_data)
        return task_data


class TaskData(BaseModel):
    id: str
    infer_args: InferArgs = InferArgs()

    # message
    messages: list[dict]
    finish_reason: str = ""

    # tokens
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # for eval
    metric: float = -1.0

    # others: original data, extract answer
    others: dict = {}


class TaskItem(BaseModel):
    task_data: TaskData
    task_cls: type[Task]


# Dataset


class Dataset:
    def __init__(self, config: "DatasetConfig"):
        self.config = config

    def __getitem__(self, index) -> TaskItem:
        raise NotImplementedError()

    def summary(self, data: list[TaskData]) -> DatasetSummary:
        return DatasetSummary(
            dataset_name=self.config.name,
            task_data=data,
            origin_num=len(self),
            result_num=len(data),
        )

    async def lazy_init(self):
        pass  # do nothing by default


class DatasetConfig(BaseModel):
    name: str

    task_cls: object
    dataset_cls: object

    infer_args: InferArgs = InferArgs()

    async def build(self) -> "Dataset":
        dataset = self.dataset_cls(self)
        await dataset.lazy_init()
        return dataset


# Runner


class Runner:
    def __init__(self, dataset_configs: list[DatasetConfig]):
        self.datasets = {config.name: config.build() for config in dataset_configs}

    async def run(self) -> list[DatasetSummary]:
        pass
