# A simple example of agent
from openai import AsyncClient

from alloylm.algorithm.base import Dataset, DatasetConfig, Task, TaskData, TaskItem
from alloylm.impl.agent.agent import BaseAgent
from alloylm.impl.agent.env import BaseEnv


class CountToNEnv(BaseEnv):
    """An environment whose counter can only be changed through its tools."""

    def __init__(self, target: int, expose_get_tool_definitions: bool = False):
        super().__init__(expose_get_tool_definitions=expose_get_tool_definitions)
        assert target > 0, "target must be positive"
        self.target = target
        self.count = 0

    @BaseEnv.node_tool(
        description="Add 1 to the counter and return its new value. Keep calling this tool while the value reaches the target. "
    )
    def increment(self) -> int:
        self.count += 1
        output = f"current count: {self.count}, target: {self.target}"
        return output

    @BaseEnv.node_tool(
        description="Subtract 1 from the counter and return its new value. Keep calling this tool while the value reaches the target. "
    )
    def decrement(self) -> int:
        self.count -= 1
        output = f"current count: {self.count}, target: {self.target}"
        return output

    @BaseEnv.node_tool(description="Return the current counter value")
    def get_count(self) -> int:
        return self.count


class CountToNDataset(Dataset):
    """A single-item dataset that asks an agent to count to a target."""

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        target = index + 1
        return TaskItem(
            task_data=TaskData(
                id=f"{self.config.name}_{target}",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You control a counter only through tool calls. Call increment or decrement to change the counter. Your goal is to reach the target value. You can also call get_count to check the current value of the counter."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Count from 0 to {target}, then finish.",
                    },
                ],
                others={"target": target},
                infer_args=self.config.infer_args,
            ),
            task_cls=self.config.task_cls,
        )

    def __len__(self):
        return self.config.max_target - 1


class CountToNTask(Task):
    @classmethod
    async def infer(cls, task_data: TaskData) -> TaskData:
        target = task_data.others["target"]
        env = CountToNEnv(target)
        client = task_data.infer_args.get_client(client_type=AsyncClient)
        agent = BaseAgent(client, env=env, max_steps=target * 2)

        try:
            task_data.messages = await agent.solve(messages=task_data.messages)

            task_data.messages = agent.messages
            task_data.finish_reason = agent.finish_reason
            task_data.input_tokens = sum(agent.used_tokens[::2])
            task_data.output_tokens = sum(agent.used_tokens[1::2])
            task_data.total_tokens = sum(agent.used_tokens)
            task_data.others.update(count=env.count)
        except TimeoutError as e:
            raise RuntimeError(
                f"Inference timeout {task_data.infer_args.sample_args.get('timeout', 'unknow')} seconds"
            ) from e
        finally:
            await client.close()
        return task_data

    @classmethod
    async def eval(cls, task_data: TaskData) -> TaskData:
        task_data.metric = 1 if task_data.others["count"] == task_data.others["target"] else 0
        return task_data


class CountToNDatasetConfig(DatasetConfig):
    dataset_cls: object = CountToNDataset
    name: str = "count_to_n"
    task_cls: object = CountToNTask
    max_target: int = 20

    def model_post_init(self, context):
        self.name += f"_{self.max_target}"
        return super().model_post_init(context)
