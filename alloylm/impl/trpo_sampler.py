import heapq
import pickle
import random
from typing import Any

import numpy as np
from pydantic import BaseModel

from alloylm.algorithm.base import Dataset, TaskData
from alloylm.algorithm.rl.rl_algo import TaskInfo, get_logger


class StdTaskInfo(BaseModel):
    index: int
    rewards: list[float] = []
    recent_n: int = 2
    callback: Any | None = None
    random_order: float = -1.0

    def model_post_init(self, context):
        self.random_order = random.random()

    def std(self):
        if len(self.rewards) < 2 or len(self.rewards) < self.recent_n:
            std = 2**31 - 1 - len(self.rewards)
        else:
            std = np.std(self.rewards[-self.recent_n :])
        return (-std, self.random_order)  # <=0, make random order for items with the same std.

    def __lt__(self, other: "StdTaskInfo"):
        return self.std() < other.std()

    def __hash__(self):
        return self.index

    def model_dump(self, *, exclude=("callback", "random_order"), **kwargs):
        return super().model_dump(exclude=exclude, **kwargs)


class StdTaskHeap:
    def __init__(self, num_tasks, rng: random.Random):
        self.num_tasks = num_tasks
        self.rng = rng
        self.recent_n = 2

        self.heap = [StdTaskInfo(index=i, recent_n=self.recent_n) for i in range(num_tasks)]
        self.rng.shuffle(self.heap)
        heapq.heapify(self.heap)

        self.unfinished = set()

    def pop(self):
        item = heapq.heappop(self.heap)

        if item.std()[0] == 0.0:  # all items have zero std.
            self.recent_n += 1
            get_logger().info(f"All items have zero std, increase recent_n to {self.recent_n}.")
            for task_info in self.heap + list(self.unfinished):
                task_info.recent_n = self.recent_n
            self.rng.shuffle(self.heap)
            heapq.heapify(self.heap)
        self.unfinished.add(item)

        def callback(data: list[TaskData]):
            item.rewards.extend([d.metric for d in data])
            self.push(item)
            self.unfinished.remove(item)

        item.callback = callback
        return item

    def push(self, task_info: StdTaskInfo):
        task_info.random_order = self.rng.random()
        heapq.heappush(self.heap, task_info)

    def checkpoint(self):
        return [x.model_dump() for x in self.heap + list(self.unfinished)]

    def resume(self, state):
        self.heap = [StdTaskInfo.model_validate(x) for x in state]
        assert len(self.heap) == self.num_tasks, (
            "The number of tasks in the checkpoint does not match the expected number."
        )
        self.unfinished = set()
        heapq.heapify(self.heap)


class TRPO_Sampler:
    def __init__(self, dataset: Dataset, tb_writer):
        self.dataset = dataset
        self.name = dataset.config.name
        self.rng = random.Random()
        self.wait = StdTaskHeap(num_tasks=len(self.dataset), rng=self.rng)

        self.epoch = 0
        self.tb_writer = tb_writer

    def __iter__(self):
        while True:
            item = self.wait.pop()
            task = self.dataset[item.index]
            task.task_data.others["origin_dataset"] = self.dataset.config.name

            yield TaskInfo(task_item=task, callback=item.callback)

    def checkpoint(self):
        return {
            "epoch": self.epoch,
            "wait": self.wait.checkpoint(),
            "rng": pickle.dumps(self.rng.getstate()).hex(),
        }

    def resume(self, state):
        self.wait.resume(state["wait"])
        self.epoch = state["epoch"]
        self.rng.setstate(pickle.loads(bytes.fromhex(state["rng"])))
