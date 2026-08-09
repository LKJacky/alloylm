import asyncio
import copy
import functools
import gc
import json
import math
import operator
import os
import pickle
import random
import shutil
import time
import traceback
from collections import defaultdict, deque
from typing import Any

import aiofiles
import numpy as np
import torch
import tqdm
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from alloylm.algorithm.base import Dataset, DatasetConfig, Task, TaskData, TaskItem
from alloylm.algorithm.rl.utils import get_logger, get_tb_writer
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.hack_client import (
    HighConcurrentClient as AsyncClient,
)
from alloylm.engine.train_engine.train_engine import RLInput
from alloylm.engine.train_engine.train_infer_engine import (
    SpmdTrainInferEngine,
    TrainInferEngineConfig,
)

from ...utils import init_logger
from .utils import DummySummaryWriter, MeasureTime

DEFAULT_MODEL_NAME = "ALLOYLM"
logger = get_logger()
tb_writer = get_tb_writer()


def to_rl_data(task_data: TaskData) -> RLInput:
    rollout_data = task_data.others["rl_data"]
    return RLInput(
        input_ids=rollout_data["input_ids"],
        labels=rollout_data["labels"],
        inference_logprobs=rollout_data["log_probs"],
        advantages=rollout_data["advantages"],
    )


# config


def filter_json_serializable(obj):
    def _is_json_serializable(value):
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError, OverflowError):
            return False

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            filtered = filter_json_serializable(v)
            if _is_json_serializable(filtered):
                result[k] = filtered
        return result
    if isinstance(obj, list):
        result = []
        for v in obj:
            filtered = filter_json_serializable(v)
            if _is_json_serializable(filtered):
                result.append(filtered)
        return result
    return obj


class RLAlgorithmConfig(BaseModel):
    # model and engine
    llm_config: AlloyLMModelConfig
    engine_config: TrainInferEngineConfig

    # Dataset configuration
    datasets: list[DatasetConfig | Any] = []
    train_sample_ratios: list[float] = []  # like sample args

    # Evaluation configuration
    eval_datasets: list[DatasetConfig | Any] = []
    eval_sample_ratio: list[float] = []
    eval_interval: int = 50

    # Algorithm configuration
    roll_out_bs: int = 128
    num_rl_group: int = 8
    max_length: int = 16384
    total_training_steps: int = 800
    checkpoint_interval: int = 10
    max_checkpoints: int = 1
    auto_resume: bool = True

    async_rollout: str = "group"
    filter_group: str = "none"

    data_post_process_func: Any = None

    # infra
    work_dir: str = "./work_dirs/"
    max_concurrency: int = 8192
    model_name: str = DEFAULT_MODEL_NAME

    def model_post_init(self, context):
        assert self.num_rl_group <= self.max_concurrency, (
            "num_rl_group should be larger than max_concurrency to fully utilize async rollout"
        )


# Task proto
class TaskCounter:
    count = 0

    def __enter__(self):
        self.__class__.count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__.count -= 1


class BaseRepeatTask(Task):
    def __init__(self, task: TaskItem, repeat, filter=False):
        self.repeat = repeat
        self.task = task
        self.filter = filter

    async def run_single(self, task_cls: type[Task], task_data: TaskData, semaphore: asyncio.Semaphore):
        task_data = copy.deepcopy(task_data)
        with TaskCounter():
            async with semaphore:
                try:
                    task_data = await task_cls.infer(task_data)
                    task_data = await task_cls.eval(task_data)

                except TimeoutError:
                    raise RuntimeError(f"Task {task_cls.__name__} execute timed out.")
        return task_data

    async def run(self, semaphore):
        try:
            futures = [
                asyncio.get_running_loop().create_task(
                    self.run_single(self.task.task_cls, self.task.task_data, semaphore)
                )
                for _ in range(self.repeat)
            ]
            results = await asyncio.gather(*futures)
            return await self.post_process(results)
        except RuntimeError as e:
            logger.critical(f"Runtime Error: {e}")
            return {"result": [], "filter": "error"}
        except Exception as e:  # noqa: BLE001
            logger.critical(f"BaseRepeatTask encountered an error: {e}\n{traceback.format_exc()}")
            return {"result": [], "filter": "error"}

    async def post_process(self, results: list[TaskData]):
        # compute accuracy
        accuracy = sum([1 if res.metric == 1 else 0 for res in results]) / len(results)

        reward = torch.tensor([res.metric for res in results]).float()

        if self.repeat == 1:
            advantage = reward
            filter_reason = "succeed"
        else:
            if self.filter:
                if (reward == 0).all():
                    filter_reason = "no_pass"
                elif (reward == 1).all():
                    filter_reason = "all_pass"
                else:
                    filter_reason = "succeed"
            else:
                filter_reason = "succeed"

            advantage = (reward - reward.mean()) / (reward.std() + 1e-8).item()
        for i in range(len(results)):
            results[i].others["rl_data"] = {}
            results[i].others["rl_data"]["advantages"] = advantage[i].item()
            results[i].others["rl_data"]["accuracy"] = accuracy
        return {"result": results, "filter": filter_reason}


# sampler


class TaskInfo(BaseModel):
    task_item: TaskItem
    callback: Any = lambda x: None


class TaskSampler:
    def __init__(self, dataset: Dataset, tb_writer):
        self.dataset = dataset
        self.name = dataset.config.name
        self.rng = random.Random()
        self.wait = deque(list(range(len(dataset))))
        self.rng.shuffle(self.wait)
        self.wait.append(None)  # indicate the end of an epoch

        self.epoch = 0
        self.tb_writer = tb_writer

    def __iter__(self):
        while True:
            idx = self.wait.popleft()
            if idx is None:
                self.epoch += 1
                logger.warning(f"{self.name} All tasks finished, reshuffling. Starting epoch {self.epoch}.")
                self.tb_writer.add_scalar(f"sampler/{self.name}_epoch", self.epoch, global_step=self.epoch)
                self.rng.shuffle(self.wait)
                self.wait.append(idx)
            else:
                task = self.dataset[idx]
                task.task_data.others["origin_dataset"] = self.dataset.config.name
                self.wait.append(idx)
                yield TaskInfo(task_item=task)

    def checkpoint(self):
        return {
            "epoch": self.epoch,
            "wait": copy.deepcopy(list(self.wait)),
            "rng": pickle.dumps(self.rng.getstate()).hex(),
        }

    def resume(self, state):
        self.wait = deque(state["wait"] + state.get("finished", []))
        if len(self.wait) != len(self.dataset):
            no_save = list(set(self.wait) - set(list(range(len(self.dataset))) + [None]))
            get_logger().warning(
                f"Resume TaskSampler {self.name} with {len(self.wait)} tasks, expected {len(self.dataset)}. Missing tasks: {no_save}"
            )
        self.epoch = state["epoch"]
        self.rng.setstate(pickle.loads(bytes.fromhex(state["rng"])))


class RatioSampler:
    def __init__(self, datasets: list[Dataset], sample_ratios: list[float], tb_writer):
        self.samplers = [TaskSampler(dataset=d, tb_writer=tb_writer) for d in datasets]
        self.sampler_iters = [iter(s) for s in self.samplers]
        self.sample_ratios = list(sample_ratios)
        self.rng = random.Random()
        self.num_per_epoch = int(len(datasets[0]) / self.sample_ratios[0])

        logger.info(
            f"Initialized RatioSampler with datasets: {[d.config.name for d in datasets]}, with sample ratio: {self.sample_ratios}"
        )

    def __iter__(self):
        while True:
            sampler_idx = self.rng.choices(list(range(len(self.samplers))), weights=self.sample_ratios, k=1)[0]
            item = next(self.sampler_iters[sampler_idx])
            yield item

    def checkpoint(self):
        return {
            "samplers": {s.dataset.config.name: s.checkpoint() for s in self.samplers},
            "rng": pickle.dumps(self.rng.getstate()).hex(),
        }

    def resume(self, state):
        for s in self.samplers:
            s.resume(state["samplers"][s.dataset.config.name])
        self.rng.setstate(pickle.loads(bytes.fromhex(state["rng"])))


# rollout task


async def aquire_semaphore(semaphore: asyncio.Semaphore, num=1):
    try:
        num_acquired = 0
        for _ in range(num):
            await semaphore.acquire()
            num_acquired += 1
    except asyncio.CancelledError:
        for _ in range(num_acquired):
            semaphore.release()
        raise asyncio.CancelledError()
    except Exception as e:  # noqa: BLE001
        logger.critical(f"Error in acquiring semaphore: {e}\n{traceback.format_exc()}")
    return num_acquired


class ReleaseOnlySemaphore:
    def __init__(self, semaphore: asyncio.Semaphore):
        self.semaphore = semaphore

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        self.semaphore.release()

    async def acquire(self):
        pass

    def release(self):
        self.semaphore.release()


class TrainTask(Task):
    def __init__(
        self,
        datasets,
        sample_ratios,
        tb_writer,
        rollout_bs=16,
        repeat=8,
        async_rollout=False,
        filter="none",  # none, discard, resubmit
    ):
        assert filter in ["none", "discard", "resubmit"], "filter must be 'none', 'discard' or 'resubmit'"
        super().__init__()
        self.tb_writer = tb_writer
        self.task_sampler = RatioSampler(datasets, sample_ratios, tb_writer=tb_writer)
        logger.info(f"Initialized TrainTask, about {self.task_sampler.num_per_epoch // rollout_bs} steps per epoch.")
        self.sampler_iter = iter(self.task_sampler)

        self.rollout_bs = rollout_bs
        self.repeat = repeat
        self.async_rollout = async_rollout
        self.filter = filter

        self.running_futures: set[asyncio.Future] = set()
        self.submitted = 0

        self.finish_reason = defaultdict(int)
        self.utilize_ratio = -1.0

    async def produce(
        self, submitted: asyncio.Queue, submit_semaphore: asyncio.Semaphore, engine_semaphore: asyncio.Semaphore
    ):
        while True:
            try:
                await submit_semaphore.acquire()  # concurrent training concurrency
                await aquire_semaphore(engine_semaphore, num=self.repeat)  # total rollout concurrency
                task_info = next(self.sampler_iter)
                task = BaseRepeatTask(task_info.task_item, repeat=self.repeat, filter=True)
                future = asyncio.create_task(task.run(ReleaseOnlySemaphore(engine_semaphore)))
                future._task_info = task_info
                submitted.put_nowait(future)
                self.submitted += 1
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                get_logger().critical(f"Error in produce: {e}\n{traceback.format_exc()}")
                return

    async def consume(self, submitted: asyncio.Queue, submit_semaphore: asyncio.Semaphore):
        async def get_finished(submitted, running_tasks: set, finish_reason):
            # submitted to running_tasks
            if submitted.empty() and len(running_tasks) == 0:
                item = await submitted.get()
                running_tasks.add(item)
                finish_reason["submitted"] += 1
            while submitted.empty() is False:
                running_tasks.add(submitted.get_nowait())
                finish_reason["submitted"] += 1

            done, _ = await asyncio.wait(running_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for item in done:
                running_tasks.remove(item)
                yield item

        def log_str(finish_reason):
            return f"[Rollout] running: {len(self.running_futures)},  finished: {', '.join([f'{k}: {v}' for k, v in finish_reason.items()])}"

        train_batch = []
        filtered_batch = []
        finish_reason = defaultdict(int)
        tqdm_bar = tqdm.tqdm(total=self.rollout_bs, desc="Collecting Rollouts")

        while len(train_batch) < self.rollout_bs:
            async for future in get_finished(submitted, self.running_futures, finish_reason):
                result = await future
                future._task_info.callback(result["result"])
                finish_reason[result["filter"]] += 1

                # Filter
                succeed = result["filter"] == "succeed"

                if self.filter == "none":
                    train_batch.append(result["result"])
                else:
                    if succeed:
                        train_batch.append(result["result"])
                    else:
                        filtered_batch.append(result["result"])
                        if self.filter == "discard":
                            train_batch.append(None)

                if self.async_rollout or (not succeed and self.filter == "resubmit"):
                    submit_semaphore.release()

                if succeed or self.filter == "none" or (self.async_rollout is False and self.filter == "discard"):
                    tqdm_bar.update(1)
                tqdm_bar.set_description(log_str(finish_reason))
                if len(train_batch) >= self.rollout_bs:
                    break
        train_batch = [x for x in train_batch if x is not None]
        train_batch = [y for x in train_batch for y in x]
        filtered_batch = [y for x in filtered_batch for y in x]
        get_logger().info(log_str(finish_reason))
        return train_batch, filtered_batch, finish_reason

    async def run(self, engine_semaphore: asyncio.Semaphore):
        self.submitted = 0
        if self.async_rollout:
            ultilize_ratio = max(0.01, (0.5 if self.utilize_ratio == -1.0 else self.utilize_ratio) - 0.2)
            max_submit = max(int(self.rollout_bs / ultilize_ratio), len(self.running_futures))
            get_logger().info(
                f"Async rollout enabled, setting max submit to {max_submit} based on utilize ratio {ultilize_ratio:.4f}, engine semaphore max concurrency {engine_semaphore._value}, current running futures {len(self.running_futures)}"
            )
            submit_semaphore = asyncio.Semaphore(max_submit)
        else:
            submit_semaphore = asyncio.Semaphore(self.rollout_bs)
        for _ in range(len(self.running_futures)):
            await submit_semaphore.acquire()
        submitted = asyncio.Queue()

        produce_handle = asyncio.create_task(self.produce(submitted, submit_semaphore, engine_semaphore))

        train_data, filter_batch, self.finish_reason = await self.consume(submitted, submit_semaphore)
        produce_handle.cancel()
        await produce_handle
        if self.async_rollout is False:
            assert len(self.running_futures) == 0
        else:
            while submitted.empty() is False:
                self.running_futures.add(submitted.get_nowait())
        if TaskCounter.count + engine_semaphore._value != engine_semaphore._initial_value:
            get_logger().warning(
                f"Semaphore mismatch, expected {engine_semaphore._initial_value}, got {engine_semaphore._value}, TaskCounter: {TaskCounter.count}, running futures: {len(self.running_futures)}"
            )
        if self.utilize_ratio == -1.0:
            self.utilize_ratio = len(train_data) / (len(train_data) + len(filter_batch))
        else:
            self.utilize_ratio = (
                self.utilize_ratio * 0.9 + (len(train_data) / (len(train_data) + len(filter_batch))) * 0.1
            )
        get_logger().info(f"Finish rollout, inference utilize ratio: {self.utilize_ratio:.4f}")
        return train_data, filter_batch

    def log_tensorboard(self, step):
        for k, v in self.finish_reason.items():
            self.tb_writer.add_scalar(f"rollout/{k}", v, global_step=step)
        self.tb_writer.add_scalar("rollout/submitted", self.submitted, global_step=step)

    def checkpoint(self):
        return self.task_sampler.checkpoint()

    def resume(self, state):
        self.task_sampler.resume(state)


class SingleEvalTask(Task):
    def __init__(self, dataset: list[Dataset], sample_ratio=1):
        super().__init__()
        self.dataset = dataset
        self.name = os.path.basename(self.dataset.config.name)
        self.sample_ratio = sample_ratio
        if sample_ratio > 1:
            assert int(sample_ratio) == sample_ratio
            self.index = [i for i in range(len(self.dataset)) for _ in range(int(sample_ratio))]
        else:
            self.index = random.sample(range(len(self.dataset)), math.ceil(len(self.dataset) * sample_ratio))

        logger.info(
            f"Eval Task on dataset {self.dataset.config.name}, sample ratio: {self.sample_ratio}, num samples: {len(self.index)}"
        )

    async def run(self, semaphore):
        async def eval_one_item(task_item: TaskItem, semaphore):
            with TaskCounter():
                try:
                    async with semaphore:
                        task_data = await task_item.task_cls.infer(task_item.task_data)
                        task_data = await task_item.task_cls.eval(task_data)
                    return task_data
                except RuntimeError as e:
                    logger.critical(f"Runtime Error: {e}")
                    return None
                except Exception as e:  # noqa: BLE001
                    logger.critical(
                        f"Eval task encountered an error on dataset {self.dataset.config.name}: {e}\n{traceback.format_exc()}"
                    )
                    return None

        futures = [
            asyncio.get_running_loop().create_task(eval_one_item(self.dataset[i], semaphore=semaphore))
            for i in self.index
        ]

        results: list[TaskData] = []
        for future in tqdm.tqdm(asyncio.as_completed(futures), total=len(futures), desc=f"Evaluating {self.name}"):
            results.append(await future)
        results = [x for x in results if x is not None]
        if len(results) != len(futures):
            logger.warning(
                f"Some eval tasks failed on dataset {self.dataset.config.name}, {len(results)}/{len(futures)} succeeded."
            )
        for item in results:
            item.others["origin_dataset"] = self.dataset.config.name
        return results


class EvalTask(Task):
    def __init__(self, datasets, sample_ratios):
        super().__init__()
        self.tasks = [
            SingleEvalTask(dataset=dataset, sample_ratio=ratio) for dataset, ratio in zip(datasets, sample_ratios)
        ]

    async def run(self, semaphore):
        futures = [asyncio.create_task(task.run(semaphore)) for task in self.tasks]
        results = await asyncio.gather(*futures)
        final_results = {}
        for name, res in zip([t.dataset.config.name for t in self.tasks], results):
            final_results[name] = res
        return final_results


# algorithm


class RLAlgorithm:
    def __init__(self, args: RLAlgorithmConfig, tb_writer=None):
        if tb_writer is None:
            self.tb_writer = SummaryWriter(log_dir=args.work_dir)
        else:
            self.tb_writer = tb_writer

        self.args = args
        self.total_steps = self.args.total_training_steps
        self.global_step = 0
        self.semaphore = asyncio.Semaphore(self.args.max_concurrency)
        self.semaphore._initial_value = self.args.max_concurrency

    async def lazy_init(self):
        # build train task
        self.datasets = [await d_config.build() for d_config in self.args.datasets]
        self.train_queue = asyncio.Queue(maxsize=self.args.roll_out_bs)
        assert self.args.async_rollout in ["none", "task", "group"], "async_rollout must be 'none' or 'task'"
        self.train_task = TrainTask(
            datasets=self.datasets,
            sample_ratios=self.args.train_sample_ratios,
            tb_writer=self.tb_writer,
            rollout_bs=self.args.roll_out_bs,
            repeat=self.args.num_rl_group,
            async_rollout=self.args.async_rollout != "none",
            filter=self.args.filter_group,
        )
        self.train_task_handle = None
        self.eval_handle = None
        # build eval task

        if len(self.args.eval_datasets) > 0:
            self.eval_dataset = [await d_config.build() for d_config in self.args.eval_datasets]
            self.eval_task = EvalTask(
                datasets=self.eval_dataset,
                sample_ratios=self.args.eval_sample_ratio,
            )
        else:
            self.eval_dataset = []
            self.eval_task = None

    def prepare_fake_data(self):
        train_batch = []
        seq_len = 1024
        prompt_len = 200
        for _ in range(self.args.roll_out_bs * self.args.num_rl_group):
            input_ids = [random.randint(0, 1000) for _ in range(seq_len)]
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            log_probs = [random.uniform(-5.0, -0.1) for _ in range(seq_len)]
            advantage = random.uniform(-1.0, 1.0)
            item = TaskData(
                id=f"fake_{random.randint(0, 100000)}",
                messages=[{"role": "user", "content": "fake"}, {"role": "assistant", "content": "fake"}],
                metric=random.choice([0, 1]),
                total_tokens=seq_len,
                finish_reason=random.choice(["stop", "length"]),
                others={
                    "rl_data": {
                        "input_ids": input_ids,
                        "labels": labels,
                        "log_probs": log_probs,
                        "advantages": advantage,
                    },
                },
            )
            train_batch.append(item)
        return train_batch

    async def step(self, step, url):
        if os.environ.get("USE_FAKE_DATA", "0") == "1":
            logger.warning("Using fake data for testing.")
            train_batch = self.prepare_fake_data()
            eval_reward = -100000
        else:
            # set model url
            self.global_step = step
            for dataset in self.datasets + self.eval_dataset:
                dataset.config.infer_args.model_name = self.args.model_name
                dataset.config.infer_args.model_url = url

            async def finish_eval():
                eval_result = await self.eval_handle
                self.eval_handle = None
                if eval_result is not None:
                    _, eval_reward = self.log_data(
                        functools.reduce(operator.iadd, list(eval_result.values()), []), step, "evalset"
                    )
                    self.dump_result(functools.reduce(operator.iadd, list(eval_result.values()), []), step, "evalset")
                else:
                    eval_reward = -100000
                return eval_reward

            # run task
            if self.eval_task is not None and (
                (step == 0 and os.environ.get("DISABLE_INIT_EVAL", "0") != "1")
                or (step + 1) % self.args.eval_interval == 0
                or step == self.total_steps - 1
            ):
                self.eval_handle = asyncio.create_task(self.eval_task.run(self.semaphore))
                await asyncio.sleep(10)  # allow eval to start first
            else:
                self.eval_handle = asyncio.create_task(asyncio.sleep(0))
            train_batch, filtered_batch = await self.train_task.run(self.semaphore)
            eval_reward = await finish_eval()

            # log
            self.train_task.log_tensorboard(step)
            self.log_data(train_batch + filtered_batch, step, "trainset")

            # post process
            for x in train_batch:
                try:
                    retrieved = AsyncClient.retrieve_collected_tokens(x.messages)
                except ValueError:
                    logger.critical(f"Failed to retrieve tokens for messages: {x.messages}")
                    retrieved = {}
                x.others["rl_data"].update(retrieved)
            train_batch = [x for x in train_batch if "input_ids" in x.others["rl_data"]]

            if step == 0 or (step + 1) % self.args.eval_interval == 0 or step == self.total_steps - 1:
                self.dump_result(train_batch + filtered_batch, step, "trainset")
        if self.args.num_rl_group == 1:  # use batch advantage normalization when num_rl_group == 1.
            advtanges = torch.tensor([x.others["rl_data"]["advantages"] for x in train_batch])
            advtanges = (advtanges - advtanges.mean()) / (advtanges.std() + 1e-8)
            for i in range(len(train_batch)):
                train_batch[i].others["rl_data"]["advantages"] = advtanges[i].item()
        if self.args.data_post_process_func is not None:
            train_batch = self.args.data_post_process_func(train_batch)
        return [to_rl_data(task_data) for task_data in train_batch], eval_reward

    async def resume(self, folder):
        path = os.path.join(folder, "algo.json")
        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}, skipping resume.")
            return
        async with aiofiles.open(path, "r") as f:
            data = json.loads(await f.read())

        self.train_task.resume(data["train_task"])
        self.global_step = data["global_step"]
        logger.info(f"Resumed AsyncAlgorithm from {path}, global step: {self.global_step}")
        return self.global_step

    async def checkpoint(self, folder):
        state = self.train_task.checkpoint()
        data = {
            "train_task": state,
            "global_step": self.global_step,
        }
        async with aiofiles.open(os.path.join(folder, "algo.json"), "w") as f:
            await f.write(json.dumps(data))

    # internal

    def log_data(self, data: list[TaskData], step, key):
        def _log_one(data: list[TaskData], step, key):
            data = [x for x in data if x is not None]
            rewards = [x.metric for x in data]
            accuracy = np.mean([1 if r == 1 else 0 for r in rewards])
            length = np.mean([x.total_tokens for x in data])
            finish_reasons = defaultdict(int)
            for x in data:
                finish_reasons[x.finish_reason] += 1
            logger.info(
                f"[{key}][Step {step}] Num samples: {len(data)}, Avg reward: {np.mean(rewards):.4f}, Accuracy: {accuracy:.4f}, Avg length: {length:.2f}, finish_reason: {dict(finish_reasons)}"
            )
            self.tb_writer.add_scalar(f"{key}/avg_reward", np.mean(rewards), global_step=step)
            self.tb_writer.add_scalar(f"{key}/accuracy", accuracy, global_step=step)
            self.tb_writer.add_scalar(f"{key}/avg_length", length, global_step=step)
            for finish_reason in finish_reasons:
                self.tb_writer.add_scalar(
                    f"{key}/finish_reason_{finish_reason}", finish_reasons[finish_reason], global_step=step
                )
            return {
                "avg_reward": np.mean(rewards),
                "accuracy": accuracy,
                "avg_length": length,
                "num": len(data),
                "finish_reasons": dict(finish_reasons),
            }

        data = [x for x in data if x is not None]

        date_by_dataset = defaultdict(list)
        for item in data:
            dataset_name = item.others.get("origin_dataset", "unknown")
            date_by_dataset[dataset_name].append(item)
        result = {}
        for dataset_name, dataset_data in date_by_dataset.items():
            result[dataset_name] = _log_one(dataset_data, step, f"{key}_{dataset_name}".replace("/", "_"))

        acc_avg = np.mean([r["accuracy"] for r in result.values()])
        reward_avg = np.mean([r["avg_reward"] for r in result.values()])
        self.tb_writer.add_scalar(key + "_all/reward", reward_avg, global_step=step)
        self.tb_writer.add_scalar(key + "_all/accuracy", acc_avg, global_step=step)
        logger.info(f"[{key}][Step {step}] Avg accuracy across eval sets: {acc_avg:.4f}; Avg reward: {reward_avg:.4f}")

        return result, reward_avg

    def dump_result(self, data: list[TaskData], step, key):
        def _dump_one(data: list[TaskData], step, key):
            file_path = self.args.work_dir + "/trajectories/" + f"{step:06d}_{key}.jsonl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            jsonl_data = [filter_json_serializable(x.model_dump()) if isinstance(x, TaskData) else x for x in data]

            with open(file_path, "w") as f:
                for item in jsonl_data:
                    f.write(json.dumps(item) + "\n")

        data = [x for x in data if x is not None]
        date_by_dataset = defaultdict(list)
        for item in data:
            dataset_name = item.others.get("origin_dataset", "unknown")
            date_by_dataset[dataset_name].append(item)
        for dataset_name, dataset_data in date_by_dataset.items():
            _dump_one(dataset_data, step, f"{key}_{dataset_name}".replace("/", "_"))


# rl trainer


class RLTrainer:
    def __init__(self, config: RLAlgorithmConfig):
        self.config = config
        self.algorithm = RLAlgorithm(config)
        self.model_engine = SpmdTrainInferEngine(model_config=config.llm_config, engine_config=config.engine_config)

        self.cur_step = 0

        init_logger(self.config.work_dir + "/trainer.log")
        DummySummaryWriter.init_writer(self.config.work_dir)

        self.logger = get_logger()
        self.tb_writer = get_tb_writer()

        self.logger.info(str(self.algorithm.args.model_dump()))
        self.logger.info(str(self.model_engine.config.model_dump()))
        self.logger.info(str(self.config.model_dump()))

        self.best_reward = -1

    async def lazy_init(self):
        await asyncio.gather(self.model_engine.lazy_init(), self.algorithm.lazy_init())
        if self.config.auto_resume:
            await self.resume()

    async def fit(self):
        for step in range(self.cur_step, self.config.total_training_steps):
            with MeasureTime("step"):
                self.cur_step = step
                self.logger.info(f"*Starting training step {step}")

                self.logger.info("**Starting model serving...")
                with MeasureTime("start_server"):
                    await self.model_engine.serve()
                    t0 = time.time()

                with MeasureTime("algo"):
                    train_data, eval_reward = await self.algorithm.step(
                        step, url=await self.model_engine.get_server_ip()
                    )

                with MeasureTime("stop_server"):
                    self.logger.info("**Algorithm ends, stopping model serving...")
                    generate_tokens = await self.model_engine.stop_serve()
                    get_logger().info(
                        f"**Model serving ends, generated {generate_tokens // 1024} k tokens,throughput: {int(generate_tokens / self.model_engine.config.num_workers / (time.time() - t0))} tokens/s"
                    )

                if len(train_data) != 0:
                    with MeasureTime("train"):
                        self.logger.info("**Starting model training...")
                        train_log = await self.model_engine.train_wrapper(train_data, step)
                        log_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_log.items()])
                        self.logger.info(f"**Training step {step} logs: {log_str}")
                        for k, v in train_log.items():
                            self.tb_writer.add_scalar(f"{k}", v, step)
                    self.logger.info("**Model training ends, synchronizing weights...")
                else:
                    self.logger.warning("No training data received, skipping this step.")
                gc.collect()
                torch.cuda.empty_cache()
                # checkpoint
                if (step + 1) % self.config.checkpoint_interval == 0:
                    with MeasureTime("ckpt"):
                        await self.checkpoint(step)
                    if eval_reward > self.best_reward:
                        ckpt_folder = self.config.work_dir + "/checkpoints/" + f"/{step:06d}/"
                        shutil.copytree(
                            ckpt_folder, os.path.join(self.config.work_dir, "best_ckpt"), dirs_exist_ok=True
                        )
                        self.logger.info(
                            f"New best reward {eval_reward:.4f} at step {step}, checkpoint saved to best_ckpt/"
                        )
                        self.best_reward = eval_reward
            MeasureTime.saved_time["others"] = 2 * MeasureTime.saved_time["step"] - sum(
                list(MeasureTime.saved_time.values())
            )
            for key in MeasureTime.saved_time:
                self.logger.info(f"Time for {key}: {int(MeasureTime.saved_time[key])} seconds")
                self.tb_writer.add_scalar(f"Time/{key}", int(MeasureTime.saved_time[key]), step)
            MeasureTime.clear()
            self.logger.info("----------------------------\n\n")

    async def checkpoint(self, step):
        ckpt_folder = self.config.work_dir + "/checkpoints/"
        step_folder = ckpt_folder + f"/{step:06d}/"
        tmp_folder = os.path.join(self.config.work_dir, "tmp_ckpt")

        os.makedirs(ckpt_folder, exist_ok=True)
        os.makedirs(tmp_folder, exist_ok=True)

        await self.model_engine.checkpoint(tmp_folder)
        await self.algorithm.checkpoint(tmp_folder)
        shutil.move(tmp_folder, step_folder)
        self.logger.info(f"Checkpoint saved at step {step} to {ckpt_folder}")

        existed = sorted(os.listdir(ckpt_folder))
        to_remove = existed[: -self.config.max_checkpoints]
        for rm in to_remove:
            step_folder = os.path.join(ckpt_folder, rm)
            shutil.rmtree(step_folder)
            self.logger.info(f"Removed old checkpoint: {step_folder}")
        async with aiofiles.open(os.path.join(self.config.work_dir, "trainer.json"), "w") as f:
            await f.write(json.dumps({"best_reward": self.best_reward}))

    async def resume(self):
        def get_ckpt_path_from_work_dir(work_dir):
            ckpt_folder = os.path.join(work_dir, "checkpoints")
            if not os.path.exists(ckpt_folder):
                return None
            existed = sorted(os.listdir(ckpt_folder))
            if len(existed) == 0:
                return None
            return os.path.join(ckpt_folder, existed[-1])

        step_folder = os.environ.get("RESUME_PATH", get_ckpt_path_from_work_dir(self.config.work_dir))
        if step_folder is None:
            self.logger.warning("No checkpoints found, starting from scratch.")
            return False
        else:
            await self.model_engine.resume(step_folder)
            self.cur_step = await self.algorithm.resume(step_folder)
            self.logger.info(f"Resumed from checkpoint: {step_folder}")

            trainer_json = os.path.join(self.config.work_dir, "trainer.json")
            if os.path.exists(trainer_json):
                async with aiofiles.open(trainer_json, mode="r") as f:
                    data = json.loads(await f.read())
                    self.best_reward = data.get("best_reward", -1)

            return True
