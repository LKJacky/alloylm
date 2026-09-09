import asyncio
import json
import os
import shutil
import time
import traceback

import aiofiles
import pandas as pd
import tqdm
from pydantic import BaseModel

from alloylm.algorithm.base import Dataset, DatasetConfig, TaskData, TaskItem
from alloylm.utils import get_logger

DEFAULT_MODEL_NAME = "ALLOYLM"


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


async def save_to_file(queue: asyncio.Queue, file_path: str, create_new=False):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if create_new and os.path.exists(file_path):
        shutil.move(file_path, file_path + ".bak")
    cached = []
    last_save_time = time.time()
    async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
        try:
            while True:
                item = await queue.get()
                cached.append(item)
                if time.time() - last_save_time > 60:
                    await f.writelines(json.dumps(d.model_dump(), ensure_ascii=False) + "\n" for d in cached)
                    cached.clear()
                    last_save_time = time.time()
        finally:
            while not queue.empty():
                item = queue.get_nowait()
                cached.append(item)
            if len(cached) > 0:
                await f.writelines(json.dumps(d.model_dump(), ensure_ascii=False) + "\n" for d in cached)
            if os.path.exists(file_path + ".bak"):
                os.remove(file_path + ".bak")


class EvalRunner:
    def __init__(self, datasets: list[Dataset], work_dir=None, resume=False):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.datasets = datasets
        self.work_dir = work_dir
        self.resume_enabled = resume
        self.logger = get_logger(path=self.work_dir + "/eval.log")

    async def _run_eval_item(self, task_item: TaskItem):
        try:
            if task_item.task_data.finish_reason == "":  # not forward
                task_data = await task_item.task_cls.infer(task_item.task_data)
                task_data = await task_item.task_cls.eval(task_data)
            elif task_item.task_data.metric == -1.0:
                task_data = await task_item.task_cls.eval(task_item.task_data)
            else:
                task_data = task_item.task_data
            return task_data
        except Exception:
            self.logger.critical(f"Error in run_eval for task {task_item.task_data.id}", exc_info=True)
            return None

    def _load_resumed_data(self):
        resumed = {}
        if not self.resume_enabled or not self.work_dir:
            return resumed
        for dataset in self.datasets:
            resume_path = os.path.join(self.work_dir, f"{dataset.config.name}.jsonl")
            if os.path.exists(resume_path):
                print(f"Resuming from {resume_path}")
                data = load_jsonl(resume_path)
                resumed[dataset.config.name] = {d["id"]: TaskData(**d) for d in data}
        return resumed

    async def produce(
        self,
        semaphore,
        running_futures,
        future_to_dataset,
        resumed,
        ds_results,
    ):
        try:
            for dataset in self.datasets:
                ds_resumed = resumed.get(dataset.config.name, {})
                for item in dataset:
                    await semaphore.acquire()
                    try:
                        if item.task_data.id in ds_resumed:
                            task_data = ds_resumed[item.task_data.id]
                            task_data.infer_args = item.task_data.infer_args
                            ds_results[dataset.config.name].append(task_data)
                            semaphore.release()
                            continue
                        else:
                            future = asyncio.create_task(self._run_eval_item(item))
                            running_futures.add(future)
                            future_to_dataset[future] = dataset.config.name
                    except BaseException:
                        semaphore.release()
                        raise
        except BaseException as e:
            self.logger.critical(f"Error in produce: {e}", exc_info=True)

    async def run(self, semaphore):
        t0 = time.time()
        resumed = self._load_resumed_data()

        running_futures = set()
        future_to_dataset = {}

        # Per-dataset save queues and progress bars
        save_queues = {}
        dump_tasks = []
        tqdm_bars = {}
        ds_results = {}
        for dataset in self.datasets:
            name = dataset.config.name
            ds_results[name] = []
            tqdm_bars[name] = tqdm.tqdm(total=len(dataset), desc=name)
            save_queues[name] = asyncio.Queue()
            save_path = os.path.join(self.work_dir, f"{name}.jsonl")
            dump_tasks.append(
                asyncio.create_task(save_to_file(save_queues[name], save_path, create_new=not self.resume_enabled))
            )

        produce_task = asyncio.create_task(
            self.produce(semaphore, running_futures, future_to_dataset, resumed, ds_results)
        )
        results = []
        try:
            while not produce_task.done() or running_futures:
                pending = set(running_futures)
                if not produce_task.done():
                    pending.add(produce_task)
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if produce_task in done:
                    produce_task.result()
                    done.remove(produce_task)

                for future in done:
                    running_futures.remove(future)
                    semaphore.release()
                    ds_name = future_to_dataset.pop(future)
                    result = future.result()
                    ds_results[ds_name].append(result)
                    if result:
                        await save_queues[ds_name].put(result)
                    tqdm_bars[ds_name].update(1)

            for dataset in self.datasets:
                name = dataset.config.name
                origin_num = len(dataset)
                valid = [r for r in ds_results[name] if r is not None]
                if len(valid) < origin_num:
                    print(f"Warning: {origin_num - len(valid)} samples failed during evaluation of {name}.")
                summary = dataset.summary(valid)
                results.append(summary)
                print(f"Dataset {name} eval done, main metric: {summary.metric:.4f}")
        except Exception as e:  # noqa: BLE001
            print("Error in EvalRunner:", e, traceback.format_exc())
            results = []
        finally:
            produce_task.cancel()
            for future in running_futures:
                future.cancel()
            for t in dump_tasks:
                t.cancel()
            await asyncio.gather(produce_task, *running_futures, *dump_tasks, return_exceptions=True)
            for _ in running_futures:
                semaphore.release()

        if self.work_dir:
            result_dict = {r.dataset_name: r for r in results}
            summary_path = os.path.join(self.work_dir, "result.json")

            if os.path.exists(summary_path):
                async with aiofiles.open(summary_path, encoding="utf-8") as f:
                    old_results = json.loads(await f.read())
            else:
                old_results = {}
            old_results.update({k: v.model_dump(exclude="task_data") for k, v in result_dict.items()})
            async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(old_results, ensure_ascii=False, indent=4))

        if len(results) > 0:
            df = pd.DataFrame(columns=list(results[0].model_dump(exclude="task_data").keys()))
            for result in results:
                df = pd.concat([df, pd.DataFrame([result.model_dump(exclude="task_data")])], ignore_index=True)
            print(df.to_markdown(index=False))
        print(f"Evaluation use {int(time.time() - t0)} seconds\nEvaluation Results:")

        # For single-dataset backward compatibility, return the single summary directly
        if len(self.datasets) == 1 and len(results) == 1:
            return results[0]
        return results


class EvalConfig(BaseModel):
    datasets: list[DatasetConfig]
    concurrency: int = 2048
    url: str = "http://localhost:8000/v1"
    model_name: str = DEFAULT_MODEL_NAME
    work_dir: str = "./work_dirs/debug/"
    one_by_one: bool = False
    mode: str = "all"
    resume: bool = False


async def run_eval(config: EvalConfig):
    if config.concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    semaphore = asyncio.Semaphore(config.concurrency)
    datasets = await asyncio.gather(*[asyncio.create_task(x.build()) for x in config.datasets])

    for dataset in datasets:
        if config.url is not None:
            dataset.config.infer_args.model_url = config.url
        if config.model_name is not None:
            dataset.config.infer_args.model_name = config.model_name
        print(f"build {dataset.config.name}, total {len(dataset)} samples")

    if config.one_by_one:
        for dataset in datasets:
            runner = EvalRunner([dataset], work_dir=config.work_dir, resume=config.resume)
            await runner.run(semaphore)
    else:
        runner = EvalRunner(datasets, work_dir=config.work_dir, resume=config.resume)
        await runner.run(semaphore)
