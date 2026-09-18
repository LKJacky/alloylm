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


class ProducerConsumer:
    def __init__(self):
        self.logger = get_logger(__name__)

    async def produce(
        self,
        semaphore,
        running_futures: asyncio.Queue,
        produce_iter: iter,
    ):
        try:
            while True:
                await semaphore.acquire()
                try:
                    future = next(produce_iter)
                except StopIteration:
                    semaphore.release()
                    break
                except BaseException as e:
                    semaphore.release()
                    self.logger.critical(f"Error in produce: {e}", exc_info=True)
                    return
                await running_futures.put(future)
        except asyncio.CancelledError:
            pass
        except BaseException as e:
            self.logger.critical(f"Error in produce: {e}", exc_info=True)

    async def consume(self, semaphore, running_futures: asyncio.Queue, consume_func: callable):
        produce_task_had_exited = False
        try:
            collected_futures = set()
            while True:
                if produce_task_had_exited:
                    while running_futures.qsize() > 0:
                        collected_futures.add(await running_futures.get())
                    if len(collected_futures) == 0:
                        return
                else:
                    if running_futures.qsize() > 0 or len(collected_futures) == 0:
                        future = await running_futures.get()
                        if future is None:
                            produce_task_had_exited = True
                        else:
                            collected_futures.add(future)
                        continue
                    else:  # No new futures to collect, wait for some to complete
                        pass

                # wait finished items
                done, _ = await asyncio.wait(collected_futures, return_when=asyncio.FIRST_COMPLETED, timeout=1)
                for future in done:
                    collected_futures.remove(future)
                    semaphore.release()
                    try:
                        await consume_func(future)
                    except BaseException as e:
                        self.logger.critical(f"Error in consume_func: {e}", exc_info=True)
        except asyncio.CancelledError:
            while running_futures.qsize() > 0:
                future = await running_futures.get()
                if future is not None:
                    collected_futures.add(future)
            for collected_future in collected_futures:
                collected_future.cancel()
                try:
                    await collected_future
                except asyncio.CancelledError:
                    pass
        except BaseException as e:
            self.logger.critical(f"Error in consume: {e}", exc_info=True)

    async def run(self, semaphore, produce_iter, consume_func):
        try:
            running_futures = asyncio.Queue()
            # produce and consume tasks
            produce_task = asyncio.create_task(self.produce(semaphore, running_futures, produce_iter))
            consume_task = asyncio.create_task(self.consume(semaphore, running_futures, consume_func))
            done, _ = await asyncio.wait([produce_task, consume_task], return_when=asyncio.FIRST_COMPLETED)
            if consume_task in done:
                produce_task.cancel()
                try:
                    await produce_task
                except asyncio.CancelledError:
                    pass
            elif produce_task in done:
                await running_futures.put(None)
                await consume_task
        except BaseException as e:
            self.logger.critical(f"Error in run: {e}", exc_info=True)
            raise


class EvalRunner:
    def __init__(self, datasets: list[Dataset], work_dir=None, resume=False):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.datasets = datasets
        self.work_dir = work_dir
        self.resume_enabled = resume
        self.logger = get_logger(path=self.work_dir + "/eval.log", force_recreate=True)

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
                data = load_jsonl(resume_path)
                resumed[dataset.config.name] = {d["id"]: TaskData(**d) for d in data}
                self.logger.info(f"Resuming from {resume_path} with {len(resumed[dataset.config.name])} items")
        return resumed

    async def run(self, semaphore):
        t0 = time.time()

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

        def produce_func():
            async def directly_return(x):
                return x

            resumed = self._load_resumed_data()
            for dataset in self.datasets:
                ds_resumed = resumed.get(dataset.config.name, {})
                for item in dataset:
                    if item.task_data.id in ds_resumed:
                        task_data = ds_resumed[item.task_data.id]
                        task_data.infer_args = item.task_data.infer_args
                        future = asyncio.create_task(directly_return(task_data))
                        future._resumed = True
                    else:
                        future = asyncio.create_task(self._run_eval_item(item))
                        future._resumed = False
                    future._dataset_name = dataset.config.name
                    yield future

        async def consume_func(future):
            ds_name = future._dataset_name
            result = future.result()
            ds_results[ds_name].append(result)
            if result and not future._resumed:
                await save_queues[ds_name].put(result)
            tqdm_bars[ds_name].update(1)

        await ProducerConsumer().run(semaphore, produce_func(), consume_func)

        # summary
        try:
            results = []
            for dataset in self.datasets:
                name = dataset.config.name
                origin_num = len(dataset)
                valid = [r for r in ds_results[name] if r is not None]
                if len(valid) < origin_num:
                    self.logger.info(f"Warning: {origin_num - len(valid)} samples failed during evaluation of {name}.")
                summary = dataset.summary(valid)
                results.append(summary)
                self.logger.info(f"Dataset {name} eval done, main metric: {summary.metric:.4f}")
        except Exception as e:  # noqa: BLE001
            self.logger.info(f"Error in EvalRunner: {e}\n {traceback.format_exc()}")
            results = []
        finally:
            for task in dump_tasks:
                task.cancel()
            await asyncio.gather(*dump_tasks, return_exceptions=True)

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
            self.logger.info(df.to_markdown(index=False))
        self.logger.info(f"Evaluation use {int(time.time() - t0)} seconds\nEvaluation Results:")

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
        get_logger().info(f"build {dataset.config.name}, total {len(dataset)} samples")

    if config.one_by_one:
        for dataset in datasets:
            runner = EvalRunner([dataset], work_dir=config.work_dir, resume=config.resume)
            await runner.run(semaphore)
    else:
        runner = EvalRunner(datasets, work_dir=config.work_dir, resume=config.resume)
        await runner.run(semaphore)
