import asyncio
import os
import random

import orjson
import ray
import tqdm
from pydantic import BaseModel
from torch.utils.data import Dataset

from alloylm.engine.train_engine.dataset import (
    distangle_train_or_not_train,
    tokenize_messages,
)
from alloylm.utils import get_logger, init_ray

logger = get_logger()


class SFTPackDataset(Dataset):
    def __init__(
        self, file_paths, sample_ratios, tokenizer, chat_template, max_length, random_seed=42, num_tokenize_workers=-1
    ):
        if num_tokenize_workers < 1 and num_tokenize_workers != -1:
            raise ValueError("num_tokenize_workers must be -1 or a positive integer")

        self.file_paths, self.sample_ratios = self.get_files_from_folder(file_paths, sample_ratios)

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length

        self.data = []  # file_idx, offset, num_tokens
        self.packed_data = []
        self.num_skip_data = 0
        self.rng = random.Random(random_seed)
        self.logger = get_logger()
        self.num_tokenize_workers = num_tokenize_workers

    async def lazy_init(self):
        init_ray()
        sample_infos = []
        for file_idx, file_path in enumerate(self.file_paths):
            sample_infos.extend((file_idx, offset) for offset in self.get_offsets(file_path))

        if not sample_infos:
            self.packed_data = []
            self.num_skip_data = 0
            return
        if self.num_tokenize_workers == -1:
            num_cpu = int(ray.cluster_resources()["CPU"])
            num_workers = max(1, num_cpu // 2, num_cpu - 16)
        else:
            num_workers = self.num_tokenize_workers
        num_workers = min(num_workers, max(1, len(sample_infos) // 1000))
        num_samples_per_worker = (len(sample_infos) + num_workers - 1) // num_workers

        remote_count_tokens = ray.remote(self.count_tokens)
        file_paths_ref = ray.put(self.file_paths)
        tokenizer_ref = ray.put(self.tokenizer)
        chat_template_ref = ray.put(self.chat_template)
        futures = []
        try:
            for start_idx in range(0, len(sample_infos), num_samples_per_worker):
                worker_sample_infos = sample_infos[start_idx : start_idx + num_samples_per_worker]
                futures.append(
                    remote_count_tokens.remote(
                        worker_sample_infos,
                        file_paths_ref,
                        tokenizer_ref,
                        chat_template_ref,
                    )
                )
            self.logger.info(
                f"Starting token counting for {len(sample_infos)} samples with {num_workers} workers",
            )

            counted_samples = [[] for _ in self.file_paths]
            for worker_results in await asyncio.gather(*futures):
                for sample in worker_results:
                    counted_samples[sample[0]].append(sample)
        finally:
            for future in futures:
                ray.cancel(future, force=True)

        for file_idx, samples in enumerate(counted_samples):
            ratio = self.sample_ratios[file_idx]
            full = int(ratio)
            selected = list(range(len(samples))) * full
            selected.extend(self.rng.sample(range(len(samples)), k=int(len(samples) * (ratio - full))))
            self.data.extend(samples[index] for index in selected)

        self.packed_data, self.num_skip_data = self.pack_data(self.data, self.max_length)

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        return self.packed_data[idx]

    @classmethod
    def get_files_from_folder(cls, file_paths, sample_ratios):
        new_file_paths = []
        new_sample_ratios = []
        for file_path, sample_ratio in zip(file_paths, sample_ratios):
            if os.path.isdir(file_path):
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if file.endswith(".jsonl"):
                            full_path = os.path.join(root, file)
                            new_file_paths.append(full_path)
                            new_sample_ratios.append(sample_ratio)
            else:
                new_file_paths.append(file_path)
                new_sample_ratios.append(sample_ratio)
        file_path_with_ratios = list(zip(new_file_paths, new_sample_ratios))
        file_path_with_ratios.sort(key=lambda x: x[0])
        new_file_paths, new_sample_ratios = zip(*file_path_with_ratios)
        return new_file_paths, new_sample_ratios

    @classmethod
    def pack_data(cls, data, max_length):
        num_skip = 0
        packed_data = []
        current_pack = []
        current_length = 0

        for i, (_, _, num_tokens) in enumerate(data):
            if num_tokens > max_length:
                # Skip this sample if it exceeds max_length
                num_skip += 1
                continue
            if current_length + num_tokens > max_length:
                if current_pack:
                    packed_data.append(current_pack)
                current_pack = [i]
                current_length = num_tokens
            else:
                current_pack.append(i)
                current_length += num_tokens

        if current_pack:
            packed_data.append(current_pack)

        return packed_data, num_skip

    @classmethod
    def get_offsets(cls, file_path):
        offsets = []
        with open(file_path, "rb") as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                offsets.append(offset)
        return offsets

    @staticmethod
    def count_tokens(
        samples,
        file_paths,
        tokenizer,
        chat_template,
    ):

        if samples[0][0] == 0 and samples[0][1] == 0:
            bar = tqdm.tqdm(total=len(samples), mininterval=10)
        else:
            bar = None
        results = []
        previous_file_index = -1
        file_handle = None
        try:
            for file_index, offset in samples:
                if previous_file_index != file_index:
                    if file_handle:
                        file_handle.close()
                    file_handle = open(file_paths[file_index], "rb")  # noqa: SIM115
                    previous_file_index = file_index
                file_handle.seek(offset)
                line = file_handle.readline()
                messages = orjson.loads(line)["messages"]
                num_tokens = len(tokenize_messages(messages, chat_template, tokenizer)[0])
                results.append((file_index, offset, num_tokens))
                if bar:
                    bar.update(1)
        finally:
            if file_handle:
                file_handle.close()
        return results

    def get_formated_message_sample(self):
        file_index, offset, _ = self.data[0]
        with open(self.file_paths[file_index], "rb") as f:
            f.seek(offset)
            line = f.readline()
        messages = orjson.loads(line)["messages"]
        content = ""
        for item, compute_loss in distangle_train_or_not_train(messages, self.chat_template):
            if compute_loss:
                content += f"<loss>{item}</loss>"
            else:
                content += item
        return content


class SFTPackDatasetConfig(BaseModel):
    file_paths: list[str]
    sample_ratios: list[float]
    max_length: int
    num_tokenize_workers: int = -1

    async def build(self, tokenizer, chat_template) -> SFTPackDataset:
        dataset = SFTPackDataset(
            self.file_paths,
            self.sample_ratios,
            tokenizer,
            chat_template,
            self.max_length,
            num_tokenize_workers=self.num_tokenize_workers,
        )
        await dataset.lazy_init()
        return dataset
