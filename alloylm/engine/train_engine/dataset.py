import json
import os
import random

import numpy as np
import torch
from datasets import concatenate_datasets
from torch import distributed as dist
from torch.utils.data import ConcatDataset, Dataset


# task datasets
class SoftPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, target=2048, blend=False, sort=False):
        self.seed = 0
        if blend:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.target = target

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _infos = self.get_pack_infos(dataset, i, num_tokens[i])
            pack_infos.append(_infos)
        self.pack_infos = concatenate_datasets(pack_infos)

    @property
    def longest(self):
        return self.pack_infos["longest"]

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        # _ori_lens = dataset['num_tokens']
        inds = [i for i in range(len(dataset))]
        rng = random.Random(self.seed)
        rng.shuffle(inds)
        from datasets import Dataset  # use transformers Dataset

        # below is as same as the original SoftPackDataset

        item_buffer = []
        length_buffer = []
        longest = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= self.target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                longest = max(longest, num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                        "longest": int(longest),
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                longest = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                "dataset_id": dataset_id,
                "indices": item_buffer,
                "longest": int(longest),
            }

            pack_infos.append(info)

        pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        indices = self.pack_infos[item]["indices"]
        dataset_id = self.pack_infos[item]["dataset_id"]
        return [self.datasets[dataset_id][i] for i in indices]


@torch.inference_mode()
def task_collate_fn(batch):
    def collated_one_sample(single_batch):
        keys = single_batch[0].keys()
        input_ids = [x for item in single_batch for x in item["input_ids"]]
        labels = [x for item in single_batch for x in item["labels"]]
        num_tokens = [item["num_tokens"] for item in single_batch]

        advantages = (
            [x for item in single_batch for x in [item["advantages"]] * item["num_tokens"]]
            if "advantages" in keys
            else None
        )
        old_log_probs = [x for item in single_batch for x in item["log_probs"]] if "log_probs" in keys else None
        entropy = [x for item in single_batch for x in item["train_entropy"]] if "train_entropy" in keys else None

        return {
            "input_ids": torch.tensor(input_ids).unsqueeze(0),
            "labels": torch.tensor(labels).unsqueeze(0),
            "num_tokens": torch.tensor(num_tokens),
            "advantages": torch.tensor(advantages).unsqueeze(0) if advantages is not None else None,
            "old_log_probs": torch.tensor(old_log_probs).unsqueeze(0) if old_log_probs is not None else None,
            "entropy": torch.tensor(entropy).unsqueeze(0) if entropy is not None else None,
            "ids": [item["id"] for item in single_batch],
        }

    batch = [x for y in batch for x in y]
    return collated_one_sample(batch)


class TaskDataset(Dataset):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

    @property
    def num_tokens(self):
        return [item["num_tokens"] for item in self.tasks]

    def dump_log(self, work_dir):
        dump_dir = os.path.join(work_dir, "trajectories", "logs")
        os.makedirs(dump_dir, exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join(dump_dir, f"rank_{dist.get_rank()}_dataset_log.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Dataset Summary\n")
            f.write(f"Total tasks: {len(self.tasks)}\n")
            f.write(f"Token counts: {self.num_tokens}\n")
            for i, task in enumerate(self.tasks):
                f.write(f"\nTask {i}:\n")
                f.write(json.dumps(task, indent=2) + "\n")
