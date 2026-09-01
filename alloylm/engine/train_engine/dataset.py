import json
import os
import random
from collections.abc import Callable

import numpy as np
import torch
from datasets import concatenate_datasets
from pydantic import BaseModel as PydanticBaseModel
from torch import distributed as dist
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer


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


@torch.inference_mode()
def sft_collate_fn(batch):
    """Collate a list of ``SFTDataset`` samples into one packed sequence.

    Each sample is the dict returned by ``SFTDataset.__getitem__`` with
    ``input_ids``/``labels`` (token id lists) and ``seq_lens`` (per-sequence
    lengths). The output packs everything into a single ``[1, total]`` sequence
    plus a ``[num_seqs]`` ``seq_lens`` tensor, matching what
    ``TrainEngine.step_sft`` consumes.
    """
    input_ids = [x for item in batch for x in item["input_ids"]]
    labels = [x for item in batch for x in item["labels"]]
    seq_lens = [x for item in batch for x in item["seq_lens"]]
    return {
        "input_ids": torch.tensor(input_ids).unsqueeze(0),
        "labels": torch.tensor(labels).unsqueeze(0),
        "seq_lens": torch.tensor(seq_lens),
    }


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


# sft dataset
class SFTData(PydanticBaseModel):
    # shape of below items: [batch, seq]
    jsonl_idx: list[int]
    offsets: list[int]
    num_tokens: list[int]


class SFTDataset(Dataset):
    def __init__(
        self,
        sft_data: list[SFTData],
        jsonl_paths: list[str],
        tokenizer: AutoTokenizer,
        chat_template: Callable[[list[dict]], str],
    ):
        super().__init__()
        self.sft_data = sft_data
        self.jsonl_paths = jsonl_paths
        self.file_handles = [open(path, encoding="utf-8") for path in self.jsonl_paths]  # noqa
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def __len__(self):
        return len(self.sft_data)

    def __getitem__(self, idx):
        sft_data = self.sft_data[idx]
        input_ids = []
        labels = []
        seq_lens = []
        for jsonl_idx, offset, num_token in zip(sft_data.jsonl_idx, sft_data.offsets, sft_data.num_tokens):
            file_handle = self.file_handles[jsonl_idx]
            file_handle.seek(offset)
            line = file_handle.readline()
            data = json.loads(line)
            input_id, label = self.tokenize(self.distangle_train_or_not_train(data["messages"]))
            input_ids.extend(input_id)
            labels.extend(label)
            seq_lens.append(len(input_id))
            assert num_token == len(input_id), f"num_token mismatch: {num_token} != {len(input_id)}"

        return {
            "input_ids": input_ids,
            "labels": labels,
            "seq_lens": seq_lens,
        }

    def distangle_train_or_not_train(self, messages: list[dict]):
        # distinguish between has loss or not.
        converted_messages = []
        pre_text = ""
        for i, message in enumerate(messages):
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                add_generation_prompt = True
            else:
                add_generation_prompt = False
            text = self.chat_template(
                messages[: i + 1], add_generation_prompt=add_generation_prompt
            )  # Convert messages to text using the chat template
            if message["role"] == "assistant":
                has_loss = True
            else:
                has_loss = False
            converted_messages.append((text[len(pre_text) :], has_loss))  # Append only the new part of the text
            pre_text = text
        return converted_messages

    def tokenize(self, converted_messages: list[(str, bool)]):
        input_ids = []
        labels = []
        for text, has_loss in converted_messages:
            tokenized = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(tokenized)
            if has_loss:
                labels.extend(tokenized)
            else:
                labels.extend([-100] * len(tokenized))  # Mask out tokens that don't contribute to loss
        return input_ids, labels
