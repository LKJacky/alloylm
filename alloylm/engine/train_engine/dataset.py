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

from .utils import pad_and_split_for_sp


def distangle_train_or_not_train(messages: list[dict], chat_template):
    # distinguish between has loss or not.
    converted_messages = []
    pre_text = ""
    render = chat_template.render if hasattr(chat_template, "render") else chat_template
    for i, message in enumerate(messages):
        if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            add_generation_prompt = True
        else:
            add_generation_prompt = False
        text = render(messages=messages[: i + 1], add_generation_prompt=add_generation_prompt)
        if message["role"] == "assistant":
            has_loss = True
        else:
            has_loss = False
        converted_messages.append((text[len(pre_text) :], has_loss))  # Append only the new part of the text
        pre_text = text
    return converted_messages


def tokenize_messages(messages, chat_template, tokenizer):

    def tokenize(converted_messages: list[(str, bool)], tokenizer):
        input_ids = []
        labels = []
        for text, has_loss in converted_messages:
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(tokenized)
            if has_loss:
                labels.extend(tokenized)
            else:
                labels.extend([-100] * len(tokenized))  # Mask out tokens that don't contribute to loss
        return input_ids, labels

    converted_messages = distangle_train_or_not_train(messages, chat_template)
    return tokenize(converted_messages, tokenizer)


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
def task_collate_fn(batch, sp_rank=0, sp_size=1):
    def collated_one_sample(single_batch):
        def pad_and_split(x, value=0):
            if x is None:
                return None
            return pad_and_split_for_sp(torch.as_tensor(x), value, sp_size, sp_rank, dim=-1)

        keys = single_batch[0].keys()
        input_ids = torch.tensor([x for item in single_batch for x in item["input_ids"]])
        labels = torch.tensor([x for item in single_batch for x in item["labels"]])
        num_tokens = [item["num_tokens"] for item in single_batch]
        position_ids = torch.tensor([x for n in num_tokens for x in range(n)])

        shift_labels = torch.roll(labels, shifts=-1, dims=-1)
        shift_labels[torch.tensor(num_tokens).cumsum(0) - 1] = -100

        advantages = (
            [x for item in single_batch for x in [item["advantages"]] * item["num_tokens"]]
            if "advantages" in keys
            else None
        )
        old_log_probs = [x for item in single_batch for x in item["log_probs"]] if "log_probs" in keys else None
        entropy = [x for item in single_batch for x in item["train_entropy"]] if "train_entropy" in keys else None

        input_ids = pad_and_split(input_ids)
        shift_labels = pad_and_split(shift_labels, value=-100)
        position_ids = pad_and_split(position_ids)
        advantages = pad_and_split(advantages)
        old_log_probs = pad_and_split(old_log_probs)
        entropy = pad_and_split(entropy)
        padding = input_ids.numel() * sp_size - sum(num_tokens)
        seq_lens = [*num_tokens, *([padding] if padding else [])]

        return {
            "input_ids": input_ids.unsqueeze(0),
            "labels": shift_labels.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "seq_lens": torch.tensor(seq_lens),
            # "num_tokens": torch.tensor(num_tokens),
            "advantages": advantages.unsqueeze(0) if advantages is not None else None,
            "old_log_probs": old_log_probs.unsqueeze(0) if old_log_probs is not None else None,
            "entropy": entropy.unsqueeze(0) if entropy is not None else None,
            "ids": [item["id"] for item in single_batch],
        }

    batch = [x for y in batch for x in y]
    return collated_one_sample(batch)


@torch.inference_mode()
def sft_collate_fn(batch, sp_size=1, sp_rank=0):
    """Collate a list of ``SFTDataset`` samples into one packed sequence.

    Each sample is the dict returned by ``SFTDataset.__getitem__`` with
    ``input_ids``/``labels`` (token id lists) and ``seq_lens`` (per-sequence
    lengths). The output packs everything into a single ``[1, total]`` sequence
    plus a ``[num_seqs]`` ``seq_lens`` tensor, matching what
    ``TrainEngine.step_sft`` consumes.
    """
    input_ids = torch.tensor([x for item in batch for x in item["input_ids"]])
    labels = torch.tensor([x for item in batch for x in item["labels"]])
    seq_lens = [x for item in batch for x in item["seq_lens"]]
    position_ids = torch.tensor([x for n in seq_lens for x in range(n)])

    # Shift labels without training across packed-sequence boundaries.
    shift_labels = torch.roll(labels, shifts=-1, dims=-1)
    shift_labels[torch.tensor(seq_lens).cumsum(0) - 1] = -100

    # deal sp
    input_ids = pad_and_split_for_sp(input_ids, value=0, sp_size=sp_size, sp_rank=sp_rank, dim=-1)
    shift_labels = pad_and_split_for_sp(shift_labels, value=-100, sp_size=sp_size, sp_rank=sp_rank, dim=-1)
    position_ids = pad_and_split_for_sp(position_ids, value=0, sp_size=sp_size, sp_rank=sp_rank, dim=-1)
    padding = input_ids.numel() * sp_size - sum(seq_lens)
    if padding:
        seq_lens.append(padding)

    return {
        "input_ids": input_ids.unsqueeze(0),
        "shift_labels": shift_labels.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
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
            input_id, label = tokenize_messages(data["messages"], self.chat_template, self.tokenizer)
            input_ids.extend(input_id)
            labels.extend(label)
            seq_lens.append(len(input_id))
            assert num_token == len(input_id), f"num_token mismatch: {num_token} != {len(input_id)}"

        return {
            "input_ids": input_ids,
            "labels": labels,
            "seq_lens": seq_lens,
        }
