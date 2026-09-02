import asyncio
import os
import random
from collections.abc import Callable

import aiofiles
import orjson
from pydantic import BaseModel
from torch.utils.data import Dataset


def sft_tokenize(messages, tokenizer, chat_template):
    """Tokenize a conversation exactly as the training engine's ``SFTDataset``
    does.

    Mirrors ``SFTDataset.distangle_train_or_not_train`` + ``tokenize`` in
    ``alloylm/engine/train_engine/dataset.py``: the chat template is applied
    incrementally and each new segment is encoded with
    ``add_special_tokens=False``. Using this length for a pack's ``num_tokens``
    guarantees the engine's ``assert num_token == len(input_id)`` holds (a whole
    conversation encoded in one call does not match, because of segment
    boundaries and implicit special tokens).
    """
    input_ids = []
    pre_text = ""
    for i, msg in enumerate(messages):
        add_gen = (i + 1 < len(messages)) and messages[i + 1]["role"] == "assistant"
        text = chat_template(messages[: i + 1], add_generation_prompt=add_gen)
        toks = tokenizer.encode(text[len(pre_text) :], add_special_tokens=False)
        pre_text = text
        input_ids.extend(toks)
    return input_ids


async def analyze_jsonl_file(file_path, tokenizer, chat_template):
    offsets = []
    num_tokens = []
    async with aiofiles.open(file_path, encoding="utf-8") as f:
        # aiofiles iterates via readline(), so tell() reports the byte offset at
        # the start of each line -- exactly what the engine's seek()+readline()
        # expects.
        offsets.append(await f.tell())
        async for line in f:
            data = orjson.loads(line)
            offsets.append(await f.tell())
            input_ids = await asyncio.to_thread(sft_tokenize, data["messages"], tokenizer, chat_template)
            num_tokens.append(len(input_ids))
    return offsets[:-1], num_tokens


class SFTPackDataset(Dataset):
    def __init__(self, file_paths, sample_ratios, tokenizer, chat_template, max_length, random_seed=42):
        self.file_paths, self.sample_ratios = self.get_files_from_folder(file_paths, sample_ratios)

        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length

        self.data = []  # file_idx, offset, num_tokens
        self.packed_data = []
        self.num_skip_data = 0
        self.rng = random.Random(random_seed)

    async def lazy_init(self):
        tasks = []

        for file_path in self.file_paths:
            tasks.append(analyze_jsonl_file(file_path, self.tokenizer, self.chat_template))

        results = await asyncio.gather(*tasks)
        for file_idx, (file_offsets, file_num_tokens) in enumerate(results):
            # Oversample (ratio >= 1 repeats every line ``full`` times) and/or
            # subsample the fractional remainder, e.g. ratio=2.5 -> all lines
            # twice + a random 50%. ``selected`` holds indices into file_offsets.
            n = len(file_offsets)
            ratio = self.sample_ratios[file_idx]
            full = int(ratio)
            selected = list(range(n)) * full + self.rng.sample(range(n), k=int(n * (ratio - full)))
            for i in selected:
                offset = file_offsets[i]
                num_token = file_num_tokens[i]
                self.data.append((file_idx, offset, num_token))

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


class SFTPackDatasetConfig(BaseModel):
    file_paths: list[str]
    sample_ratios: list[float]
    max_length: int
    chat_template: Callable[..., str] | None = None

    async def build(self, tokenizer) -> SFTPackDataset:
        dataset = SFTPackDataset(self.file_paths, self.sample_ratios, tokenizer, self.chat_template, self.max_length)
        await dataset.lazy_init()
        return dataset
