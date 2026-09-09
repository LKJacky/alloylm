import argparse
import asyncio
import time

from transformers import AutoTokenizer

from alloylm.algorithm.sft.dataset import SFTPackDatasetConfig
from alloylm.utils import get_chat_template_from_tokenizer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to the folder containing JSONL files")
    parser.add_argument("--num-workers", type=int, default=-1, help="Ray tokenization actors (default: half CPUs)")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    config = SFTPackDatasetConfig(
        file_paths=[args.folder],
        sample_ratios=[1.0],
        max_length=64 * 1024,
        num_tokenize_workers=args.num_workers,
    )
    chat_template = get_chat_template_from_tokenizer(tokenizer)
    t0 = time.time()
    dataset = await config.build(tokenizer, chat_template)
    print(len(dataset))
    t1 = time.time()
    print(f"Tokenization speed: {t1 - t0} seconds")


if __name__ == "__main__":
    asyncio.run(main())
