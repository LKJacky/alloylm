import argparse
import asyncio
import importlib

from alloylm.algorithm.sft.sft_algo import SFTTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config_module = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    trainer = SFTTrainer(config_module.config)

    await trainer.lazy_init()
    await trainer.fit()


if __name__ == "__main__":
    asyncio.run(main())
