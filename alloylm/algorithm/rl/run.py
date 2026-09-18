import argparse
import asyncio
import importlib

from alloylm.algorithm.rl.rl_algo import RLTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config_module = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    trainer = RLTrainer(config_module.config)

    await trainer.lazy_init()
    await trainer.fit()


if __name__ == "__main__":
    asyncio.run(main())
