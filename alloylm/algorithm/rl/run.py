import argparse
import asyncio
import importlib


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config_module = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    trainer = config_module.get_trainer()

    await trainer.lazy_init()
    await trainer.fit()


if __name__ == "__main__":
    asyncio.run(main())
