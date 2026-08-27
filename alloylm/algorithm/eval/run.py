import argparse
import asyncio
import importlib

from alloylm.algorithm.eval.base import EvalConfig, run_eval


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--work_dir", type=str, default="./work_dirs/debug")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--one-by-one", action="store_true")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    config_module = importlib.import_module(args.config.replace(".py", "").replace("/", "."))
    config: EvalConfig = config_module.config
    config.work_dir = args.work_dir
    config.url = args.url
    config.model_name = args.model_name
    config.one_by_one = args.one_by_one
    config.mode = args.mode
    config.resume = args.resume
    config.concurrency = args.concurrency

    print(f"Loaded config: {config}")

    await run_eval(config)
    return 0


if __name__ == "__main__":
    asyncio.run(main())
