import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str)
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()

    paths = []
    for root, dirs, files in os.walk(args.result):
        if "result.json" in files:
            paths.append(os.path.join(root, "result.json"))
    paths = sorted(paths)

    for path in paths:
        print(path)
        df = pd.DataFrame()
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if k.startswith(args.prefix)}
        for item in data.values():
            item["overlong_ratio"] = item["num_overlong"] / item["result_num"]
        df = pd.concat([df, pd.DataFrame(data).T])

        total_samples = sum(item["result_num"] for item in data.values())
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "summary": {
                            "metric": np.mean([item["metric"] for item in data.values()]),
                            "origin_num": sum(item["origin_num"] for item in data.values()),
                            "result_num": total_samples,
                            "num_overlong": sum(item["num_overlong"] for item in data.values()),
                            "mean_input_tokens": int(
                                sum([item["mean_input_tokens"] * item["result_num"] for item in data.values()])
                                / total_samples
                            ),
                            "mean_output_tokens": int(
                                sum([item["mean_output_tokens"] * item["result_num"] for item in data.values()])
                                / total_samples
                            ),
                            "overlong_ratio": sum([item["num_overlong"] for item in data.values()]) / total_samples,
                        },
                    }
                ).T,
            ]
        )
        df["metric"] = df["metric"].apply(lambda x: f"{x * 100:.2f}" if pd.notnull(x) else x)
        df["overlong_ratio"] = df["overlong_ratio"].apply(lambda x: f"{int(x * 100)}%" if pd.notnull(x) else x)
        df = df.transpose()
        print(df.to_markdown())


if __name__ == "__main__":
    main()
