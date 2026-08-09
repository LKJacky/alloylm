import argparse
import os

import torch

from alloylm.utils import load_jsonl


def get_token_ratio(data):
    adv_sample = []
    # re-normalize
    for item in data:
        item = item.get("others", {}).get("rl_data", {})
        if "entropy" in item:
            adv_sample.append(item["advantages"])
    adv_sample = torch.tensor(adv_sample)
    adv_sample_mean = torch.mean(adv_sample).item()
    adv_sample_std = torch.std(adv_sample).item()

    # analyze
    adv_sample = []
    adv_tokens = []
    entropy = []

    for item in data:
        item = item.get("others", {}).get("rl_data", {})
        if "entropy" in item:
            labels = torch.tensor(item["labels"])
            mask = labels != -100
            item["advantages"] = (item["advantages"] - adv_sample_mean) / (adv_sample_std + 1e-8)
            adv_sample.append(item["advantages"])
            adv_tokens.append(torch.tensor([item["advantages"]] * len(item["log_probs"]))[mask])
            entropy.append(torch.tensor(item["entropy"])[mask])

    adv_sample = torch.tensor(adv_sample)
    adv_tokens = torch.cat(adv_tokens)
    entropy = torch.cat(entropy)
    samll_entropy_ratio = (entropy < 0.1).float().mean().item()
    big_entropy_ratio = (entropy > 2.5).float().mean().item()
    print(
        f"sample mean: {torch.mean(adv_sample):.3f}\t token mean: {torch.mean(adv_tokens):.3f}\t max adv: {adv_sample.max():.3f}\t min adv: {adv_sample.min():.3f}\t len: {len(adv_sample)}"
        + f"\tsample acc: {(adv_sample > 0).float().mean():.3f}\t token acc: {(adv_tokens > 0).float().mean():.3f}"
        + f"\tsmall entropy ratio: {samll_entropy_ratio * 100:.3f}\t big entropy ratio: {big_entropy_ratio * 100:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Input JSON file containing entropy probabilities")
    args = parser.parse_args()

    files = []
    for root, _, fs in os.walk(args.folder):
        for f in fs:
            if f.endswith(".jsonl") and "trainset" in f:
                files.append(os.path.join(root, f))

    files = sorted(files)

    for file in files:
        data = load_jsonl(file)
        print(os.path.basename(file), end="\t")
        get_token_ratio(data)


if __name__ == "__main__":
    main()
