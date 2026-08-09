import os
import pickle
from collections import defaultdict, deque

import numpy as np
import torch
from torch import distributed as dist

from alloylm.algorithm.base import TaskData
from alloylm.algorithm.rl.rl_algo import TaskSampler

NEGATIVE_SCALE_FACTOR = float(os.environ.get("NEGATIVE_SCALE_FACTOR", "1.0"))
ENTROPY_BIN_SIZE = float(os.environ.get("ENTROPY_BIN_SIZE", "0.5"))
NEGATIVE_SCALE_FACTOR = float(os.environ.get("NEGATIVE_SCALE_FACTOR", "1.0"))


def cispo_loss_func(
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_weight: torch.Tensor,
    policy_loss_cfg: dict,
) -> torch.Tensor:
    assert (labels >= 0).all(), "Labels must be non-negative for loss computation"
    log_probs = logits.log_softmax(dim=-1)
    gathered_logprob = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        log_diff = torch.clamp(gathered_logprob - old_logprobs, -20, 20)
        ratio = log_diff.exp()
        ratio_clip = ratio.clamp(max=1 + policy_loss_cfg["cliprange_high"]).detach()

    pg_loss = -ratio_clip * advantages * gathered_logprob
    pg_loss = pg_loss.sum() * loss_weight

    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return pg_loss, entropy


def batch_normalized_advantages(batch: list[TaskData]):
    metric = defaultdict(list)
    for item in batch:
        metric[item.id].append(item.metric)
    std = np.std([x for y in metric.values() for x in y])
    for key in metric:
        metric[key] = np.mean(metric[key])
    for item in batch:
        item.others["rl_data"]["advantages"] = (item.metric - metric[item.id]) / (std + 1e-8)
    return batch


def remove_last_token_of_entropy_stop_samples(batch: list[TaskData]):
    for item in batch:
        if item.finish_reason == "entropy":
            item.others["rl_data"]["input_ids"] = item.others["rl_data"]["input_ids"][:-1]
            item.others["rl_data"]["labels"] = item.others["rl_data"]["labels"][:-1]
            item.others["rl_data"]["log_probs"] = item.others["rl_data"]["log_probs"][:-1]
            item.others["rl_data"]["entropy"] = item.others["rl_data"]["entropy"][:-1]
    return batch


def penalty_negative_advantages_per_optimize_step(step_data, negative_scale_factor=NEGATIVE_SCALE_FACTOR):
    def _all_reduce_in_place(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            reduce_tensor = tensor.cuda()
            dist.all_reduce(reduce_tensor, op=op)
            if reduce_tensor is not tensor:
                tensor.copy_(reduce_tensor.to(tensor.device))
        return tensor

    def _masked_advantages(batch: dict) -> torch.Tensor:
        return batch["advantages"][batch["labels"] != -100]

    """Scale negative advantages relative to positive ones within a training
    step.

    Args:
        step_data: list of packed batch dicts, each with:
            - "advantages": [1, seq_len] tensor (per-token, repeated from per-trajectory scalar)
            - "labels": [1, seq_len] tensor (-100 for masked positions)

    Returns:
        Modified step_data with scaled negative advantages, or [] if no positive advantages.
    """
    if not step_data:
        return step_data

    positive_sum = torch.tensor(0.0, device=step_data[0]["advantages"].device)
    negative_sum = torch.tensor(0.0, device=step_data[0]["advantages"].device)
    for batch in step_data:
        masked_adv = _masked_advantages(batch)
        positive_sum += masked_adv[masked_adv > 0].sum()
        negative_sum += -masked_adv[masked_adv < 0].sum()
    _all_reduce_in_place(positive_sum, op=dist.ReduceOp.SUM)
    _all_reduce_in_place(negative_sum, op=dist.ReduceOp.SUM)
    positive_sum = positive_sum.item()
    negative_sum = negative_sum.item()

    if positive_sum == 0:
        for batch in step_data:
            batch["advantages"] = batch["advantages"] * 0.0
        return step_data[:1]
    if negative_sum == 0:
        return step_data

    negative_scale = positive_sum / -negative_sum
    for batch in step_data:
        batch["advantages"] = batch["advantages"].clone()
        batch["advantages"][batch["advantages"] < 0] *= negative_scale * negative_scale_factor

    return step_data


@torch.no_grad()
def penalty_negative_advantages_per_optimize_step_by_entropy_bins(
    step_data,
    negative_scale_factor=NEGATIVE_SCALE_FACTOR,
    entropy_bin_size=ENTROPY_BIN_SIZE,
):
    """Scale negative advantages by entropy bin within a training step.

    Args:
        step_data: list of packed batch dicts, each with:
            - "advantages": [1, seq_len] tensor (per-token, repeated from per-trajectory scalar)
            - "labels": [1, seq_len] tensor (-100 for masked positions)
            - "entropy": [1, seq_len] tensor
        entropy_bin_size: Width of each entropy bin.

    Returns:
        Modified step_data with negative advantages reweighted independently in
        each entropy bin.
    """
    if not step_data:
        return step_data
    if entropy_bin_size <= 0:
        raise ValueError("entropy_bin_size must be positive.")

    # Entropy is bounded by ln(vocab_size); 20 covers vocab up to ~5e8 with temperature=1.
    num_bins = max(1, int(20 / entropy_bin_size))
    device = step_data[0]["advantages"].device
    positive_sum = torch.zeros(num_bins, device=device, dtype=torch.float32)
    negative_sum = torch.zeros(num_bins, device=device, dtype=torch.float32)
    for batch in step_data:
        mask = batch["labels"] != -100
        masked_adv = batch["advantages"][mask].float()
        masked_entropy = batch["entropy"][mask]
        if masked_adv.numel() == 0:
            continue
        bin_idx = torch.floor(masked_entropy.clamp_min(0) / entropy_bin_size).long()
        assert bin_idx.max().item() < num_bins, (
            f"Entropy bin index {bin_idx.max().item()} exceeds num_bins {num_bins}. Consider increasing ENTROPY_BIN_SIZE."
        )
        positive_mask = masked_adv > 0
        negative_mask = masked_adv < 0
        if positive_mask.any():
            positive_sum.scatter_add_(0, bin_idx[positive_mask], masked_adv[positive_mask])
        if negative_mask.any():
            negative_sum.scatter_add_(0, bin_idx[negative_mask], -masked_adv[negative_mask])

    if dist.is_available() and dist.is_initialized():
        positive_sum = positive_sum.cuda()
        dist.all_reduce(positive_sum, op=dist.ReduceOp.SUM)
        negative_sum = negative_sum.cuda()
        dist.all_reduce(negative_sum, op=dist.ReduceOp.SUM)
        positive_sum = positive_sum.to(device)
        negative_sum = negative_sum.to(device)

    if positive_sum.sum().item() == 0:
        for batch in step_data:
            batch["advantages"] = batch["advantages"] * 0.0
        return step_data[:1]
    elif negative_sum.sum().item() == 0:
        return step_data
    else:
        negative_scale = (positive_sum / negative_sum.clamp_min(1e-8)).clamp_(max=1.0)
        for batch in step_data:
            mask = batch["labels"] != -100
            if mask.any():
                adv = batch["advantages"].clone()
                masked_adv = adv[mask]
                masked_entropy = batch["entropy"][mask]
                bin_idx = torch.floor(masked_entropy.clamp_min(0) / entropy_bin_size).long().clamp_(max=num_bins - 1)
                neg = masked_adv < 0
                masked_adv[neg] *= negative_scale[bin_idx[neg]] * negative_scale_factor
                adv[mask] = masked_adv
                batch["advantages"] = adv

        return step_data


# sampler


class FilterSampler(TaskSampler):
    def __init__(self, dataset, tb_writer):
        super().__init__(dataset, tb_writer)

    def __iter__(self):
        def callback_wrapper(idx):
            def callback(results):
                acc = np.mean([1 if result.metric == 1 else 0 for result in results])
                if acc > 0.9 and idx in self.wait:
                    self.wait.remove(idx)

            return callback

        for x in super().__iter__():
            assert isinstance(self.wait[-1], int), "The items in wait should be indices of tasks."
            x.callback = callback_wrapper(self.wait[-1])
            yield x

    def resume(self, state):
        self.wait = deque(state["wait"] + state.get("finished", []))
        self.epoch = state["epoch"]
        self.rng.setstate(pickle.loads(bytes.fromhex(state["rng"])))
