import random
import re

import torch
from torch import distributed as dist

from alloylm.algorithm.base import TaskData


def entropy_adjust_loss_func(
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_weight: torch.Tensor,
    policy_loss_cfg: dict,
    # below should be set by partial, not passed in by trainer
    entropy_upper_bound: float = 0.65,
    entropy_lower_bound: float = 0.4,
    tau_upper: float = 0.0,
    tau_lower: float = 0.0,
    coeff_min_upper: float = 0.2,
    coeff_min_lower: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-clip PPO loss with entropy-based advantage adjustment.

    - When avg entropy > upper bound: scale down negative advantages (reduce punishment)
    - When avg entropy < lower bound: scale down positive advantages (reduce reward)

    Args:
        logits: [1, C, V] float32 logits from lm_head
        labels: [1, C] non-negative token ids
        old_logprobs: [1, C] old log probabilities
        advantages: [1, C] advantage values
        loss_weight: scalar multiplier for the loss
        policy_loss_cfg: dict with cliprange_high, cliprange_low, cliprange_c
        entropy_upper_bound: upper entropy threshold for advantage scaling
        entropy_lower_bound: lower entropy threshold for advantage scaling
        tau_upper: temperature for upper bound sigmoid scaling
        tau_lower: temperature for lower bound sigmoid scaling
        coeff_min_upper: minimum coefficient when entropy exceeds upper bound
        coeff_min_lower: minimum coefficient when entropy below lower bound

    Returns:
        (loss_scalar, entropy_tensor [1, C])
    """
    assert (labels >= 0).all(), "Labels must be non-negative for loss computation"
    log_probs = logits.log_softmax(dim=-1)

    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        avg_entropy = entropy.mean()

        if avg_entropy > entropy_upper_bound:
            delta = (avg_entropy - entropy_upper_bound) / entropy_upper_bound
            s = torch.sigmoid(-delta / max(tau_upper, 1e-8)).item()
            coeff = coeff_min_upper + (1 - coeff_min_upper) * s / 0.5
            advantages = torch.where(advantages < 0, advantages * coeff, advantages)
        elif avg_entropy < entropy_lower_bound:
            delta = (entropy_lower_bound - avg_entropy) / entropy_lower_bound
            s = torch.sigmoid(-delta / max(tau_lower, 1e-8)).item()
            coeff = coeff_min_lower + (1 - coeff_min_lower) * s / 0.5
            advantages = torch.where(advantages > 0, advantages * coeff, advantages)

    # PPO loss
    gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_diff = torch.clamp(gathered - old_logprobs, -20, 20)
    ratio = log_diff.exp()
    pg_loss1 = -ratio * advantages
    pg_loss2 = -ratio.clamp(1 - policy_loss_cfg["cliprange_low"], 1 + policy_loss_cfg["cliprange_high"]) * advantages
    clip_pg_loss1 = torch.max(pg_loss1, pg_loss2)
    # Dual-clip PPO
    pg_loss3 = -advantages * policy_loss_cfg["cliprange_c"]
    clip_pg_loss2 = torch.min(pg_loss3, clip_pg_loss1)
    pg_losses = torch.where(advantages < 0, clip_pg_loss2, clip_pg_loss1)
    pg_loss = torch.sum(pg_losses) * loss_weight
    return pg_loss, entropy


def overlong_fileter_data_post_process(batch: list[TaskData]) -> list[TaskData]:
    """Overlong filtering: drop overlong samples but keep one random overlong per group.

    Converted from OverlongRLOOGroupEntropyEstimator._compute_overlong_mask.
    Samples with finish_reason != "stop" are considered overlong.
    Also filters out zero-advantage samples.
    """
    stopped = []
    overlong = []
    for item in batch:
        if item.finish_reason == "stop":
            stopped.append(item)
        else:
            overlong.append(item)
    if overlong:
        stopped.append(random.choice(overlong))
    return [x for x in stopped if x.others["rl_data"]["advantages"] != 0]


# step_data_process_func


def penalty_negative_advantages_per_optimize_step(step_data, negative_scale_factor=1.0):
    """Scale negative advantages relative to positive ones within a training
    step.

    Args:
        step_data: list of packed batch dicts, each with:
            - "advantages": [1, seq_len] tensor (per-token, repeated from per-trajectory scalar)
            - "labels": [1, seq_len] tensor (-100 for masked positions)

    Returns:
        Modified step_data with scaled negative advantages, or [] if no positive advantages.
    """
    positive_sum = torch.tensor(0.0, device=step_data[0]["advantages"].device)
    negative_sum = torch.tensor(0.0, device=step_data[0]["advantages"].device)
    negative_sum = 0.0
    for batch in step_data:
        adv = batch["advantages"]
        mask = batch["labels"] != -100
        masked_adv = adv[mask]
        positive_sum += masked_adv[masked_adv > 0].sum()
        negative_sum += masked_adv[masked_adv < 0].sum()
    positive_sum = positive_sum.cuda()
    negative_sum = negative_sum.cuda()
    dist.all_reduce(positive_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(negative_sum, op=dist.ReduceOp.SUM)
    positive_sum = positive_sum.item()
    negative_sum = negative_sum.item()

    if positive_sum == 0:
        for batch in step_data:
            batch["advantages"] = batch["advantages"] * 0.0
        return step_data[:1]
    if negative_sum == 0:
        return step_data

    negative_scale = min(1.0, positive_sum / -negative_sum)
    for batch in step_data:
        batch["advantages"] = batch["advantages"].clone()
        batch["advantages"][batch["advantages"] < 0] *= negative_scale * negative_scale_factor

    return step_data


def has_gibberish(text: str, threshold: int = 20) -> bool:
    """Detect unexpected Unicode characters (Chinese, Arabic, etc.) in English
    math text.

    A few CJK chars may appear legitimately (e.g., problem format references). The threshold counts total non-Latin
    chars to distinguish corruption from occasional use.
    """
    gibberish_pattern = re.compile(r"[\u4e00-\u9fff\u0600-\u06ff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef]+")
    matches = gibberish_pattern.findall(text)
    total_chars = sum(len(m) for m in matches)
    return total_chars >= threshold
