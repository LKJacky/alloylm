import gc
import math
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import wait
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, TypedDict

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import transformers
from mmengine import mkdir_or_exist
from mmengine.runner import set_random_seed
from mmengine.utils.dl_utils import collect_env
from pydantic import BaseModel
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
)
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from alloylm.engine.model import AlloyLMModel, TrainInput
from alloylm.engine.spmd import init_dist
from alloylm.engine.train_engine.utils import (
    get_logger,
)

from .dataset import (
    SFTData,
    SFTDataset,
    SoftPackDataset,
    TaskDataset,
    sft_collate_fn,
    task_collate_fn,
)
from .utils import clip_grad_norm_, profile_time_and_memory

logger = get_logger()


class RLInput(TypedDict, total=False):
    input_ids: list[int]
    labels: list[int]
    inference_logprobs: list[float]
    advantages: float


# loss


def default_loss_func(
    logits: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_weight: torch.Tensor,
    policy_loss_cfg: dict,
):
    """Pure function for torch.func: log_softmax → gather → policy loss.

    Args:
        logits: [1, C, V] float32 logits from lm_head
        labels: [1, C] non-negative token ids
        old_logprobs: [1, C] old log probabilities
        advantages: [1, C] advantage values
        loss_weight: scalar multiplier for the loss
        policy_loss_cfg: dict with cliprange_high, cliprange_low, cliprange_c

    Returns:
        (loss_scalar, entropy_tensor [1, C])
    """
    assert (labels >= 0).all(), "Labels must be non-negative for loss computation"
    log_probs = logits.log_softmax(dim=-1)

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
    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return pg_loss, entropy


class ChunkPolicyLoss(torch.autograd.Function):
    """Compute policy loss in chunks to avoid materializing full [1, seq,
    vocab] logits."""

    @staticmethod
    def chunk_forward(hidden_states, head_weight, loss_fn, loss_kwargs):
        logits = F.linear(hidden_states, head_weight).float()
        return loss_fn(logits, **loss_kwargs)

    @staticmethod
    def forward(ctx, hidden_states, head_weight, loss_fn, loss_kwargs_chunks, chunk_size):
        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_hidden = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        h_chunks = torch.split(hidden_states, chunk_size, dim=1)
        grad_chunks = torch.split(grad_hidden, chunk_size, dim=1)

        entropy_parts = []
        for i in range(len(h_chunks)):
            (chunk_grad_h, chunk_grad_w), (chunk_loss, entropy) = torch.func.grad_and_value(
                ChunkPolicyLoss.chunk_forward, argnums=(0, 1), has_aux=True
            )(h_chunks[i], head_weight, loss_fn, loss_kwargs_chunks[i])

            accumulated_loss.add_(chunk_loss)
            grad_chunks[i].copy_(chunk_grad_h)
            grad_weight.add_(chunk_grad_w)
            entropy_parts.append(entropy.detach())

        ctx.save_for_backward(grad_hidden, grad_weight)
        return accumulated_loss, torch.cat(entropy_parts, dim=1)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]
        return grad_input, grad_weight, None, None, None


def split_for_sp(shifted_labels, sequence_parallel_mesh, pad_value=-100):
    from alloylm.engine.train_engine.utils import (
        pad_to_multiple_of,
        split_for_sequence_parallel,
    )

    if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
        multiple_of = sequence_parallel_mesh.size() * 1
    else:
        multiple_of = 1

    _labels = pad_to_multiple_of(shifted_labels, pad_value, multiple_of, 1)

    if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
        _labels = split_for_sequence_parallel(_labels, dim=1, sp_mesh=sequence_parallel_mesh)
    return _labels


# logger


def log_format(rank, debug=False):
    formatter = f"[AlloyLM][RANK {rank}]"
    formatter += "[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]"

    if debug:
        formatter += "[<cyan>{name}</cyan>:"
        formatter += "<cyan>{function}</cyan>:"
        formatter += "<cyan>{line}</cyan>]"

    formatter += " <level>{message}</level>"
    return formatter


# state


class TrainState(Stateful):
    def __init__(self, total_steps, seed):
        super().__init__()

        self.seed = seed
        self.cur_step = 0
        self.total_steps = total_steps
        self.if_nan_skip_steps = 0
        self.num_optimize = 0

    def load_state_dict(self, state_dict):
        assert self.total_steps == state_dict["total_steps"]
        self.cur_step = state_dict["current_step"]
        self.if_nan_skip_steps = state_dict["if_nan_skip_steps"]
        self.num_optimize = state_dict.get("num_optimize", self.cur_step)

    def state_dict(self):
        return {
            "seed": self.seed,
            "current_step": self.cur_step,
            "total_steps": self.total_steps,
            "if_nan_skip_steps": self.if_nan_skip_steps,
            "num_optimize": self.num_optimize,
        }

    def step(self):
        self.cur_step = self.cur_step + 1

    def found_nan(self):
        self.if_nan_skip_steps += 1


# main


class TrainEngineConfig(BaseModel):
    max_length: int = 1024  # max length for training

    # Optimization Related Settings
    lr: float = 1e-6
    wd: float = 0.1

    # for scheduler
    scheduler_type: str = "constant"  # "constant" or "cosine"
    warmup_ratio: float = 0.0
    lr_min: float = 0.0

    # General Settings
    work_dir: str = "work_dirs"
    seed: int = 0
    num_workers: int = 1

    # DAPO Related Settings
    total_training_steps: int = 10000
    num_optimize_per_step: int = 1
    loss_func: object = None
    h_clip: float = 0.2
    l_clip: float = 0.2
    step_data_process_func: object = None

    # INFRA
    sp_size: int = 1
    chunk_loss_size: int = 512


class TrainEngine:
    # init
    def __init__(
        self,
        model: AlloyLMModel,
        tokenizer: AutoTokenizer,
        config: TrainEngineConfig,
    ):
        self.patched_llm = model
        self.tokenizer = tokenizer
        self.config = config

        self.optimize_steps = 0
        self.start_step = 0

        set_random_seed(self.config.seed)

        # init dist
        init_dist()
        self.rank = dist.get_rank()
        self.gloo_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
        get_logger().info(f"init rank: {self.rank}")

        # prepare workdir and logger
        mkdir_or_exist(self.config.work_dir)
        log_file = os.path.join(self.config.work_dir, f"rank{self.rank}.log")
        get_logger().remove()
        get_logger().add(sys.stderr, level="INFO", format=log_format(self.rank, False))
        get_logger().add(log_file, format=log_format(self.rank), backtrace=True, catch=True, level="DEBUG")
        get_logger().info(self.config)
        if self.rank == 0:
            self.tb_writer = SummaryWriter(log_dir=self.config.work_dir)
        else:
            self.tb_writer = None

        # print env
        if self.rank == 0:
            env = collect_env()
            env["Transformers"] = transformers.__version__
            runtime_env = OrderedDict()
            runtime_env.update(env)
            runtime_env["Seed"] = self.config.seed
            runtime_env["World Size"] = os.environ["WORLD_SIZE"]
            runtime_env_info = "\n    " + "\n    ".join(f"{k}: {v}" for k, v in runtime_env.items())
            dash_line = "-" * 60
            get_logger().info("\n" + dash_line + "\nRuntime environment:" + runtime_env_info + "\n" + dash_line + "\n")

        self.dp_mesh = self.patched_llm.fsdp_config.train_mesh["dp"]
        self.sp_mesh = self.patched_llm.fsdp_config.train_mesh["sp"]
        self.dp_size = self.dp_mesh.size()
        self.setup_optim()
        dist.barrier()
        gc.collect()

    def setup_optim(self):
        self.total_steps = self.config.total_training_steps
        get_logger().info(f"Total training steps: {self.total_steps}")

        self.train_state = TrainState(total_steps=self.total_steps, seed=self.config.seed)

        self.optimizer = AdamW(
            [param for param in self.patched_llm.parameters() if param.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.wd,
            betas=(0.9, 0.95),
        )  # TODO: check whether some parameters do not need weight decay, e.g., bias.

        # warm up setup
        self.warmup_steps = int(self.config.warmup_ratio * self.total_steps)
        # self.cosine_scheduler = CosineAnnealingWithWarmup(
        #     self.optimizer, min_lr=self.args.lr_min, warmup_steps=self.warmup_steps, total_steps=self.total_steps
        # )

        def cos_lr_lambda(
            step,
            lr=self.config.lr,
            lr_min=self.config.lr_min,
            warm_up_steps=self.warmup_steps,
            total_steps=self.total_steps,
        ):
            """Warmup + cosine-decay learning-rate schedule (used by SFT)."""
            if step < warm_up_steps:
                return step / max(1, warm_up_steps)
            progress = (step - warm_up_steps) / max(1, total_steps - warm_up_steps)
            min_ratio = lr_min / lr if lr > 0 else 0.0
            return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2

        if self.config.scheduler_type == "cosine":
            self.cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, cos_lr_lambda)
        else:
            self.cosine_scheduler = ConstantLR(
                optimizer=self.optimizer, factor=1
            )  # TODO: support both cosine scheduler and constant scheduler

    def lazy_init(self):
        pass

    # training step

    def step(self, batch: list[dict[str, Any]], step):  # step rl
        dataset = TaskDataset(batch)
        pack_dataset = SoftPackDataset([dataset], target=self.config.max_length)
        dataloader = DataLoader(
            pack_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=task_collate_fn,
            persistent_workers=False,
            sampler=DistributedSampler(
                pack_dataset,
                num_replicas=self.dp_size,
                rank=self.dp_mesh.get_local_rank(),
                shuffle=True,
                seed=self.config.seed + step,
                drop_last=False,
            ),
        )
        dataloader.sampler.set_epoch(0)
        batch = self.compute_log_prob(batch, dataloader)

        batch_info = self._log_logprob_diff(batch)  # only compute logprobs in current rank

        dataloader.sampler.set_epoch(0)
        self.train_rl_step(batch, step, dataloader, batch_info)

        return batch_info

    def set_sft_data(
        self,
        batch: list[SFTData],
        jsonl_sources: list,
        chat_template: Callable,
    ):
        """SFT training step: pack tokenized samples and run cross-entropy training."""

        # build dataset
        dataset = SFTDataset(batch, jsonl_paths=jsonl_sources, tokenizer=self.tokenizer, chat_template=chat_template)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=sft_collate_fn,
            persistent_workers=False,
            sampler=DistributedSampler(
                dataset,
                num_replicas=self.dp_size,
                rank=self.dp_mesh.get_local_rank(),
                shuffle=False,
                drop_last=False,
            ),
        )
        dataloader.sampler.set_epoch(0)
        self.sft_data_iter = iter(dataloader)

    def step_sft(self, num_micro_steps: int) -> dict[str, float]:
        """Train one SFT step: standard causal-LM cross-entropy on (input_ids, labels).

        Samples are packed by SoftPackDataset into `max_length` sequences; each packed
        sequence is a forward+backward (gradient accumulation), then a single optimizer
        step with gradient clipping.
        """
        self.patched_llm.train()
        micro_batch = []
        for i in range(num_micro_steps):
            micro_batch.append(next(self.sft_data_iter))

        # total number of supervised tokens in this step (reduced across dp ranks)
        total_tokens = torch.tensor(0.0, device="cuda")
        for packed_batch in micro_batch:
            total_tokens += (packed_batch["labels"] != -100).sum().float().cuda()
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        total_tokens = total_tokens.item()
        if total_tokens == 0:
            get_logger().warning("No supervised tokens in this step, skipping.")
            return {"loss": 0.0, "grad_norm": 0.0, "num_tokens": 0}

        step_t0 = time.time()
        step_loss = 0.0
        for packed_batch in micro_batch:
            input_ids = packed_batch["input_ids"].cuda()
            labels = packed_batch["labels"].cuda()
            seq_lens = packed_batch["seq_lens"].cuda()
            position_ids = torch.cat([torch.arange(n) for n in seq_lens.tolist()], dim=0).cuda().unsqueeze_(0)

            logits = self.patched_llm.train_forward(
                TrainInput(input_ids=input_ids, position_ids=position_ids, seq_lens=seq_lens.int())
            )  # [1, seq, vocab]

            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            n_valid = (shift_labels >= 0).sum()
            if n_valid == 0:
                continue
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            # normalize so the optimizer step averages over all supervised tokens
            loss = loss * (n_valid / total_tokens * self.dp_size)
            loss.backward()
            step_loss += loss.item()

        grad_norm = clip_grad_norm_([param for param in self.patched_llm.parameters() if param.requires_grad], 1.0)
        if grad_norm.isnan() or grad_norm.isinf():
            get_logger().warning(
                f"[SFT Step {self.train_state.cur_step}] grad norm is NaN/Inf, skipping optimizer step."
            )
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.cosine_scheduler.step()

        # reduce loss across dp ranks for logging
        reduced_loss = torch.tensor(step_loss, device="cuda")
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
        step_time = time.time() - step_t0

        get_logger().info(
            f"[SFT] Step {self.train_state.cur_step}/{self.total_steps}  "
            f"loss: {reduced_loss.item():.4f}  "
            f"grad_norm: {grad_norm:.2f}  "
            f"lr: {self.cosine_scheduler.get_last_lr()[0]:.6f}  "
            f"tokens: {int(total_tokens)}  "
            f"time: {step_time:.2f}s  "
            f"Mem: {torch.cuda.max_memory_allocated() / 1024**3:.1f} G"
        )

        self.train_state.step()
        return {"loss": reduced_loss.item(), "grad_norm": grad_norm.item(), "num_tokens": int(total_tokens)}

    @torch.no_grad()
    def _log_logprob_diff(self, tasks: list[dict[str, Any]]):
        try:
            infer_logprobs = []
            train_logprobs = []
            labels = []
            entropy = []

            for item in tasks:
                if "log_probs" in item:
                    labels.extend(item["labels"])
                    infer_logprobs.extend(item["inference_logprobs"])
                    train_logprobs.extend(item["log_probs"])
                    entropy.extend(item["train_entropy"])

            infer_logprobs = torch.tensor(infer_logprobs)
            train_logprobs = torch.tensor(train_logprobs)

            mask = torch.tensor(labels) != -100
            shift_mask = torch.roll(mask, shifts=-1, dims=0)
            infer_logprobs = infer_logprobs[mask]
            train_logprobs = train_logprobs[shift_mask]
            entropy = torch.tensor(entropy)[shift_mask]

            logprob_diff = (train_logprobs - infer_logprobs).abs()
            has_tokens = logprob_diff.numel() > 0

            # Local stats — use safe defaults so empty ranks don't affect reduction
            local_sum_diff = logprob_diff.sum() if has_tokens else torch.tensor(0.0)
            local_num_tokens = torch.tensor(logprob_diff.numel(), dtype=torch.float64)
            local_max_diff = logprob_diff.max() if has_tokens else torch.tensor(float("-inf"))
            local_min_diff = logprob_diff.min() if has_tokens else torch.tensor(float("inf"))
            local_min_prob_infer = infer_logprobs.min().exp() if has_tokens else torch.tensor(float("inf"))
            local_min_prob_train = train_logprobs.min().exp() if has_tokens else torch.tensor(float("inf"))
            entropy_sum = entropy.sum() if has_tokens else torch.tensor(0.0)

            # Reduce across data-parallel ranks
            stats = torch.stack(
                [
                    local_sum_diff,
                    local_num_tokens.float(),
                    entropy_sum,
                    local_max_diff,
                    local_min_diff,
                    local_min_prob_infer,
                    local_min_prob_train,
                ]
            ).cuda()
            sum_stats = stats[:3].clone()
            max_stats = stats[3:4].clone()
            min_stats = stats[4:].clone()

            dist.all_reduce(sum_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_stats, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_stats, op=dist.ReduceOp.MIN)

            global_sum_diff = sum_stats[0].item()
            global_num_tokens = sum_stats[1].item()
            entropy_mean = sum_stats[2].item() / max(global_num_tokens, 1)
            global_max_diff = max_stats[0].item()
            global_min_diff = min_stats[0].item()
            global_min_prob_infer = min_stats[1].item()
            global_min_prob_train = min_stats[2].item()

            avg_diff = global_sum_diff / max(global_num_tokens, 1)

            return {
                "logprob_diff/avg_diff": avg_diff,
                "logprob_diff/max_diff": global_max_diff,
                "logprob_diff/min_diff": global_min_diff,
                "logprob_diff/num_tokens": global_num_tokens,
                "logprob_diff/min_prob_infer": global_min_prob_infer,
                "logprob_diff/min_prob_train": global_min_prob_train,
                "logprob_diff/entropy_mean": entropy_mean,
            }
        except Exception as e:
            get_logger().warning(f"Failed to log logprob diff: {e}")
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def train_rl_step(
        self, trajectories: list[dict[str, Any]], rl_step: int, rl_dataloader: DataLoader, batch_info: dict
    ) -> dict[str, float]:
        """Train model on collected trajectories using GRPO with varlen
        support."""
        self.patched_llm.train()

        # dataset
        num_steps = min(self.config.num_optimize_per_step, len(rl_dataloader))
        if num_steps == 0:
            return num_steps
        max_iters_per_step = (len(rl_dataloader) + num_steps - 1) // num_steps
        num_steps = (len(rl_dataloader) + max_iters_per_step - 1) // max_iters_per_step

        data_iter = iter(rl_dataloader)
        get_logger().info(
            f"[Train Data] {len(trajectories)} traj are packed to dataloader {len(rl_dataloader)}. {num_steps} steps with {max_iters_per_step} iters per step, "
        )

        for step_i in range(num_steps):
            # info
            step_t0 = time.time()
            step_policy_loss = 0.0

            # prepare batch data
            step_data = []
            for _ in range(max_iters_per_step):
                try:
                    step_data.append(next(data_iter))
                except StopIteration:
                    break

            with torch.no_grad():
                if self.config.step_data_process_func is not None:
                    step_data = self.config.step_data_process_func(step_data)

            # compute global num tokens
            with torch.no_grad():
                global_num_tokens = 0
                for packed_batch in step_data:
                    global_num_tokens += (packed_batch["labels"] != -100).sum()
                global_num_tokens = global_num_tokens.cuda()
                dist.all_reduce(global_num_tokens, op=dist.ReduceOp.SUM)
                global_reduce_num = global_num_tokens / self.config.sp_size

                # for per_seq loss
                # global_num_seq = 0
                # for packed_batch in step_data:
                #     global_num_seq += len(packed_batch["num_tokens"])
                # global_num_seq = torch.tensor(global_num_seq).cuda()
                # dist.all_reduce(global_num_seq, op=dist.ReduceOp.SUM)
                # global_reduce_num = global_num_seq

            total_step_entropy = 0
            total_step_entropy_squared = 0
            policy_loss_cfg = {
                "cliprange_high": self.config.h_clip,
                "cliprange_low": self.config.l_clip,
                "cliprange_c": 3,
                **batch_info,
            }
            count_small_entropy = 0
            for packed_batch in step_data:
                # prepare data
                input_ids = packed_batch["input_ids"].cuda()
                labels = packed_batch["labels"].cuda()
                advantages = packed_batch["advantages"].cuda()
                num_tokens = packed_batch["num_tokens"].cuda()
                old_log_probs = packed_batch["old_log_probs"].cuda()
                num_tokens_list = num_tokens.tolist()

                position_ids = [torch.arange(num) for num in num_tokens_list]
                position_ids = torch.cat(position_ids, dim=0).cuda().unsqueeze_(0)

                shifted_labels = torch.roll(labels, shifts=-1, dims=-1)
                shift_labels_for_sp = split_for_sp(shifted_labels, self.sp_mesh, pad_value=-100)
                old_log_prob_sp = split_for_sp(old_log_probs, self.sp_mesh, pad_value=0)
                advantages_sp = split_for_sp(advantages, self.sp_mesh, pad_value=0)

                # Chunk mode: monkey-patch lm_head with ChunkPolicyLoss
                # mask is 1D [seq_sp] — excludes both -100 (mask) and SP padding (negative pad values)
                mask = shift_labels_for_sp[0] >= 0

                def _chunk_loss(
                    hidden_states,
                    _shift_labels_for_sp=shift_labels_for_sp,
                    _old_log_prob_sp=old_log_prob_sp,
                    _advantages_sp=advantages_sp,
                    _mask=mask,
                    _global_reduce_num=global_reduce_num,
                    _policy_loss_cfg=policy_loss_cfg,
                ):
                    # Trim hidden_states to match labels (SP padding stripped)
                    hidden_states = hidden_states[:, : _shift_labels_for_sp.size(1)]
                    loss, entropy = ChunkPolicyLoss.apply(
                        hidden_states[:, _mask],
                        self.patched_llm.lm_head.weight,
                        default_loss_func if self.config.loss_func is None else self.config.loss_func,
                        [
                            {
                                "labels": lc,
                                "old_logprobs": oc,
                                "advantages": ac,
                                "loss_weight": 1.0 / _global_reduce_num,
                                "policy_loss_cfg": _policy_loss_cfg,
                            }
                            for lc, oc, ac in zip(
                                torch.split(_shift_labels_for_sp[:, _mask], self.config.chunk_loss_size, dim=1),
                                torch.split(_old_log_prob_sp[:, _mask], self.config.chunk_loss_size, dim=1),
                                torch.split(_advantages_sp[:, _mask], self.config.chunk_loss_size, dim=1),
                            )
                        ],
                        self.config.chunk_loss_size,
                    )
                    return loss, entropy.detach()

                with self._dispatch_lm_head(_chunk_loss):
                    policy_loss, entropy = self.patched_llm.train_forward(
                        TrainInput(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            seq_lens=num_tokens.int(),
                        )
                    )

                policy_loss = policy_loss * self.dp_size * self.config.sp_size
                policy_loss.backward()

                with torch.no_grad():
                    # entropy is already for masked (valid) positions only
                    step_policy_loss += policy_loss.detach()
                    total_step_entropy += entropy.sum() / global_reduce_num
                    total_step_entropy_squared += entropy.pow(2).sum() / global_reduce_num
                    count_small_entropy += (entropy < 0.1).sum()

            # update parameters
            grad_norm = clip_grad_norm_([param for param in self.patched_llm.parameters() if param.requires_grad], 1.0)
            if grad_norm.isnan() or grad_norm.isinf():
                get_logger().warning(f"[Step {step_i}] The grad norm is NaN or Inf, skip this step.")
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            # log per step
            step_time = time.time() - step_t0
            reduced_step_policy_loss = step_policy_loss.clone().detach()
            dist.all_reduce(reduced_step_policy_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_step_entropy, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_step_entropy_squared, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_small_entropy, op=dist.ReduceOp.SUM)
            entropy_var = (total_step_entropy_squared - total_step_entropy**2).clamp(min=0)
            entropy_std = torch.sqrt(entropy_var)
            tgs = int(global_reduce_num.item() / step_time / self.dp_size / self.config.sp_size)

            get_logger().info(
                f"[RL] (Step {rl_step}) Step "
                f"{step_i + 1}/{num_steps}  "
                f"Optimize Step: {self.optimize_steps}  "
                f"loss: {step_policy_loss:.4f}  "
                f"loss(reduced): {reduced_step_policy_loss.item():.3f}  "
                f"entropy: {total_step_entropy:.4f}  "
                f"entropy_std: {entropy_std:.4f}  "
                f"small_entropy_ratio: {count_small_entropy.item() / global_reduce_num.item():.4f}  "
                f"grad_norm: {grad_norm:.2f}  "
                f"tgs: {tgs}  "
                f"tokens: {global_reduce_num.item()}  "
                f"time: {step_time:.2f}s  "
                f"Mem: {torch.cuda.max_memory_allocated() / 1024**3:.1f} G  "
            )

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/policy_loss", reduced_step_policy_loss, self.optimize_steps)
                self.tb_writer.add_scalar("train/grad_norm", grad_norm, self.optimize_steps)
                self.tb_writer.add_scalar("train/entropy", total_step_entropy, self.optimize_steps)
                self.tb_writer.add_scalar("train/entropy_std", entropy_std, self.optimize_steps)
                self.tb_writer.add_scalar(
                    "train/small_entropy_ratio",
                    count_small_entropy.item() / global_reduce_num.item(),
                    self.optimize_steps,
                )
                self.tb_writer.add_scalar("train/tgs", tgs, self.optimize_steps)
                self.tb_writer.add_scalar("train/global_tokens", global_reduce_num.item(), self.optimize_steps)

            self.optimize_steps += 1
        self.train_state.num_optimize = self.optimize_steps
        return num_steps

    @torch.inference_mode()
    def compute_log_prob(self, tasks: list[dict[str, Any]], dataloader: DataLoader):
        self.patched_llm.eval()
        task_map = {item["id"]: item for item in tasks}

        for batch in dataloader:
            input_ids, labels, num_tokens = (
                batch["input_ids"].cuda(),
                batch["labels"].cuda(),
                batch["num_tokens"].cuda(),
            )
            position_ids = [torch.arange(num) for num in num_tokens]
            position_ids = torch.cat(position_ids, dim=0).cuda().unsqueeze_(0)
            cu_seq_lens = torch.cumsum(torch.IntTensor([0] + num_tokens.tolist()), dim=0).cuda().int()

            shifted_labels = torch.roll(labels, shifts=-1, dims=-1)
            labels_for_sp = split_for_sp(shifted_labels, self.sp_mesh, pad_value=-100)

            def chunk_func(x, labels_for_sp=labels_for_sp):
                return self.__class__._chunkly_compute_logprob_entropy(
                    x, self.patched_llm.lm_head.weight, labels_for_sp, self.config.chunk_loss_size * 16
                )

            with self._dispatch_lm_head(chunk_func):
                log_prob, entropy = self.patched_llm.train_forward(
                    TrainInput(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        seq_lens=num_tokens.int(),
                    )
                )

            if self.sp_mesh and self.sp_mesh.size() > 1:
                log_prob = dist.nn.all_gather(log_prob, group=self.sp_mesh.get_group())
                log_prob = torch.cat(log_prob, dim=1)[:, : input_ids.numel()]

            input_ids, labels, num_tokens, log_prob, entropy = (
                input_ids.cpu(),
                labels.cpu(),
                num_tokens.cpu(),
                log_prob.cpu(),
                entropy.cpu(),
            )
            for i in range(num_tokens.numel()):
                start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
                bid = batch["ids"][i]
                task_map[bid]["log_probs"] = log_prob[:, start:end].flatten().tolist()  # shifted
                task_map[bid]["train_entropy"] = entropy[:, start:end].flatten().tolist()
                assert task_map[bid]["num_tokens"] == num_tokens[i]
                assert num_tokens[i] == end - start
        return tasks

    # for optimizer offloading and activation

    def offload_optimizer(self):
        device = torch.device("cpu")
        for val in self.optimizer.state.values():
            val: dict
            val["exp_avg"] = val["exp_avg"].to(device, non_blocking=True)
            val["exp_avg_sq"] = val["exp_avg_sq"].to(device, non_blocking=True)
        torch.cuda.synchronize()

    def activate_optimizer(self):
        device = torch.device("cuda")
        for val in self.optimizer.state.values():
            val: dict
            val["exp_avg"] = val["exp_avg"].to(device, non_blocking=True)
            val["exp_avg_sq"] = val["exp_avg_sq"].to(device, non_blocking=True)
        torch.cuda.synchronize()

    # checkpointing

    def resume(self, folder):
        ckpt_dir = os.path.join(folder, "ckpt")
        with profile_time_and_memory(f"[Resume from {ckpt_dir}]"):
            _options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True)
            (shard_model_state_dict, shard_optimizer_state_dict) = get_state_dict(
                self.patched_llm, self.optimizer, options=_options
            )
            state_dict = {
                "model": shard_model_state_dict,
                "optimizer": shard_optimizer_state_dict,
                "train_state": self.train_state,
            }
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=ckpt_dir,
            )
            _options = StateDictOptions(cpu_offload=True, strict=False)
            set_state_dict(
                self.patched_llm,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=_options,
            )
        self.start_step = self.train_state.cur_step + 1
        self.optimize_steps = self.train_state.num_optimize

    def checkpoint(self, folder):
        ckpt_dir = os.path.join(folder, "ckpt")
        hf_dir = os.path.join(folder, "hf")

        # save hf model
        with profile_time_and_memory("[HF Checkpoint]"):
            self.patched_llm.save_pretrained(hf_dir)
            if self.rank == 0:
                self.tokenizer.save_pretrained(hf_dir)

        with profile_time_and_memory("[PT Checkpoint]"):
            _options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True)
            (shard_model_state_dict, shard_optimizer_state_dict) = get_state_dict(
                self.patched_llm, self.optimizer, options=_options
            )
            state_dict = {
                "model": shard_model_state_dict,
                "optimizer": shard_optimizer_state_dict,
                "train_state": self.train_state.state_dict(),
            }
            mkdir_or_exist(ckpt_dir)
            self.ckpt_handle = dcp.async_save(state_dict, checkpoint_id=ckpt_dir, process_group=self.gloo_group)
            wait([self.ckpt_handle])

    # for chunk lmhead

    @classmethod
    def _chunkly_compute_logprob_entropy(cls, hidden_states, weight, labels, chunk_size):
        # Defensive trim: labels and hidden_states should have same seq length
        hidden_states = hidden_states[:, : labels.size(1)]
        logprob_parts = []
        entropy_parts = []
        for i in range(0, labels.size(1), chunk_size):
            h_chunk = hidden_states[:, i : i + chunk_size]
            l_chunk = labels[:, i : i + chunk_size]
            logits = F.linear(h_chunk, weight).float()
            lp = logits.log_softmax(dim=-1)
            entropy_parts.append(-(lp.exp() * lp).sum(dim=-1))
            logprob_parts.append(lp.gather(dim=-1, index=l_chunk.clamp(min=0).reshape(1, -1, 1)).reshape(1, -1))

            del logits, lp
        return torch.cat(logprob_parts, dim=1), torch.cat(entropy_parts, dim=1)

    @contextmanager
    def _dispatch_lm_head(self, forward_fn):
        orig_forward = self.patched_llm.lm_head.forward
        self.patched_llm.lm_head.forward = forward_fn
        try:
            yield
        finally:
            self.patched_llm.lm_head.forward = orig_forward
