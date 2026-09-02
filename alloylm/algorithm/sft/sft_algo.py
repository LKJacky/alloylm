import asyncio
import gc
import json
import os
import random
import shutil
from typing import Any

import aiofiles
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from alloylm.algorithm.rl.utils import (
    DummySummaryWriter,
    MeasureTime,
    get_logger,
    get_tb_writer,
)
from alloylm.algorithm.sft.dataset import SFTPackDatasetConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.train_engine.dataset import SFTData
from alloylm.engine.train_engine.train_infer_engine import (
    SpmdTrainInferEngine,
    TrainInferEngineConfig,
)
from alloylm.utils import init_logger


class ChatTemplate:
    """Callable wrapping ``tokenizer.apply_chat_template`` to the signature the
    SFT data path expects: ``(messages, add_generation_prompt=False) -> str``.

    The same instance is used both while packing (``sft_tokenize`` /
    ``analyze_jsonl_file`` on the driver) and by the engine's ``SFTDataset``
    (via ``set_sft_data`` on the workers), so the ``num_tokens`` computed while
    packing match what the engine recomputes and its
    ``assert num_token == len(input_id)`` holds. A plain instance (not a bound
    function) is used so it survives being pickled to the Ray actors.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, messages, add_generation_prompt=False):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )


# config


class SFTAlgorithmConfig(BaseModel):
    # model and engine
    llm_config: AlloyLMModelConfig
    engine_config: TrainInferEngineConfig

    # Dataset configuration
    datasets: list[SFTPackDatasetConfig | Any] = []
    max_length: int = 16384  # packing length; must fit the engine's train max_length

    # Algorithm configuration
    global_batch_size: int = 1  # packs pulled per dp-rank per optimizer step (== step_sft's num_micro_steps)
    micro_batch_size: int = -1  # packs pulled per dp-rank per micro-step (== step_sft's micro_batch_size)
    total_training_steps: int = -1
    checkpoint_interval: int = 10
    max_checkpoints: int = 1
    auto_resume: bool = True
    seed: int = 0

    # infra
    work_dir: str = "./work_dirs/"

    def model_post_init(self, context):
        dp_size = self.llm_config.fsdp_config.train_mesh["mesh_shape"][0]
        assert self.global_batch_size % dp_size == 0, (
            f"global_batch_size={self.global_batch_size} must be divisible by dp_size={dp_size}"
        )
        self.micro_batch_size = self.global_batch_size // dp_size
        return super().model_post_init(context)


# sft trainer


class SFTTrainer:
    """Drives SFT training end to end: owns the model engine *and* the packed
    data, runs a fixed ``total_training_steps`` loop, and handles
    checkpoint/resume.

    Mirrors ``RLTrainer`` minus the serve / rollout / eval phases. Unlike RL
    (which keeps its data in a separate ``RLAlgorithm``), SFT trains on static,
    pre-tokenized conversations, so the data-owning logic -- building the packs,
    the per-epoch shuffle, and ``global_step`` bookkeeping -- lives directly on
    the trainer rather than in a separate algorithm object.
    """

    def __init__(self, config: SFTAlgorithmConfig):
        self.config = config
        self.model_engine = SpmdTrainInferEngine(model_config=config.llm_config, engine_config=config.engine_config)

        # Data path: a tokenizer + chat template shared by packing (driver) and
        # the engine's SFTDataset (workers), so num_tokens agree on both sides.
        tokenizer_path = config.llm_config.tokenizer_path or config.llm_config.path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )
        self.chat_template = ChatTemplate(self.tokenizer)
        self.dp_size = config.llm_config.fsdp_config.train_mesh["dp"].size()

        # Populated by lazy_init / advanced by fit.
        self.jsonl_paths: list[str] = []
        self.packs: list[SFTData] = []
        self.steps_per_epoch = 0
        self.global_step = 0
        self.cur_step = 0

        init_logger(self.config.work_dir + "/trainer.log")
        DummySummaryWriter.init_writer(self.config.work_dir)
        self.logger = get_logger()
        self.tb_writer = get_tb_writer()

        self.logger.info(str(self.config.model_dump()))
        self.logger.info(str(self.model_engine.config.model_dump()))

    async def lazy_init(self):
        # Build the datasets while the engine spins up -- both are slow and
        # independent, so run them concurrently.
        await asyncio.gather(self.model_engine.lazy_init(), self._build_data())
        if self.config.auto_resume:
            await self.resume()

    async def _build_data(self):
        # Build every dataset and merge their packs into one global list. Each
        # dataset's file indices are local, so remap them onto the global path
        # list as we concatenate.
        self.jsonl_paths = []
        self.packs = []
        num_skip = 0
        for cfg in self.config.datasets:
            cfg.chat_template = self.chat_template
            cfg.max_length = self.config.max_length
            dataset = await cfg.build(self.tokenizer)
            num_skip += dataset.num_skip_data
            base = len(self.jsonl_paths)
            self.jsonl_paths.extend(dataset.file_paths)
            for pack in dataset.packed_data:
                self.packs.append(
                    SFTData(
                        jsonl_idx=[base + dataset.data[i][0] for i in pack],
                        offsets=[dataset.data[i][1] for i in pack],
                        num_tokens=[dataset.data[i][2] for i in pack],
                    )
                )

        assert len(self.packs) > 0, "SFT datasets produced 0 packs; check file_paths / sample_ratios / max_length."
        self.steps_per_epoch = len(self.packs) // self.config.global_batch_size

        assert self.steps_per_epoch >= 1, (
            f"Not enough packs for a single optimizer step: {len(self.packs)} packs across dp_size={self.dp_size} "
            f"Add more data or lower global_batch_size={self.config.global_batch_size}."
        )

        total_tokens = sum(sum(p.num_tokens) for p in self.packs)
        self.logger.info(
            f"SFT data ready: {len(self.packs)} packs, {total_tokens} tokens, {len(self.jsonl_paths)} files, {num_skip} skipped samples, "
            f"dp_size={self.dp_size}, {self.steps_per_epoch} steps/epoch (global_batch_size={self.config.global_batch_size})."
        )
        if self.config.total_training_steps == -1:
            self.config.total_training_steps = self.steps_per_epoch
            self.logger.info(
                f"total_training_steps not set; defaulting to 1 epoch ({self.config.total_training_steps} steps)."
            )

    def epoch_packs(self, epoch) -> list[SFTData]:
        # Deterministic per-epoch shuffle so a resumed run replays the same order.
        packs = list(self.packs)
        random.Random(self.config.seed + epoch).shuffle(packs)
        return packs

    async def fit(self):
        epoch = -1
        for step in range(self.cur_step, self.config.total_training_steps):
            with MeasureTime("step"):
                self.cur_step = step

                # (Re)feed the engine at each epoch boundary. Feeding once per
                # epoch (not per step) avoids re-serializing the tokenizer-bearing
                # chat_template to the Ray actors and re-opening the jsonl files on
                # every step.
                cur_epoch = step // self.steps_per_epoch
                if cur_epoch != epoch:
                    epoch = cur_epoch
                    self.logger.info(f"*Loading SFT data for epoch {epoch} (step {step})")
                    with MeasureTime("set_data"):
                        await self.model_engine.set_sft_data(
                            self.epoch_packs(epoch),
                            self.jsonl_paths,
                            self.chat_template,
                        )

                with MeasureTime("train"):
                    train_log = await self.model_engine.step_sft(self.config.micro_batch_size)

                log_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_log.items())
                self.logger.info(f"*SFT step {step} (epoch {epoch}) logs: {log_str}")
                for k, v in train_log.items():
                    self.tb_writer.add_scalar(f"sft/{k}", v, step)
                self.tb_writer.add_scalar("sft/epoch", epoch, step)

                gc.collect()
                torch.cuda.empty_cache()

                if (step + 1) % self.config.checkpoint_interval == 0:
                    with MeasureTime("ckpt"):
                        await self.checkpoint(step)

            MeasureTime.saved_time["others"] = 2 * MeasureTime.saved_time["step"] - sum(
                list(MeasureTime.saved_time.values())
            )
            for key in MeasureTime.saved_time:
                self.logger.info(f"Time for {key}: {int(MeasureTime.saved_time[key])} seconds")
                self.tb_writer.add_scalar(f"Time/{key}", int(MeasureTime.saved_time[key]), step)
            MeasureTime.clear()
            self.logger.info("----------------------------\n\n")

    async def checkpoint(self, step):
        ckpt_folder = self.config.work_dir + "/checkpoints/"
        step_folder = ckpt_folder + f"/{step:06d}/"
        tmp_folder = os.path.join(self.config.work_dir, "tmp_ckpt")

        os.makedirs(ckpt_folder, exist_ok=True)
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder, exist_ok=True)

        await self.model_engine.checkpoint(tmp_folder)
        # Record the NEXT step to run so a resumed run continues past this one.
        self.global_step = step + 1
        async with aiofiles.open(os.path.join(tmp_folder, "algo.json"), "w") as f:
            await f.write(json.dumps({"global_step": self.global_step}))

        # A resumed run can re-reach the same step number (resume snaps to the
        # epoch start), so clear a pre-existing folder before moving into place.
        if os.path.exists(step_folder):
            shutil.rmtree(step_folder)
        shutil.move(tmp_folder, step_folder)
        self.logger.info(f"Checkpoint saved at step {step} to {step_folder}")

        existed = sorted(os.listdir(ckpt_folder))
        to_remove = existed[: -self.config.max_checkpoints]
        for rm in to_remove:
            rm_folder = os.path.join(ckpt_folder, rm)
            shutil.rmtree(rm_folder)
            self.logger.info(f"Removed old checkpoint: {rm_folder}")

    async def resume(self):
        def get_ckpt_path_from_work_dir(work_dir):
            ckpt_folder = os.path.join(work_dir, "checkpoints")
            if not os.path.exists(ckpt_folder):
                return None
            existed = sorted(os.listdir(ckpt_folder))
            if len(existed) == 0:
                return None
            return os.path.join(ckpt_folder, existed[-1])

        step_folder = os.environ.get("RESUME_PATH", get_ckpt_path_from_work_dir(self.config.work_dir))
        if step_folder is None:
            self.logger.warning("No checkpoints found, starting from scratch.")
            return False

        await self.model_engine.resume(step_folder)

        algo_path = os.path.join(step_folder, "algo.json")
        if os.path.exists(algo_path):
            async with aiofiles.open(algo_path, "r") as f:
                self.global_step = json.loads(await f.read()).get("global_step", 0)
        else:
            self.logger.warning(f"No SFT algo state at {algo_path}, resuming engine state only.")
        saved_step = self.global_step

        # The engine's set_sft_data DataLoader iterator can't be checkpointed, so
        # snap back to the start of the interrupted epoch and re-feed it. This may
        # re-train part of one epoch -- acceptable for SFT.
        self.cur_step = (saved_step // self.steps_per_epoch) * self.steps_per_epoch
        self.logger.info(
            f"Resumed from checkpoint {step_folder}: saved step {saved_step}, restarting epoch at step {self.cur_step}."
        )
        return True
