import asyncio
import gc
import json
import math
import os
import shutil
import tempfile
import unittest

import torch
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.algorithm.sft.dataset import SFTPackDatasetConfig
from alloylm.algorithm.sft.sft_algo import SFTAlgorithmConfig, SFTTrainer
from alloylm.engine.infer_engine.engine import InferEngineConfig
from alloylm.engine.model import AlloyLMModelConfig
from alloylm.engine.spmd import init_dist
from alloylm.engine.train_engine.dataset import SFTData, SFTDataset, sft_collate_fn
from alloylm.engine.train_engine.train_engine import TrainEngine, TrainEngineConfig
from alloylm.engine.train_engine.train_infer_engine import TrainInferEngineConfig
from alloylm.engine.train_engine.utils import FSDPConfig
from alloylm.impl.engines.qwen.qwen2_modeling2 import FSDPQwen2ForCausalLM

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_SFT_STEPS = 12
# Real multi-turn conversations (system/user/assistant) shared with the dataset tests.
MSGS_JSONL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource", "msgs.jsonl"))


class FakeTokenizer:
    """Deterministic tokenizer stub: each character maps to its code point."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def chat_template(converted_messages, add_generation_prompt=False):
    text = ""
    for msg in converted_messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


class SFTDatasetTest(unittest.TestCase):
    def setUp(self):
        self.conversations = [
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            [
                {"role": "user", "content": "second turn"},
                {"role": "assistant", "content": "ok"},
            ],
        ]
        self.tmpdir = tempfile.mkdtemp()
        self.jsonl_path = os.path.join(self.tmpdir, "sft.jsonl")
        self.offsets = self._write_jsonl(self.jsonl_path, self.conversations)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _write_jsonl(path, conversations):
        offsets = []
        with open(path, "w", encoding="utf-8") as f:
            offset = 0
            for msgs in conversations:
                line = json.dumps({"messages": msgs}, ensure_ascii=False) + "\n"
                offsets.append(offset)
                f.write(line)
                offset += len(line.encode("utf-8"))
        return offsets

    def _make_dataset(self, sft_data, conversation):
        return SFTDataset(
            sft_data,
            [self.jsonl_path],
            FakeTokenizer(),
            chat_template,
        )

    def test_len(self):
        sft_data = [
            SFTData(jsonl_idx=[0], offsets=[self.offsets[0]], num_tokens=[74]),
            SFTData(jsonl_idx=[0], offsets=[self.offsets[1]], num_tokens=[74]),
        ]
        dataset = self._make_dataset(sft_data, self.conversations[0])
        self.assertEqual(len(dataset), 2)

    def test_getitem_masks_non_assistant_tokens(self):
        user_text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        assistant_text = "hi there<|im_end|>\n"
        num_tokens = len(user_text) + len(assistant_text)

        sft_data = [SFTData(jsonl_idx=[0], offsets=[self.offsets[0]], num_tokens=[num_tokens])]
        dataset = self._make_dataset(sft_data, self.conversations[0])

        sample = dataset[0]

        self.assertEqual(sample["input_ids"], [ord(c) for c in user_text + assistant_text])
        self.assertEqual(
            sample["labels"],
            [-100] * len(user_text) + [ord(c) for c in assistant_text],
        )
        self.assertEqual(sample["seq_lens"], [num_tokens])

    def test_getitem_reads_correct_offset(self):
        user_text = "<|im_start|>user\nsecond turn<|im_end|>\n<|im_start|>assistant\n"
        assistant_text = "ok<|im_end|>\n"
        num_tokens = len(user_text) + len(assistant_text)

        sft_data = [SFTData(jsonl_idx=[0], offsets=[self.offsets[1]], num_tokens=[num_tokens])]
        dataset = self._make_dataset(sft_data, self.conversations[1])

        sample = dataset[0]

        self.assertEqual(sample["input_ids"], [ord(c) for c in user_text + assistant_text])
        self.assertEqual(sample["seq_lens"], [num_tokens])


class SftCollateFnTest(unittest.TestCase):
    """Tests the SFT collation used by ``TrainEngine.set_sft_data``."""

    def test_packs_samples_into_single_sequence(self):
        batch = [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, 3], "seq_lens": [3]},
            {"input_ids": [4, 5], "labels": [4, 5], "seq_lens": [2]},
        ]

        out = sft_collate_fn(batch)

        self.assertTrue(torch.equal(out["input_ids"], torch.tensor([[1, 2, 3, 4, 5]])))
        self.assertTrue(torch.equal(out["labels"], torch.tensor([[-100, -100, 3, 4, 5]])))
        self.assertTrue(torch.equal(out["seq_lens"], torch.tensor([3, 2])))


# ---------------------------------------------------------------------------
# End-to-end SFT training (real model + tokenizer + TrainEngine on GPU)
# ---------------------------------------------------------------------------


class QwenChatTemplate:
    """Callable wrapping ``tokenizer.apply_chat_template`` to the signature
    ``SFTDataset`` expects: ``(messages, add_generation_prompt=False) -> str``.

    A plain instance (not a function) is used on purpose: assigning a bare
    function to a ``unittest`` class attribute turns ``self.chat_template`` into a
    bound method whose ``self`` would leak into the call and shift the arguments.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, messages, add_generation_prompt=False):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )


def _sft_token_count(tokenizer, chat_template, messages):
    """Mirror ``SFTDataset.distangle_train_or_not_train`` + ``tokenize`` to
    compute the token count a given conversation will produce (must match
    ``SFTData.num_tokens``)."""
    converted = []
    pre_text = ""
    for i, msg in enumerate(messages):
        add_gen = (i + 1 < len(messages)) and messages[i + 1]["role"] == "assistant"
        text = chat_template(messages[: i + 1], add_generation_prompt=add_gen)
        has_loss = msg["role"] == "assistant"
        converted.append((text[len(pre_text) :], has_loss))
        pre_text = text
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t, _ in converted)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class SFTTrainEndToEndTest(unittest.TestCase):
    """Runs a full SFT training loop: SFTDataset -> sft_collate_fn -> forward ->
    cross-entropy -> backward -> optimizer step, on a real model."""

    @classmethod
    def setUpClass(cls):
        init_dist()

        fsdp_config = FSDPConfig(
            train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
            infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
            shard_dtype=torch.bfloat16,
            lm_head_dtype=torch.bfloat16,
        )
        cls.model = FSDPQwen2ForCausalLM.from_pretrained(MODEL_PATH, fsdp_config=fsdp_config)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)

        cls.tmpdir = tempfile.mkdtemp()
        cls.engine = TrainEngine(
            cls.model,
            cls.tokenizer,
            TrainEngineConfig(
                work_dir=cls.tmpdir,
                lr=1e-4,
                scheduler_type="constant",
                total_training_steps=50,
            ),
        )

        # One fixed conversation, repeated so the model can overfit it and the
        # loss reliably decreases across steps.
        cls.conversation = [
            {"role": "user", "content": "Explain the water cycle."},
            {
                "role": "assistant",
                "content": "The water cycle describes how water evaporates, condenses into clouds, and falls back as precipitation.",
            },
        ]
        cls.jsonl_path = os.path.join(cls.tmpdir, "sft.jsonl")
        with open(cls.jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"messages": cls.conversation}, ensure_ascii=False) + "\n")

        cls.chat_template = QwenChatTemplate(cls.tokenizer)
        num_tokens = _sft_token_count(cls.tokenizer, cls.chat_template, cls.conversation)
        cls.sft_data = [SFTData(jsonl_idx=[0], offsets=[0], num_tokens=[num_tokens]) for _ in range(NUM_SFT_STEPS)]

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        del cls.engine
        del cls.model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_sft_training_reduces_loss(self):
        self.engine.set_sft_data(self.sft_data, [self.jsonl_path], self.chat_template)

        losses = []
        for _ in range(NUM_SFT_STEPS):
            result = self.engine.step_sft(num_micro_steps=1)
            self.assertTrue(math.isfinite(result["loss"]), f"non-finite loss: {result['loss']}")
            self.assertGreater(result["num_tokens"], 0)
            losses.append(result["loss"])

        self.assertLess(losses[-1], losses[0])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class SFTTrainerEndToEndTest(unittest.TestCase):
    """End-to-end SFT through the full orchestration stack -- ``SFTTrainer`` +
    ``SpmdTrainInferEngine`` -- driven exactly as ``alloylm/algorithm/sft/run.py``
    does (``lazy_init`` then ``fit``).

    Distinct from ``SFTTrainEndToEndTest`` above, which drives the raw in-process
    ``TrainEngine``. This exercises the algorithm layer end to end: dataset build /
    packing from a real jsonl, per-epoch ``set_sft_data``, the SPMD ``step_sft``
    path, and checkpoint writing/pruning.

    With ``num_workers=1`` the SPMD engine runs in-process, so this needs a GPU.
    The data (``tests/resource/msgs.jsonl``, 3 conversations) packs into 2 packs at
    ``max_length=1024`` -> ``steps_per_epoch=2``; ``total_training_steps=4`` is two
    epochs, and a checkpoint every 2 steps lands at steps 1 and 3.
    """

    MAX_LENGTH = 1024
    TOTAL_STEPS = 4
    CKPT_INTERVAL = 2
    MAX_CKPTS = 2

    def setUp(self):
        self.work_dir = "work_dirs/debug"
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)
        # A leaked RESUME_PATH from another run would derail auto_resume.
        os.environ.pop("RESUME_PATH", None)

    def tearDown(self):
        os.environ.pop("RESUME_PATH", None)
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _make_trainer(self):
        fsdp_config = FSDPConfig(
            train_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "sp"], "device_type": "cuda"},
            infer_mesh={"mesh_shape": (1, 1), "mesh_dim_names": ["dp", "tp"], "device_type": "cuda"},
            shard_dtype=torch.bfloat16,
            lm_head_dtype=torch.bfloat16,
        )
        config = SFTAlgorithmConfig(
            llm_config=AlloyLMModelConfig(path=MODEL_PATH, model_cls=FSDPQwen2ForCausalLM, fsdp_config=fsdp_config),
            engine_config=TrainInferEngineConfig(
                train_config=TrainEngineConfig(
                    max_length=self.MAX_LENGTH,
                    work_dir=self.work_dir,
                    num_workers=1,
                    sp_size=1,
                    lr=1e-4,
                    scheduler_type="constant",
                    total_training_steps=self.TOTAL_STEPS,
                ),
                infer_config=InferEngineConfig(model_name="ALLOYLM"),
            ),
            datasets=[SFTPackDatasetConfig(file_paths=[MSGS_JSONL], sample_ratios=[1.0], max_length=self.MAX_LENGTH)],
            max_length=self.MAX_LENGTH,
            global_batch_size=1,
            total_training_steps=self.TOTAL_STEPS,
            checkpoint_interval=self.CKPT_INTERVAL,
            max_checkpoints=self.MAX_CKPTS,
            work_dir=self.work_dir,
        )
        return SFTTrainer(config)

    def test_trains_and_checkpoints(self):
        trainer = self._make_trainer()

        # Capture each step's log. Wrapping the SPMD engine's step_sft here also
        # guards its list->dict unwrapping (return results[0]), which the raw
        # TrainEngine path in SFTTrainEndToEndTest does not exercise.
        step_logs = []
        engine_step_sft = trainer.model_engine.step_sft

        async def recording_step_sft(num_micro_steps):
            result = await engine_step_sft(num_micro_steps)
            step_logs.append(result)
            return result

        trainer.model_engine.step_sft = recording_step_sft

        async def run(t):
            await t.lazy_init()
            await t.fit()

        try:
            asyncio.run(run(trainer))

            # 3 conversations pack into 2 packs at max_length=1024, dp_size=1.
            self.assertEqual(trainer.steps_per_epoch, 2)

            # One log per optimizer step, each a finite loss over real tokens.
            self.assertEqual(len(step_logs), self.TOTAL_STEPS)
            for log in step_logs:
                self.assertIsInstance(log, dict)
                self.assertTrue(math.isfinite(log["loss"]), f"non-finite loss: {log['loss']}")
                self.assertGreater(log["num_tokens"], 0)

            # checkpoint_interval=2 -> checkpoints at steps 1 and 3, both kept.
            ckpt_root = os.path.join(self.work_dir, "checkpoints")
            self.assertEqual(sorted(os.listdir(ckpt_root)), ["000001", "000003"])
            # algo.json records the NEXT step to run, so a resume continues past it.
            for folder, expected_next in (("000001", 2), ("000003", 4)):
                with open(os.path.join(ckpt_root, folder, "algo.json")) as f:
                    self.assertEqual(json.load(f)["global_step"], expected_next)
                # the model checkpoint lands alongside it
                self.assertTrue(os.path.isdir(os.path.join(ckpt_root, folder, "hf")))
        finally:
            del trainer
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
