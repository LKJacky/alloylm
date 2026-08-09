"""Throughput benchmark for prefill and decode stages.

Measures tokens/s at the engine level (no HTTP overhead) using CUDA events for accurate GPU timing.  Run with:

python -m unittest -v tests.test_throughput
"""

import gc
import os
import random
import statistics
import time
import traceback
import unittest

import torch
from mmengine.dist import init_dist
from torch import distributed as dist
from transformers import AutoTokenizer

from alloylm.engine.infer_engine.sampler import BatchSampler
from alloylm.impl.engines.qwen.qwen2_modeling2 import (
    FSDPConfig,
    FSDPQwen2ForCausalLM,
)
from alloylm.test_utils import CudaAsyncTestCase

MODEL_PATH = "Qwen/Qwen3-0.6B"
MEMORY_USAGE = 0.4
MAX_LENGTH = 8 * 1024
MAX_PREFILL_LENGTH = 16 * 1024


class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_


def has_other_cuda_programs():
    if torch.cuda.device_count() > 0:
        try:
            # Get current GPU memory usage
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.get_device_properties(0).total_memory
            # Skip if more than 10% of GPU memory is already used
            if current_memory / max_memory > 0.1:
                return True
            else:
                return False
        except Exception:
            return True


def _cuda_sync_and_time(start_event, end_event):
    """Return elapsed milliseconds between two CUDA events."""
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestThroughput(CudaAsyncTestCase):
    """Benchmark prefill and decode throughput at the engine level."""

    model: FSDPQwen2ForCausalLM
    tokenizer: AutoTokenizer
    sampler: BatchSampler

    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", str(random.randint(20000, 30000)))
            init_dist("pytorch")
        with LoadWoInit():
            model = FSDPQwen2ForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype="bfloat16",
                trust_remote_code=True,
                fsdp_config=FSDPConfig(
                    train_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["fsdp", "sp"]),
                    infer_mesh=dict(device_type="cuda", mesh_shape=(1, 1), mesh_dim_names=["dp", "tp"]),
                    lm_head_dtype=torch.bfloat16,
                ),
            ).cuda()
        gc.collect()
        torch.cuda.empty_cache()
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        cls.model = model
        cls.cache = model.create_cache(memory_usage=MEMORY_USAGE)
        cls.sampler = BatchSampler(real_vocab_size=model.get_real_vocab_size(cls.tokenizer))
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory allocated after engine init: {torch.cuda.memory_allocated() / (1024 * 1024 * 1024):.2f} GB")
        cls.cache.prepare()
        cls.model.infer_shard(MAX_PREFILL_LENGTH)

    @classmethod
    def tearDownClass(cls):
        cls.cache.close()
        cls.model.train_shard()
        cls.model._rank0_model = None
        cls.model.to_empty(device="cpu")
        del cls.cache
        del cls.model
        del cls.tokenizer
        del cls.sampler
        gc.collect()
        torch.cuda.empty_cache()
        used_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        if dist.is_initialized():
            dist.destroy_process_group()
        assert used_memory < 0.5, f"Memory leak detected: {used_memory:.2f} GB used after test"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_sessions(self, batch_size, seq_len):
        """Create *batch_size* sessions each with *seq_len* random tokens."""
        sessions = []
        for i in range(batch_size):
            s = self.cache.create_device_session(i + 1000)
            tokens = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
            s.append_input_tokens(tokens)
            if not self.cache.allocate_cache(s):
                self.cache.reset(sessions)
                raise RuntimeError(f"OOM allocating cache for session {i}")
            sessions.append(s)
        return sessions

    def _cleanup_sessions(self, sessions):
        if sessions:
            self.cache.reset(sessions)

    # ------------------------------------------------------------------
    # Prefill benchmark
    # ------------------------------------------------------------------

    def _bench_prefill(self, batch_size, seq_len, warmup=2, repeats=5):
        """Return prefill throughput in tokens/s."""
        total_tokens = batch_size * seq_len
        for _ in range(warmup):
            sessions = self._make_sessions(batch_size, seq_len)
            try:
                self.model.prefill(sessions, self.cache)
            finally:
                self._cleanup_sessions(sessions)

        elapsed_ms_list = []
        for _ in range(repeats):
            sessions = self._make_sessions(batch_size, seq_len)
            try:
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=0, warmup=2, active=5, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        f"./work_dirs/tests/profile/profile_prefill_{batch_size}_{1}_{5}"
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    self.model.prefill(sessions, self.cache)
                    prof.step()
                end_ev.record()
                elapsed_ms = _cuda_sync_and_time(start_ev, end_ev)
                elapsed_ms_list.append(elapsed_ms)
            finally:
                self._cleanup_sessions(sessions)

        median_ms = sorted(elapsed_ms_list)[len(elapsed_ms_list) // 2]
        throughput = total_tokens / (median_ms / 1000.0)
        return throughput, median_ms

    # ------------------------------------------------------------------
    # Decode benchmark
    # ------------------------------------------------------------------

    def _bench_decode(self, batch_size, context_len=128, decode_steps=64, warmup=2):
        """Return decode throughput in tokens/s."""
        sessions = self._make_sessions(batch_size, context_len)
        try:
            logits = torch.cat([self.model.prefill([session], self.cache) for session in sessions], dim=0)

            tokens = logits.argmax(dim=-1)
            for i, s in enumerate(sessions):
                s.append_input_tokens(tokens[i : i + 1].unsqueeze(0))
                self.cache.allocate_cache(s)

            def append_next_tokens(next_tokens):
                next_tokens = next_tokens.cpu().flatten().tolist()
                for i, s in enumerate(sessions):
                    s.step()
                    s.append_input_tokens(next_tokens[i : i + 1])
                    self.cache.allocate_cache(s)

            # warmup (triggers CUDA graph capture)
            for _ in range(warmup):
                decode_logits = self.model.decode(sessions, self.cache)
                append_next_tokens(decode_logits[:batch_size].argmax(dim=-1))
            torch.cuda.empty_cache()
            # timed

            def step():
                decode_logits = self.model.decode(sessions, self.cache)
                append_next_tokens(decode_logits[:batch_size].argmax(dim=-1))

            if os.environ.get("ALLOYLM_PROFILE_THROUGHPUT") == "1":
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        f"./work_dirs/tests/profile/profile_decode_{batch_size}_{context_len}_{5}"
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    for _ in range(5):
                        prof.step()
                        step()

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(decode_steps):
                step()
            end_ev.record()

            elapsed_ms = _cuda_sync_and_time(start_ev, end_ev)
            total_tokens = batch_size * decode_steps
            throughput = total_tokens / (elapsed_ms / 1000.0)
            return throughput, elapsed_ms, decode_steps
        finally:
            self._cleanup_sessions(sessions)

    # ------------------------------------------------------------------
    # Test methods (a_ prefix ensures prefill runs before decode)
    # ------------------------------------------------------------------

    def test_a_prefill_throughput(self):
        """Measure prefill throughput across different configurations."""
        configs = [
            (32, 128),
            (16, 512),
            (8, 1024),
            (4, 2048),
            (2, 4096),
            (1, 8192),
        ]
        print("\n" + "=" * 70)
        print(f"{'PREFILL THROUGHPUT':^70}")
        print("=" * 70)
        print(f"{'Batch':>6} {'SeqLen':>8} {'Tokens':>8} {'Median ms':>12} {'Tokens/s':>14}")
        print("-" * 70)

        for batch_size, seq_len in configs:
            total = batch_size * seq_len
            if total > MAX_PREFILL_LENGTH:
                continue
            try:
                throughput, median_ms = self._bench_prefill(batch_size, seq_len)
                print(f"{batch_size:>6} {seq_len:>8} {total:>8} {median_ms:>12.2f} {throughput:>14,.0f}")
            except Exception as e:
                print(f"{batch_size:>6} {seq_len:>8} {total:>8} {'SKIP':>12} {str(e)[:30]}")
        print("=" * 70)

    @unittest.skipUnless(not has_other_cuda_programs(), "CUDA required")
    @torch.inference_mode()
    def test_b_decode_throughput(self):
        """Measure decode throughput across different batch sizes."""
        configs = [
            (1, 1024, 64),
            (8, 1024, 64),
            (16, 1024, 64),
            (32, 1024, 64),
        ]
        print("\n" + "=" * 70)
        print(f"{'DECODE THROUGHPUT':^70}")
        print("=" * 70)
        print(f"{'Batch':>6} {'Context':>8} {'Steps':>6} {'Total ms':>12} {'Tokens/s':>14}")
        print("-" * 70)

        for batch_size, context_len, decode_steps in configs:
            try:
                if batch_size <= self.cache.max_infer_batch_size:
                    throughput, elapsed_ms, steps = self._bench_decode(batch_size, context_len, decode_steps)
                    print(f"{batch_size:>6} {context_len:>8} {steps:>6} {elapsed_ms:>12.2f} {throughput:>14,.0f}")
            except Exception as e:
                print(f"{batch_size:>6} {context_len:>8} {decode_steps:>6} {'SKIP':>12} {str(e)[:30]}")
                if not str(e).startswith("CUDA out of memory"):
                    traceback.print_exc()
                    break
        print("=" * 70)


@unittest.skipUnless(False, "SGLANG benchmark disabled by default")
class TestSGLangThroughput(CudaAsyncTestCase):
    """Benchmark sglang prefill and decode by initializing the model runner
    directly (like ``a.py``), bypassing the HTTP server and the full inference
    engine system."""

    @staticmethod
    def _load_sglang_runner():
        """Initialize a sglang ModelRunner directly and return the bench runner
        plus the synthetic-input builder."""

        # Newer sglang routes extend/prefill through fixed-size eager static
        # buffers whose token ceiling can be smaller than our benchmark batch.
        # This env var skips that copy and runs the batch as-is.
        os.environ.setdefault("SGLANG_EAGER_INPUT_NO_COPY", "1")

        from sglang.bench_one_batch import (
            load_model,
            prepare_synthetic_inputs_for_latency_test,
        )
        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(
            model_path=MODEL_PATH,
            tokenizer_path=MODEL_PATH,
            trust_remote_code=True,
            mem_fraction_static=0.5,
            tp_size=1,
            context_length=MAX_LENGTH,
            disable_radix_cache=True,
            log_level="error",
        )
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        model_runner, _tokenizer = load_model(server_args, port_args, gpu_id=0, tp_rank=0)
        return model_runner, prepare_synthetic_inputs_for_latency_test

    @unittest.skipUnless(not has_other_cuda_programs(), "CUDA required")
    @torch.inference_mode()
    def test_sglang_prefill(self):
        """Measure sglang prefill throughput across batch/seq configs."""
        model_runner, prepare_inputs = self._load_sglang_runner()

        prefill_configs = [
            (32, 128),
            (16, 512),
            (8, 1024),
            (4, 2048),
            (2, 4096),
            (1, 8192),
        ]
        print("\n" + "=" * 70)
        print(f"{'SGLANG PREFILL THROUGHPUT':^70}")
        print("=" * 70)
        print(f"{'Batch':>6} {'InLen':>8} {'Tokens':>8} {'Latency ms':>12} {'Tokens/s':>14}")
        print("-" * 70)

        for batch_size, input_len in prefill_configs:
            total_tokens = batch_size * input_len
            if total_tokens > MAX_PREFILL_LENGTH:
                continue
            try:
                # Warm up
                for _ in range(2):
                    reqs = prepare_inputs(batch_size, input_len)
                    model_runner.clear()
                    model_runner.synchronize()
                    model_runner.extend(reqs)
                    model_runner.synchronize()

                # Measure
                elapsed_s = []
                for _ in range(5):
                    reqs = prepare_inputs(batch_size, input_len)
                    model_runner.clear()
                    model_runner.synchronize()
                    tic = time.perf_counter()
                    model_runner.extend(reqs)
                    model_runner.synchronize()
                    elapsed_s.append(time.perf_counter() - tic)

                median_ms = statistics.median(elapsed_s) * 1000.0
                throughput = total_tokens / (median_ms / 1000.0)
                print(f"{batch_size:>6} {input_len:>8} {total_tokens:>8} {median_ms:>12.2f} {throughput:>14,.0f}")
            except Exception as e:  # noqa: BLE001
                print(f"{batch_size:>6} {input_len:>8} {total_tokens:>8} {'SKIP':>12} {str(e)[:30]}")
                if not str(e).startswith("CUDA out of memory"):
                    traceback.print_exc()
        print("=" * 70)

    @unittest.skipUnless(not has_other_cuda_programs(), "CUDA required")
    @torch.inference_mode()
    def test_sglang_decoding(self):
        """Measure sglang decode throughput (tokens/s after prefill)."""
        model_runner, prepare_inputs = self._load_sglang_runner()

        decode_configs = [
            (1, 1024, 64),
            (8, 1024, 64),
            (16, 1024, 64),
            (32, 1024, 64),
        ]
        print("\n" + "=" * 70)
        print(f"{'SGLANG DECODE THROUGHPUT':^70}")
        print("=" * 70)
        print(f"{'Batch':>6} {'CtxLen':>8} {'Steps':>6} {'Total ms':>12} {'Tokens/s':>14}")
        print("-" * 70)

        for batch_size, context_len, decode_steps in decode_configs:
            try:
                # Warm up: full prefill + decode loop
                reqs = prepare_inputs(batch_size, context_len)
                model_runner.clear()
                model_runner.synchronize()
                next_token_ids, _logits, batch = model_runner.extend(reqs)
                for _ in range(decode_steps):
                    next_token_ids, _ = model_runner.decode(next_token_ids, batch)
                model_runner.synchronize()

                # Measure: prefill (untimed) then time the decode steps only
                reqs = prepare_inputs(batch_size, context_len)
                model_runner.clear()
                model_runner.synchronize()
                next_token_ids, _logits, batch = model_runner.extend(reqs)
                model_runner.synchronize()

                tic = time.perf_counter()
                for _ in range(decode_steps):
                    next_token_ids, _ = model_runner.decode(next_token_ids, batch)
                model_runner.synchronize()
                elapsed_ms = (time.perf_counter() - tic) * 1000.0

                total_tokens = batch_size * decode_steps
                throughput = total_tokens / (elapsed_ms / 1000.0)
                print(f"{batch_size:>6} {context_len:>8} {decode_steps:>6} {elapsed_ms:>12.2f} {throughput:>14,.0f}")
            except Exception as e:  # noqa: BLE001
                print(f"{batch_size:>6} {context_len:>8} {decode_steps:>6} {'SKIP':>12} {str(e)[:30]}")
                if not str(e).startswith("CUDA out of memory"):
                    traceback.print_exc()
        print("=" * 70)
