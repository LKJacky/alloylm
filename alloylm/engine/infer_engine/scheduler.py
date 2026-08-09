import asyncio
import time
import traceback
from asyncio import Queue

import torch
from sortedcontainers import SortedList

from alloylm.engine.model import AlloyLMModel, Cache, DeviceSession

from .sampler import BatchSampler
from .utils import GeneConfig, get_logger

# action


class TaskItem:
    priority = -1

    def __init__(self, session_id):
        self.session_id = session_id
        self.finished_event = asyncio.Event()

    def __lt__(self, other: "TaskItem"):
        return self.priority < other.priority


class InferItem(TaskItem):
    priority = 1

    def __init__(self, session_id, input_ids, gene_config: GeneConfig):
        super().__init__(session_id)
        self.input_ids: list = input_ids
        self.num_init_tokens = 0
        self.gene_config = gene_config
        self.device_session: DeviceSession = None

    def set_device_session(self, device_session: DeviceSession):
        self.device_session = device_session
        self.num_init_tokens = self.device_session.total_num_tokens()
        self.device_session.append_input_tokens(self.input_ids)

    def should_stop(self):
        if len(self.device_session.tokens) > 0 and self.device_session.tokens[-1] in self.gene_config.stop_token:
            return "stop"
        elif (self.device_session.total_num_tokens() - self.num_init_tokens) >= self.gene_config.total_max_length:
            return "length"
        elif self.device_session.entropy[-1] >= self.gene_config.max_entropy:
            return "entropy"
        else:
            return "none"

    def get_finish_result(self):
        if self.device_session is None:
            return {
                "tokens": [],
                "log_prob": [],
                "entropy": [],
                "finish_reason": "length",
                "usage": {
                    "history_tokens": self.num_init_tokens,
                    "input_tokens": len(self.input_ids),
                    "output_tokens": 0,
                },
            }
        else:
            num_prompt_tokens = self.num_init_tokens + len(self.input_ids)
            tokens = (self.device_session.forwarded_tokens + self.device_session.tokens)[num_prompt_tokens:]
            log_prob = (self.device_session.forwarded_log_probs + self.device_session.log_probs)[num_prompt_tokens:]
            entropy = (self.device_session.forwarded_entropy + self.device_session.entropy)[num_prompt_tokens:]
            return {
                "tokens": tokens,
                "log_prob": log_prob,
                "entropy": entropy,
                "finish_reason": self.should_stop(),
                "usage": {
                    "history_tokens": self.num_init_tokens,
                    "input_tokens": len(self.input_ids),
                    "output_tokens": len(tokens),
                },
            }


class ResetItem(TaskItem):
    priority = 0


class ReleaseItem(TaskItem):
    priority = -1


class ResultItem(TaskItem):
    def __init__(self, session_id, result):
        super().__init__(session_id)
        self.result = result


# scheduler


class RleasedSession:
    def __init__(
        self,
        freee_queue: dict[InferItem],
        prefill_queue: SortedList,
        decode_queue: list[InferItem],
        cache: Cache,
    ):
        self.free_queue = freee_queue
        self.prefill_queue = prefill_queue
        self.decode_queue = decode_queue
        self.cache = cache

        self._released_prefill_sessions = []

    def __enter__(self):
        return iter(self._iterative_release())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prefill_queue is not None:
            self.prefill_queue.update(self._released_prefill_sessions)
            self._released_prefill_sessions = []

    def _iterative_release(self):
        if self.free_queue is not None:
            for session in self.free_queue.values():
                if session.on_device():
                    self.cache.reset([session])
                    yield session, 1

        if self.prefill_queue is not None:
            self._released_prefill_sessions = []
            while len(self.prefill_queue) > 0:
                session = self.prefill_queue.pop()
                self._released_prefill_sessions.append(session)
                if session.device_session.on_device():
                    self.cache.reset([session.device_session])
                    yield session, 1
        if self.prefill_queue is not None:
            self.prefill_queue.update(self._released_prefill_sessions)
            self._released_prefill_sessions = []

        if self.decode_queue is not None:
            while len(self.decode_queue) > 0:
                session = self.decode_queue.pop()
                self.prefill_queue.add(session)
                if session.device_session.on_device():
                    self.cache.reset([session.device_session])
                    yield session, 1


class SchedulerServer:
    def __init__(
        self,
        model: AlloyLMModel,
        cache: Cache,
        max_prefill_length: int,
        task_queue: Queue[GeneConfig] = Queue(),
        real_vocab_size=-1,
    ):
        self.model = model
        self.cache = cache
        self.max_prefill_length = max_prefill_length
        self.max_infer_length = self.cache.max_infer_length

        self.task = None

        self.wait_queue = task_queue

        self.prefill_queue: SortedList = SortedList(
            key=lambda x: (-self.cache.cache_usage([x.device_session]), x.device_session.create_time)
        )
        self.decode_queue: list[InferItem] = []

        self.batch_sampler = BatchSampler(real_vocab_size=real_vocab_size)
        self.device_sessions: dict[int, DeviceSession] = {}

        self.fowarded_sample_config = set()

        self.top_cache_usage_for_decode = 0.9
        self.bottom_cache_usage_for_decode = 0.7

    def apply_device_session(self, infer_item: InferItem):
        assert infer_item.device_session is None, "Device session already set for this infer item"
        if infer_item.session_id not in self.device_sessions:
            session = self.cache.create_device_session(infer_item.session_id)
            self.device_sessions[infer_item.session_id] = session
        session = self.device_sessions.pop(infer_item.session_id)
        infer_item.set_device_session(session)

    # schedule

    @torch.inference_mode()
    async def update_queue(self):
        while (len(self.prefill_queue) + len(self.decode_queue) == 0) or (not self.wait_queue.empty()):
            session: TaskItem = await self.wait_queue.get()
            if isinstance(session, InferItem):
                if session.gene_config.total_max_length >= self.max_infer_length:
                    get_logger().warning(
                        f"Requested max length {session.gene_config.total_max_length} exceeds engine limit {self.max_infer_length}, truncating."
                    )
                    await self.finish_item(session, session, reason="length")
                else:
                    session.gene_config.total_max_length = min(
                        session.gene_config.total_max_length, self.max_infer_length
                    )
                    self.apply_device_session(session)
                    self.prefill_queue.add(session)
                    sample_arg = (
                        session.gene_config.temperature,
                        session.gene_config.top_k,
                        session.gene_config.top_p,
                    )
                    if sample_arg not in self.fowarded_sample_config:
                        self.fowarded_sample_config.add(sample_arg)
                        get_logger().debug(
                            f"Request: temperature: {session.gene_config.temperature}, top_k: {session.gene_config.top_k}, top_p: {session.gene_config.top_p}, entropy: {session.gene_config.max_entropy}"
                        )
            elif isinstance(session, ResetItem):
                exist_session = self.device_sessions.pop(session.session_id, None)
                if exist_session is not None:
                    self.cache.reset([exist_session])
                await self.finish_item(session, None)
            elif isinstance(session, ReleaseItem):
                num = (
                    len(self.decode_queue)
                    + len(self.prefill_queue)
                    + len(self.device_sessions)
                    + self.wait_queue.qsize()
                )
                while len(self.decode_queue) > 0:
                    seq = self.decode_queue.pop()
                    await self.finish_item(seq, seq, reason="abort")
                while len(self.prefill_queue) > 0:
                    seq = self.prefill_queue.pop()
                    await self.finish_item(seq, seq, reason="abort")
                while len(self.device_sessions) > 0:
                    _, _session = self.device_sessions.popitem()
                    self.cache.reset([_session])
                while self.wait_queue.empty() is False:
                    _session = await self.wait_queue.get()
                    await self.finish_item(_session, _session, reason="abort")
                if num != 0:
                    get_logger().info(f"Release all {num} tasks in scheduler")
                await self.finish_item(session, None)
            else:
                raise ValueError(f"Unknown item type in wait_queue: {type(session)}")

    @torch.inference_mode()
    def update_with_logits(self, sessions: list[InferItem], batch_logits):
        temperature = torch.tensor([seq.gene_config.temperature for seq in sessions], device=batch_logits.device)
        top_k = torch.tensor(
            [seq.gene_config.top_k for seq in sessions], dtype=torch.int32, device=batch_logits.device
        )
        top_p = torch.tensor([seq.gene_config.top_p for seq in sessions], device=batch_logits.device)

        tokens, logprobs, entropys = self.batch_sampler.batch_sample(batch_logits, temperature, top_k, top_p)
        tokens, logprobs, entropys = tokens.tolist(), logprobs.tolist(), entropys.tolist()
        for seq, token, logprob, entropy in zip(sessions, tokens, logprobs, entropys):
            seq.device_session.append_input_tokens([token], [logprob], [entropy])

    # inference

    @torch.inference_mode()
    def _iter_prefill(self):
        with RleasedSession(self.device_sessions, self.prefill_queue, None, self.cache) as release_iter:
            while (
                len(self.prefill_queue) > 0
                and self.cache.cache_usage([x.device_session for x in self.decode_queue])
                < self.top_cache_usage_for_decode
            ):
                chunk_sessions = []
                cur_length = 0
                whether_decode_token = []
                num_release = 0

                session = None
                cached_for_session = None
                enough_memory = True

                while (
                    len(self.prefill_queue) > 0
                    and self.cache.cache_usage([x.device_session for x in chunk_sessions + self.decode_queue])
                    < self.top_cache_usage_for_decode
                ) and (
                    cur_length < self.max_prefill_length
                    and cur_length != self.max_prefill_length - 1  # avoid 1 token prefill
                    and len(chunk_sessions) < self.cache.max_infer_batch_size
                    and enough_memory
                ):
                    assert cached_for_session is None, (
                        "cached_for_session should be None at the beginning of each loop"
                    )
                    session: InferItem = self.prefill_queue.pop(0)
                    if len(session.device_session.tokens) == 1:
                        self.decode_queue.append(session)
                    else:
                        # truncate if needed
                        max_forward_tokens = self.max_prefill_length - cur_length
                        assert max_forward_tokens > 1, "1 token should not be prefilled"
                        if len(session.device_session.tokens) > max_forward_tokens:
                            cached_for_session = session.device_session.truncate_tokens(max_forward_tokens)
                        # allocate cache
                        if self.cache.allocate_cache(session.device_session):
                            enough_memory = True
                        else:
                            enough_memory = False
                            if len(self.decode_queue) + len(chunk_sessions) == 0:
                                for release_session, _num_release in release_iter:
                                    num_release += _num_release
                                    if self.cache.allocate_cache(session.device_session):
                                        enough_memory = True
                                        break
                        # update queues
                        if enough_memory:
                            chunk_sessions.append(session)
                            cur_length += len(session.device_session.tokens)
                            whether_decode_token.append(cached_for_session is None)
                if len(chunk_sessions) > 0:
                    yield chunk_sessions, whether_decode_token, num_release
                # resume
                if session is not None:
                    if cached_for_session is not None:
                        session.device_session.append_input_tokens(*cached_for_session)
                    if enough_memory is False or cached_for_session is not None:
                        self.prefill_queue.add(session)
                if len(chunk_sessions) == 0 and enough_memory is False:
                    break

    async def prefill(self):
        num_prefill_tokens = 0
        num_prefill_sessions = 0
        total_num_release = 0

        for part_sessions, whether_decode, num_release in self._iter_prefill():
            # collection info
            num_prefill_tokens += sum(len(s.device_session.tokens) for s in part_sessions)
            num_prefill_sessions += len(part_sessions) if whether_decode[-1] else len(part_sessions) - 1
            total_num_release += num_release
            # forward
            logits = self.model.prefill([s.device_session for s in part_sessions], self.cache)
            if any(whether_decode):
                self.update_with_logits(
                    part_sessions if whether_decode[-1] else part_sessions[:-1],
                    logits if whether_decode[-1] else logits[:-1],
                )
                self.decode_queue.extend(part_sessions if whether_decode[-1] else part_sessions[:-1])
        if total_num_release > 0:
            get_logger().debug(f"Reset {total_num_release} for prefill")
        return num_prefill_sessions, num_prefill_tokens

    async def decode(self):
        decode_sessions = []
        num_reset_sessions = 0
        with RleasedSession(self.device_sessions, self.prefill_queue, self.decode_queue, self.cache) as release_iter:
            while len(self.decode_queue) > 0 and len(decode_sessions) < self.cache.max_infer_batch_size:
                session = self.decode_queue.pop(0)
                allocate = False
                if self.cache.allocate_cache(session.device_session):
                    allocate = True
                else:
                    for _, release_num in release_iter:
                        num_reset_sessions += release_num
                        if self.cache.allocate_cache(session.device_session):
                            allocate = True
                            break
                if allocate:
                    decode_sessions.append(session)
                else:
                    self.decode_queue.insert(0, session)
                    break
        if num_reset_sessions != 0:
            get_logger().debug(f"Reset {num_reset_sessions} for decode")

        # decode
        if len(decode_sessions) > 0:
            batch_logits = self.model.decode([x.device_session for x in decode_sessions], self.cache)
            self.update_with_logits(decode_sessions, batch_logits)
        # check finish
        num_finish = 0
        for session in decode_sessions:
            finish_reason = session.should_stop()
            if finish_reason != "none":
                await self.finish_item(session, session, reason=finish_reason)
                num_finish += 1
            else:
                self.decode_queue.append(session)

        return decode_sessions, num_finish

    # serve

    @torch.inference_mode()
    async def _serve(self):
        def get_engine_status():
            return "\t".join(
                [
                    f"Cache: {int(self.cache.cache_usage() * 100)}%",
                    f"PQ: {len(self.prefill_queue)}({int(self.cache.cache_usage([x.device_session for x in self.prefill_queue]) * 100)}%)",
                    f"DQ: {len(self.decode_queue)}({int(self.cache.cache_usage([x.device_session for x in self.decode_queue]) * 100)}%)",
                    f"FQ: {len(self.device_sessions)}({int(self.cache.cache_usage(list(self.device_sessions.values())) * 100)}%)",
                ]
            )

        def log_decode(decode_status, engine_status=None):
            if decode_status["step"] > 0:
                num_tokens = decode_status["batch"] * decode_status["step"]
                get_logger().debug(
                    f"Decode:\tbatch: {decode_status['batch']},\tstep:   {decode_status['step']}\tThroughput: {int(num_tokens / (max(time.time() - decode_status['start'], 1e-5))):>5} tokens/s\t"
                    + (engine_status if engine_status is not None else get_engine_status())
                )
            decode_status["start"] = time.time()
            decode_status["batch"] = -1
            decode_status["step"] = 0

        decode_status = {"start": time.time(), "batch": 0, "step": 0}

        while True:
            try:
                release = await self.update_queue()
                if release:
                    num = len(self.decode_queue) + len(self.prefill_queue)
                    while len(self.decode_queue) > 0:
                        seq = self.decode_queue.pop()
                        await self.finish_item(seq, seq, reason="abort")
                    while len(self.prefill_queue) > 0:
                        seq = self.prefill_queue.pop()
                        await self.finish_item(seq, seq, reason="abort")
                    get_logger().info(f"Release all {num} tasks in scheduler")
                    await self.finish_item(release, None)
                    return "sleep", 0.1, 0
                else:
                    # try prefill
                    if (
                        self.cache.cache_usage([x.device_session for x in self.decode_queue])
                        < self.bottom_cache_usage_for_decode
                    ):
                        status_before_prefill = get_engine_status()
                        t0 = time.time()
                        num_prefill_sessions, num_prefill_tokens = await self.prefill()
                        if num_prefill_sessions > 0:
                            log_decode(decode_status, status_before_prefill)
                            get_logger().debug(
                                f"Prefill:\tbatch: {num_prefill_sessions},\ttokens:\t{num_prefill_tokens}\tThroughput: {int(num_prefill_tokens / (max(time.time() - t0, 1e-5))):>5} tokens/s\t"
                                + get_engine_status()
                            )

                    # try decode
                    decode_sessions, num_finish = await self.decode()

                    if decode_status["batch"] != len(decode_sessions) or (decode_status["step"] + 1) % 1024 == 0:
                        log_decode(decode_status)
                        decode_status["batch"] = len(decode_sessions)
                    if len(decode_sessions) > 0:
                        decode_status["step"] += 1

                    if (
                        (decode_status["step"] + 1) % 16 == 0
                        or num_finish > 0
                        or len(self.decode_queue) < self.cache.max_infer_batch_size
                    ):
                        await asyncio.sleep(0)  # yield to server for finishing tasks

            except Exception as e:
                get_logger().error(f"Scheduler server encountered an error: {e}")
                print("Full traceback:", traceback.format_exc())
                raise e

    async def launch(self):
        loop = asyncio.get_event_loop()
        self.cache.prepare()
        self.max_infer_length = self.cache.max_infer_length
        self.model.infer_shard(self.max_prefill_length)
        task = loop.create_task(self._serve())
        self.task = task
        get_logger().info("launch scheduler successfully")

    async def stop_server(self):
        if self.task is not None:
            task = self.task
            self.task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self.cache.close()
                get_logger().info("Scheduler server stopped successfully")

    async def finish_item(self, item: TaskItem, result, reason="stop"):
        if isinstance(item, InferItem):
            result = item.get_finish_result()
            if reason != "none":
                result["finish_reason"] = reason
            if item.device_session is not None:
                if item.gene_config.release_at_once:
                    self.cache.reset([item.device_session])
                else:
                    self.device_sessions[item.session_id] = item.device_session
        item._result = {"item": result, "reason": reason}
        item.finished_event.set()
