from itertools import chain

import flashinfer
import torch
from pydantic import BaseModel
from torch.nn import functional as F

from alloylm.engine.model import Cache, DeviceSession

from ....engine.infer_engine.utils import get_logger

# functions


def packed_cumulative_length(num_tokens: torch.Tensor):
    # support multi dim
    _pad_length = F.pad(num_tokens, (1, 0), value=0)  # pad at the beginning
    return torch.cumsum(_pad_length, -1).int()


# kernel


class InferKernel:
    _prefill_kernel = None
    _decode_kernel = {}

    _float_workspace_buffer = None

    _paged_kv_indptr_buffer = None
    _paged_kv_last_page_len_buffer = None
    _paged_kv_indices_buffer = None

    _paged_kv_indices_buffer_for_cuda_graph = None  # shared for all cuda graph

    @classmethod
    def init_buffer(cls, max_num_page, max_batch_size):
        cls._float_workspace_buffer = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        cls._paged_kv_indptr_buffer = torch.empty(max_batch_size + 1, dtype=torch.int32, device="cuda")
        cls._paged_kv_indices_buffer = torch.empty(max_num_page, dtype=torch.int32, device="cuda")
        cls._paged_kv_last_page_len_buffer = torch.empty(max_batch_size, dtype=torch.int32, device="cuda")

        cls._paged_kv_indices_buffer_for_cuda_graph = torch.empty(max_num_page, dtype=torch.int32, device="cuda")

    @classmethod
    def release_buffer(cls):
        cls._prefill_kernel = None
        cls._decode_kernel = {}
        cls._float_workspace_buffer = None
        cls._paged_kv_indptr_buffer = None
        cls._paged_kv_indices_buffer = None
        cls._paged_kv_last_page_len_buffer = None
        cls._paged_kv_indices_buffer_for_cuda_graph = None

    @classmethod
    def prefill_kernel(cls):
        if cls._prefill_kernel is None:
            cls._prefill_kernel = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                cls._float_workspace_buffer, "NHD"
            )
        return cls._prefill_kernel

    @classmethod
    def decode_kernel(cls, batch_size, cuda_graph=False):
        if cuda_graph:
            kwargs = dict(
                use_cuda_graph=True,
                paged_kv_indptr_buffer=cls._paged_kv_indptr_buffer[: batch_size + 1],
                paged_kv_indices_buffer=cls._paged_kv_indices_buffer,
                paged_kv_last_page_len_buffer=cls._paged_kv_last_page_len_buffer[:batch_size],
            )
        else:
            kwargs = dict()
            batch_size = 0
        if batch_size not in cls._decode_kernel:
            cls._decode_kernel[batch_size] = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                cls._float_workspace_buffer, "NHD", use_tensor_cores=True, **kwargs
            )

        return cls._decode_kernel[batch_size]


class AttentionArgs(BaseModel):
    cu_seq_lens_q: torch.Tensor
    cum_num_block: torch.Tensor
    block_index: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    kv_cache: torch.Tensor
    num_cache_tokens: torch.Tensor

    max_decode_batch_size: int = 0
    cuda_graph: bool = False

    # for cache fill
    total_q_len: int | None = None
    batch_indices: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    # kernels
    decode_kernel: flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper | None = None
    prefill_kernel: flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper | None = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, context):
        self.decode_kernel = InferKernel.decode_kernel(
            batch_size=self.cu_seq_lens_q.shape[0] - 1, cuda_graph=self.cuda_graph
        )
        self.prefill_kernel = InferKernel.prefill_kernel()
        self.total_q_len = self.cu_seq_lens_q[-1].item()

    def prepare_for_fill_cache(self):
        if self.batch_indices is None or self.positions is None:
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                self.cu_seq_lens_q, self.num_cache_tokens, self.total_q_len
            )
            self.batch_indices = batch_indices
            self.positions = positions

    def copy_for_cuda_graph(self, args: "AttentionArgs"):
        assert args.cuda_graph, "copy_for_cuda_graph should only be called when cuda_graph is True"
        self.cu_seq_lens_q.copy_(args.cu_seq_lens_q)
        self.cum_num_block.copy_(args.cum_num_block)
        self.kv_cache = args.kv_cache  # not copy kv_cache, just reference
        self.paged_kv_last_page_len.copy_(args.paged_kv_last_page_len)
        if self.block_index.numel() < args.block_index.numel():
            raise RuntimeError(
                f"CUDA graph block index capacity is too small: {self.block_index.numel()} < {args.block_index.numel()}"
            )
        self.block_index[: args.block_index.numel()].copy_(args.block_index)
        self.num_cache_tokens.copy_(args.num_cache_tokens)
        self.batch_indices.copy_(args.batch_indices)
        self.positions.copy_(args.positions)

        self.max_decode_batch_size = args.max_decode_batch_size

    def create_copy_for_cuda_graph(self):
        assert self.cuda_graph, "create_copy_for_cuda_graph should only be called when cuda_graph is True"
        InferKernel._paged_kv_indices_buffer_for_cuda_graph[: self.block_index.numel()].copy_(self.block_index)
        return AttentionArgs(
            cu_seq_lens_q=self.cu_seq_lens_q.clone(),
            cum_num_block=self.cum_num_block.clone(),
            block_index=InferKernel._paged_kv_indices_buffer_for_cuda_graph,
            paged_kv_last_page_len=self.paged_kv_last_page_len.clone(),
            kv_cache=self.kv_cache,  # not copy kv_cache, just reference
            num_cache_tokens=self.num_cache_tokens.clone(),
            batch_indices=self.batch_indices.clone(),
            positions=self.positions.clone(),
            max_decode_batch_size=self.max_decode_batch_size,
            cuda_graph=self.cuda_graph,
        )

    def cuda(self):
        self.cu_seq_lens_q = self.cu_seq_lens_q.cuda()
        self.cum_num_block = self.cum_num_block.cuda()
        self.block_index = self.block_index.cuda()
        self.paged_kv_last_page_len = self.paged_kv_last_page_len.cuda()
        self.num_cache_tokens = self.num_cache_tokens.cuda()
        return self


# cache manager
class AttentionDeviceSession(DeviceSession):
    def __init__(self, session_id: int, block_size=16):
        super().__init__(session_id)

        self.block_table: list = []
        self.block_size = block_size
        self.cached_num = 0

    @property
    def num_cache_needed(self):
        return (len(self.tokens) + self.num_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def num_cache_allocated(self):
        return len(self.block_table)

    def release_forwarded(self):
        super().release_forwarded()
        block_table = self.block_table
        self.block_table = []
        self.cached_num = 0
        return block_table

    def on_device(self):
        return len(self.block_table) > 0

    @property
    def num_cached_tokens(self):
        return self.cached_num

    def step(self):
        self.cached_num += len(self.tokens)
        return super().step()


class SwaCacheManager(Cache):
    def __init__(
        self,
        num_layers=32,
        num_head=32,
        head_dim=128,
        window_size=[4096],
        block_size=16,
        memory_usage=0.8,
        pad_size=8,
        device=torch.cuda,
        dtype=torch.bfloat16,
        use_cuda_graph=True,
        prepare_on_init=True,
    ):
        self.num_layer = num_layers
        self.num_head = num_head
        self.head_dim = head_dim
        self.window_size = [ws for ws in window_size][0]
        self.block_size = block_size
        self.memory_usage = memory_usage
        self.pad_size = pad_size
        self.device = device
        self.dtype = dtype
        self.use_cuda_graph = use_cuda_graph
        self._prepared = False
        self.total_num_blocks = 0
        self.max_batch_size = 0
        self.max_length = 0
        self.past_key_values = None
        self.free_blocks = set()
        self.cuda_graphs = {}
        self.graph_logits = None
        if prepare_on_init:
            self.prepare()

    def prepare(self):
        if self._prepared:
            return
        self.total_num_blocks = self.guess_num_block(memory_usage=self.memory_usage) // self.num_layer // 2
        assert self.total_num_blocks > 0, "Not enough memory to allocate any cache block. increase memory_usage "
        if self.window_size == -1:
            self.max_batch_size = (
                (self.total_num_blocks * self.block_size)
                // max(self.block_size, 1024)
                // self.pad_size
                * self.pad_size
            )
        else:
            self.max_batch_size = (
                (self.total_num_blocks * self.block_size) // self.window_size // self.pad_size * self.pad_size
            )
        self.max_batch_size = min(self.max_batch_size, 512)
        if self.window_size == -1:
            self.max_length = self.total_num_blocks * self.block_size
        else:
            self.max_length = 1000**3  # no length limitation for window attention
        get_logger().info(f"Init Cache for {self.max_length} tokens, max decode batch size {self.max_batch_size}")

        InferKernel.init_buffer(
            max_num_page=self.total_num_blocks + self.pad_size, max_batch_size=self.max_batch_size
        )  # max_batch_size is just a guess, extra pad_size is for decode padding

        # unified paged kv cache pool [L, 2, N_BLOCK, PAGE, H, D]
        self.past_key_values = torch.zeros(
            [
                self.num_layer,
                2,
                self.total_num_blocks,
                self.block_size,
                self.num_head,
                self.head_dim,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.free_blocks = set(range(self.total_num_blocks))
        self.cuda_graphs = {}
        self.graph_logits = None
        self._prepared = True

    def close(self):
        if not self._prepared:
            return
        InferKernel.release_buffer()
        self.past_key_values = None
        self.free_blocks = set()
        self.cuda_graphs = {}
        self.graph_logits = None
        self._prepared = False
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()

    def guess_num_block(self, memory_usage=0.8):
        # Estimate available memory and compute max number of blocks that can be allocated.
        # Use device-wide free memory so we do not over-allocate when other processes occupy GPU memory.
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        free_mem, _ = torch.cuda.mem_get_info(device)
        usable_mem = int(free_mem * memory_usage)
        assert usable_mem > 0, (
            "Not enough usable GPU memory. Try reducing memory_usage or close other GPU applications."
        )
        block_bytes = (
            self.block_size * self.num_head * self.head_dim * torch.tensor([], dtype=self.dtype).element_size()
        )
        max_blocks = usable_mem // block_bytes
        return max_blocks

    @torch.inference_mode()
    def get_args(self, sessions: list[AttentionDeviceSession]):
        assert len(sessions) > 0
        for session in sessions:
            assert session.num_cache_allocated >= session.num_cache_needed, (
                "Session cache is not allocated. Call allocate_cache before forward."
            )

        input_ids = torch.tensor(
            list(chain.from_iterable([session.tokens for session in sessions])), dtype=torch.int32, device="cpu"
        )
        num_q_tokens = torch.tensor([len(session.tokens) for session in sessions], dtype=torch.int32, device="cpu")
        num_cache_tokens = torch.tensor(
            [session.num_cached_tokens + len(session.tokens) for session in sessions],
            dtype=torch.int32,
            device="cpu",
        )
        num_blocks = torch.tensor(
            [session.num_cache_needed for session in sessions], dtype=torch.int32, device="cpu"
        )  # block table more than needed will cause wrong result.
        block_index = torch.tensor(
            list(chain.from_iterable([session.block_table[: session.num_cache_needed] for session in sessions])),
            dtype=torch.int32,
            device="cpu",
        )
        paged_kv_last_page_len = ((num_cache_tokens - 1) % self.block_size + 1).int().clamp(min=1)

        if (num_q_tokens == 1).all():
            position_ids = torch.tensor(
                [len(session.forwarded_tokens) for session in sessions], dtype=torch.int32, device="cpu"
            ).reshape([1, -1])
        else:
            position_ids = [
                torch.arange(
                    len(session.forwarded_tokens),
                    len(session.forwarded_tokens) + num_q_tokens[i].item(),
                    dtype=torch.int32,
                    device="cpu",
                )
                for i, session in enumerate(sessions)
            ]
            position_ids = torch.cat(position_ids, dim=0).reshape([1, -1])

        cu_seq_lens_q = packed_cumulative_length(num_q_tokens)
        cum_num_block = packed_cumulative_length(num_blocks)

        return (
            input_ids.reshape([1, -1]),
            position_ids.reshape([1, -1]),
            AttentionArgs(
                cu_seq_lens_q=cu_seq_lens_q,
                cum_num_block=cum_num_block,
                block_index=block_index,
                paged_kv_last_page_len=paged_kv_last_page_len,
                kv_cache=self.past_key_values,
                num_cache_tokens=num_cache_tokens,
                max_decode_batch_size=self.max_batch_size,
                cuda_graph=self.use_cuda_graph,
            ),
        )

    def allocate_cache(self, session: AttentionDeviceSession) -> bool:
        if not self._prepared:
            raise RuntimeError("Cache is not prepared")
        num_block_needs = session.num_cache_needed - session.num_cache_allocated
        if num_block_needs > len(self.free_blocks):
            return False
        else:
            blocks = []
            for _ in range(num_block_needs):
                blocks.append(self.free_blocks.pop())
            session.block_table.extend(blocks)
            return True

    def create_device_session(self, session_id: int) -> AttentionDeviceSession:
        return AttentionDeviceSession(session_id=session_id, block_size=self.block_size)

    def reset(self, sessions: list[AttentionDeviceSession]):
        for session in sessions:
            self.release_cache(session)

    @property
    def max_infer_length(self):
        return self.max_length

    @property
    def max_infer_batch_size(self):
        return self.max_batch_size

    def release_cache(self, session: AttentionDeviceSession):
        self.free_blocks.update(session.release_forwarded())

    def release_cache_for_swa(self, session: AttentionDeviceSession):
        if self.window_size != -1:
            max_cache_size = self.window_size - 1
            if session.cached_num - max_cache_size >= self.block_size:
                num_release = (session.cached_num - max_cache_size) // self.block_size
                released_blocks = session.block_table[:num_release]
                if len(session.block_table) - len(released_blocks) == self.window_size // self.block_size:
                    # pre-allocate one block to avoid frequent allocation and release when the cache size is around the window size
                    session.block_table = session.block_table[num_release:] + [released_blocks.pop()]
                else:
                    session.block_table = session.block_table[num_release:]
                self.free_blocks.update(released_blocks)
                session.cached_num -= num_release * self.block_size

    def cache_usage(self, device_sessions: list[AttentionDeviceSession] = None):
        if not self._prepared:
            return 0.0
        if device_sessions is None:
            return 1 - len(self.free_blocks) / self.total_num_blocks
        else:
            return sum([session.num_cache_allocated for session in device_sessions]) / self.total_num_blocks
