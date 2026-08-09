import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.integrations import use_experts_implementation
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from alloylm.engine.train_engine.utils import FSDPConfig


@use_experts_implementation
class Qwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Qwen3MoeExpertsWithGroupGemm(Qwen3MoeExperts):
    def __init__(self, config):
        super().__init__(config)
        # Pre-allocated buffers for CUDA-graph-compatible token counting.
        # Populated by prepare_for_cuda_graph(); None means dynamic allocation.
        self._tpe_buf: torch.Tensor | None = None
        self._ones_buf: torch.Tensor | None = None

        # Expert-parallel state. Populated by configure_ep(); None means EP is
        # disabled and forward_ep() must not be called.
        self.ep_rank: int | None = None
        self.ep_size: int | None = None
        self.ep_start: int | None = None
        self.ep_end: int | None = None
        self.ep_group = None
        self._ep_gate_up_proj: torch.Tensor | None = None
        self._ep_down_proj: torch.Tensor | None = None

        # Tensor-parallel state. Populated by infer_shard_tp(); tp_group is None
        # when TP is disabled and forward() runs the full (unsharded)
        # intermediate dim. tp_local_chunk is this rank's real (un-padded)
        # intermediate slice width, used to drop padding before down_proj.
        self.tp_group = None
        self.tp_local_chunk: int | None = None

    def prepare_for_cuda_graph(self, max_num_tokens: int) -> None:
        """Pre-allocate counting buffers so forward() is CUDA-graph-compatible.

        Args:
            max_num_tokens: Maximum number of *input* tokens per forward call
                (i.e. the decode batch size).  Expanded slots = max_num_tokens * top_k.
        """
        max_expanded = max(max_num_tokens * self.top_k, 1024 * 1024)
        self._tpe_buf = torch.zeros(self.num_experts, dtype=torch.int64, device="cuda")
        self._ones_buf = torch.ones(max_expanded, dtype=torch.int64, device="cuda")

    def _count_tokens_per_expert(self, top_k_index: torch.Tensor, num_experts: int | None = None) -> torch.Tensor:
        """Count tokens routed to each expert.

        CUDA-graph-compatible when ``num_experts`` is ``None`` (uses the
        pre-allocated buffers set up by ``prepare_for_cuda_graph``, if any).
        """
        num_experts = self.num_experts if num_experts is None else num_experts
        flat = top_k_index.view(-1).to(torch.int64)
        n = flat.numel()
        if num_experts == self.num_experts and self._tpe_buf is not None:
            # Fast path: reuse pre-allocated buffers (zero-allocation, graph-safe).
            tpe = self._tpe_buf.zero_()
            tpe.scatter_add_(0, flat, self._ones_buf[:n])
        else:
            # Slow path: allocate on first use or when not in CUDA-graph mode.
            tpe = torch.zeros(num_experts, dtype=torch.int64, device=top_k_index.device)
            tpe.scatter_add_(0, flat, torch.ones(n, dtype=torch.int64, device=top_k_index.device))
        return tpe

    def configure_ep(self, ep_rank: int, ep_size: int, group=None) -> None:
        """Enable expert-parallel forward (see ``forward_ep``) with *real*
        weight sharding: this rank keeps (and ever only materializes) its
        own contiguous slice of ``num_experts // ep_size`` experts -- no
        full replication of expert weights across ranks, unlike a
        mask-and-all-reduce scheme that first gathers every expert onto
        every rank.

        Must be called while ``gate_up_proj`` / ``down_proj`` are still in
        their native FSDP-sharded form (a ``DTensor`` with ``Shard(0)``
        placement), i.e. *before* calling ``unshard()`` on this module. The
        local shard handed out by FSDP (``DTensor.to_local()``) is used
        directly and with zero extra communication as this rank's expert
        weights, so it must line up with ``ep_rank``/``ep_size`` (i.e. the
        FSDP mesh used to shard this module must match the EP group).
        """
        assert self.num_experts % ep_size == 0, "num_experts must be divisible by ep_size for EP"
        experts_per_rank = self.num_experts // ep_size
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ep_start = ep_rank * experts_per_rank
        self.ep_end = self.ep_start + experts_per_rank
        self.ep_group = group

        gate_up = self.gate_up_proj.data
        down = self.down_proj.data
        assert isinstance(gate_up, DTensor) and isinstance(down, DTensor), (
            "configure_ep() requires gate_up_proj/down_proj to still be FSDP-sharded "
            "DTensors (call it before unshard()) so each rank only ever materializes "
            "its own expert shard"
        )
        # DTensor.to_local() is a zero-communication, zero-extra-memory view: this
        # rank's own contiguous chunk of experts as already stored by FSDP sharding.
        self._ep_gate_up_proj = gate_up.to_local()
        self._ep_down_proj = down.to_local()

    def disable_ep(self) -> None:
        self.ep_rank = None
        self.ep_size = None
        self.ep_start = None
        self.ep_end = None
        self.ep_group = None
        self._ep_gate_up_proj = None
        self._ep_down_proj = None

    def infer_shard_tp(self, tp_mesh) -> None:
        """Tensor-parallel shard the expert weights across ``tp_mesh``.

        Each rank keeps a contiguous ``intermediate_dim // tp_size`` slice of
        every expert's intermediate neurons: ``gate_up_proj`` is sharded along
        its output-feature dim (with gate/up kept as a local pair) and
        ``down_proj`` along its input-feature dim. ``forward()`` then produces a
        *partial* hidden-state contribution that is summed across the TP group
        with an ``all_reduce``.

        Must be called while the weights are the full (unsharded) tensors, i.e.
        after FSDP ``unshard()`` (see ``Qwen2ForCausalLM.infer_context``). The
        replaced parameters are transient: FSDP restores the original sharded
        parameters on ``reshard()`` when returning to train mode.
        """
        tp_size = tp_mesh.size()
        if tp_size <= 1:
            return
        tp_rank = tp_mesh.get_local_rank()
        inter = self.intermediate_dim
        assert inter % tp_size == 0, f"moe_intermediate_size ({inter}) must be divisible by tp_size ({tp_size})"
        chunk = inter // tp_size
        # The triton group-gemm requires the gate_up output feature dim
        # (2 * local_chunk) to be a multiple of BLOCK_N (256), i.e. local_chunk
        # a multiple of 128. Pad this rank's slice up if needed; the padded
        # neurons carry zero weights and are sliced off before down_proj, so
        # they contribute nothing.
        padded = ((chunk + 127) // 128) * 128
        pad = padded - chunk

        E = self.num_experts
        gate_up = self.gate_up_proj.data  # (E, 2 * inter, hidden)
        down = self.down_proj.data  # (E, hidden, inter)
        hidden = gate_up.shape[-1]
        gu_req = self.gate_up_proj.requires_grad
        dp_req = self.down_proj.requires_grad

        gate_slice = gate_up[:, tp_rank * chunk : (tp_rank + 1) * chunk, :]
        up_slice = gate_up[:, inter + tp_rank * chunk : inter + (tp_rank + 1) * chunk, :]
        if pad:
            zpad = torch.zeros(E, pad, hidden, device=gate_up.device, dtype=gate_up.dtype)
            local_gate_up = torch.cat([gate_slice, zpad, up_slice, zpad], dim=1).contiguous()
        else:
            local_gate_up = torch.cat([gate_slice, up_slice], dim=1).contiguous()
        local_down = down[:, :, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()

        self.gate_up_proj = nn.Parameter(local_gate_up, requires_grad=gu_req)
        self.down_proj = nn.Parameter(local_down, requires_grad=dp_req)
        self.tp_local_chunk = chunk
        self.tp_group = tp_mesh.get_group()

    def disable_tp(self) -> None:
        self.tp_group = None
        self.tp_local_chunk = None

    def forward_ep(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Expert-parallel forward with real token dispatch.

        Must be called after ``configure_ep``. ``hidden_states`` /
        ``top_k_index`` / ``top_k_weights`` are this rank's own (already
        data-parallel-sharded) slice of tokens -- different ranks may hold
        different tokens.

        Rather than every rank computing every token against a masked-out
        local expert slice and combining with an all-reduce, tokens are
        dispatched with a real ``all_to_all_single`` directly to the rank
        that owns their selected expert: each rank only ever computes the
        tokens it actually received, using its own local expert shard, and
        results are shipped back to the originating rank with a second
        (reverse) ``all_to_all_single``.
        """
        from .group_gemm import (
            cuda_token_permute,
            cuda_token_unpermute,
            triton_group_gemm,
        )

        assert self.ep_start is not None, "call configure_ep() before forward_ep()"
        experts_per_rank = self.ep_end - self.ep_start
        num_tokens, hidden_dim = hidden_states.shape
        top_k = top_k_index.shape[1]

        # 1) Expand top-k slots into individual dispatch rows and figure out
        #    which EP rank owns each row's selected expert.
        expanded_hidden = hidden_states.repeat_interleave(top_k, dim=0)
        flat_expert_id = top_k_index.reshape(-1).to(torch.int64)
        flat_weight = top_k_weights.reshape(-1)
        dest_rank = torch.div(flat_expert_id, experts_per_rank, rounding_mode="floor")

        # 2) Sort rows by destination rank so a single all_to_all_single call
        #    can ship contiguous per-rank blocks.
        sort_idx = torch.argsort(dest_rank, stable=True)
        sorted_hidden = expanded_hidden.index_select(0, sort_idx).contiguous()
        sorted_expert_id = flat_expert_id.index_select(0, sort_idx).contiguous()
        input_split_sizes = torch.bincount(dest_rank, minlength=self.ep_size).tolist()

        # 3) Exchange split sizes (tiny) so every rank knows how many rows it
        #    will receive from every other rank.
        input_splits_t = torch.tensor(input_split_sizes, device=hidden_states.device, dtype=torch.int64)
        output_splits_t = torch.empty_like(input_splits_t)
        dist.all_to_all_single(output_splits_t, input_splits_t, group=self.ep_group)
        output_split_sizes = output_splits_t.tolist()
        num_recv = sum(output_split_sizes)

        # 4) Dispatch: send each token to the rank owning its selected expert.
        recv_hidden = torch.empty(num_recv, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        dist.all_to_all_single(recv_hidden, sorted_hidden, output_split_sizes, input_split_sizes, group=self.ep_group)
        recv_expert_id = torch.empty(num_recv, dtype=torch.int64, device=hidden_states.device)
        dist.all_to_all_single(
            recv_expert_id, sorted_expert_id, output_split_sizes, input_split_sizes, group=self.ep_group
        )

        # 5) Compute: run the group-gemm on this rank's own expert shard, only
        #    for the tokens it actually received -- no wasted compute.
        local_expert_id = recv_expert_id - self.ep_start
        permuted_tokens, row_id_map = cuda_token_permute(recv_hidden, local_expert_id)
        tokens_per_expert = self._count_tokens_per_expert(local_expert_id, num_experts=experts_per_rank)

        gate_up = triton_group_gemm(permuted_tokens, self._ep_gate_up_proj, tokens_per_expert)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up
        down_out = triton_group_gemm(intermediate, self._ep_down_proj, tokens_per_expert)
        expert_out = cuda_token_unpermute(down_out, row_id_map).to(hidden_states.dtype)

        # 6) Combine: send results back to the rank that originally owned
        #    each token (reverse all_to_all -- split sizes swapped).
        combined = torch.empty_like(sorted_hidden)
        dist.all_to_all_single(combined, expert_out, input_split_sizes, output_split_sizes, group=self.ep_group)

        # 7) Undo the dispatch sort, apply routing weights, and sum the
        #    top-k slots of each original token back into a single row.
        weighted = combined * flat_weight.index_select(0, sort_idx).unsqueeze(-1).to(combined.dtype)
        output = torch.empty_like(weighted)
        output[sort_idx] = weighted
        return output.view(num_tokens, top_k, hidden_dim).sum(dim=1)

    @torch.compiler.disable()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from .group_gemm import (
            cuda_token_permute,
            cuda_token_unpermute,
            triton_group_gemm,
        )

        # Permute tokens so they are grouped by expert (sorted by expert id).
        # permuted_tokens: [num_tokens * top_k, hidden_dim]
        # row_id_map:      [num_tokens * top_k]  (argsort of flattened expert ids)
        permuted_tokens, row_id_map = cuda_token_permute(hidden_states, top_k_index)

        # Count how many tokens are routed to each expert.
        tokens_per_expert = self._count_tokens_per_expert(top_k_index)

        # First group-GEMM: gate + up projection
        # gate_up: [num_tokens * top_k, 2 * intermediate_dim]
        gate_up = triton_group_gemm(permuted_tokens, self.gate_up_proj, tokens_per_expert)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.act_fn(gate) * up  # [num_tokens * top_k, intermediate_dim]

        if self.tp_local_chunk is not None:
            # Drop any padding added by infer_shard_tp() so down_proj sees this
            # rank's real intermediate slice width.
            intermediate = intermediate[:, : self.tp_local_chunk].contiguous()

        # Second group-GEMM: down projection
        # down_out: [num_tokens * top_k, hidden_dim]
        down_out = triton_group_gemm(intermediate, self.down_proj, tokens_per_expert)

        # Unpermute tokens back to original order and apply routing weights.
        # output: [num_tokens, hidden_dim]
        output = cuda_token_unpermute(down_out, row_id_map, top_k_weights.float())

        if self.tp_group is not None:
            # Each rank only computed its intermediate-dim slice, so sum the
            # partial hidden-state contributions across the TP group.
            output = output.contiguous()
            dist.all_reduce(output, group=self.tp_group)

        return output.to(hidden_states.dtype)


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    @torch.compile()
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExpertsWithGroupGemm(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def forward_ep(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expert-parallel forward with real token dispatch; see
        ``Qwen3MoeExpertsWithGroupGemm.forward_ep``.

        ``hidden_states`` should be this rank's own data-parallel-sharded
        slice of the batch (e.g. ``torch.chunk(full_batch, world_size,
        dim=0)[rank]``) -- every rank may pass in different tokens.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts.forward_ep(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def infer_shard(self, fsdp_config: FSDPConfig):
        tp_mesh = fsdp_config.infer_mesh["tp"]
        self.experts.infer_shard_tp(tp_mesh)
