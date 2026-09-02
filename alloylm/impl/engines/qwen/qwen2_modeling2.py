import copy
import gc
import re
from contextlib import contextmanager
from functools import partial
from typing import Optional, Unpack

import flashinfer
import torch
from accelerate.utils import set_module_tensor_to_device
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed._tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.utils import TransformersKwargs

from alloylm.engine.infer_engine.utils import GatherContext, get_logger
from alloylm.engine.model import AlloyLMModel, DeviceSession, TrainInput
from alloylm.engine.train_engine.utils import (
    DEFAULT_FSDP_CONFIG,
    FSDPConfig,
    HFCheckpointLoader,
    lazy_init_fn,
    pad_to_multiple_of,
    split_for_sequence_parallel,
)

from .flash_attn import flash_attn_varlen_fwd
from .moe_layer import Qwen3MoeSparseMoeBlock
from .swa_cache import AttentionArgs, InferKernel, SwaCacheManager


def all_to_all(
    input: torch.Tensor, scatter_dim: int, gather_dim: int, mesh: DeviceMesh, training=True
) -> torch.Tensor:
    world_size = mesh.size()
    split_size = input.size(scatter_dim) // world_size
    input_split_sizes = [split_size] * world_size
    output_split_sizes = input_split_sizes

    input = input.contiguous()
    input = input.movedim(scatter_dim, 0)
    if training:
        all_to_all_function = all_to_all_single_autograd
    else:
        all_to_all_function = all_to_all_single

    output = all_to_all_function(
        input,
        group=mesh.get_group(),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )
    output = output.transpose(0, scatter_dim)

    output_list = [t for t in torch.tensor_split(output, world_size, scatter_dim)]
    output = torch.cat(output_list, dim=gather_dim).contiguous()
    return output


class DisableGcGollect:
    def __init__(self):
        self.gc_enabled = gc.isenabled()

    def __enter__(self):
        gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gc_enabled:
            gc.enable()


class Qwen2MLP(nn.Module):
    class GateUpProj(nn.Linear):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__(hidden_size, 2 * intermediate_size, bias=False)
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = self.__class__.GateUpProj(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        down_proj = self.down_proj(self.act_fn(gate) * up)
        return down_proj

    def infer_shard(self, fsdp_config: FSDPConfig):
        tp_mesh = fsdp_config.infer_mesh["tp"]
        tp_size = tp_mesh.size()
        if tp_size > 1:
            assert self.intermediate_size % tp_size == 0, (
                f"intermediate_size ({self.intermediate_size}) must be divisible by tp_size ({tp_size})"
            )
            chunk = self.intermediate_size // tp_size
            # permute gate_up_proj to adapt to the tensor parallelism sharding
            perm = torch.cat(
                [
                    torch.cat(
                        [
                            torch.arange(r * chunk, (r + 1) * chunk),
                            self.intermediate_size + torch.arange(r * chunk, (r + 1) * chunk),
                        ]
                    )
                    for r in range(tp_size)
                ]
            )
            # use parameter to avoid overwriting the original weight data which is used by fsdp
            self.gate_up_proj.weight = nn.Parameter(
                self.gate_up_proj.weight.data[perm].contiguous(), requires_grad=self.gate_up_proj.weight.requires_grad
            )

            self.gate_up_proj = parallelize_module(self.gate_up_proj, tp_mesh, parallelize_plan=ColwiseParallel())
            self.down_proj = parallelize_module(self.down_proj, tp_mesh, parallelize_plan=RowwiseParallel())


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen,
    head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """Qwen3RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=self.weight.shape, weight=self.weight, eps=self.variance_epsilon
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: Qwen2Config | Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        if isinstance(config, (Qwen3Config, Qwen3MoeConfig)):
            qkv_bias = config.attention_bias
            o_bias = config.attention_bias
            self.qk_norm = True
            self.q_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            qkv_bias = True
            o_bias = False
            self.qk_norm = False

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=o_bias)
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types") and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states, input_shape = self.qkv_proj(hidden_states, position_embeddings)
        attention_args = kwargs.get("attention_args", None)
        if attention_args:
            query_states, key_states, value_states = (
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
            )
            prefilling = kwargs.get("prefilling", False)
            infer_args = [query_states, key_states, value_states, attention_args]
            if prefilling:
                attn_output = self.forward_prefill(*infer_args)[0]
            else:
                attn_output = self.forward_decoding(*infer_args)[0]
        else:
            attn_output = self.forward_training(query_states, key_states, value_states, **kwargs)[0]
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def qkv_proj(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
        """Project hidden states to Q/K/V and apply rotary embeddings."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.qk_norm:
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        return query_states, key_states, value_states, input_shape

    @torch.compiler.disable
    def append_kv_cache(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_args: AttentionArgs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Squeeze/transpose Q/K/V and append K/V to the paged KV cache."""
        layer_k_pool = attention_args.kv_cache[self.layer_idx, 0]
        layer_v_pool = attention_args.kv_cache[self.layer_idx, 1]

        k_new = key_states.squeeze(0).transpose(0, 1).contiguous()
        v_new = value_states.squeeze(0).transpose(0, 1).contiguous()
        q_states = query_states.squeeze(0).transpose(0, 1).contiguous()

        flashinfer.page.append_paged_kv_cache(
            k_new,
            v_new,
            attention_args.batch_indices,
            attention_args.positions,
            (layer_k_pool, layer_v_pool),
            attention_args.block_index,
            attention_args.cum_num_block,
            attention_args.paged_kv_last_page_len,
            kv_layout="NHD",
        )
        return q_states, layer_k_pool, layer_v_pool

    def forward_training(
        self: "Qwen2Attention",
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        sequence_parallel_mesh: DeviceMesh | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            window_size = (self.config.sliding_window - 1, self.config.sliding_window - 1)
        else:
            window_size = (None, None)

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            raise NotImplementedError("Sequence parallelism is not supported in training mode yet.")
            sp_size = sequence_parallel_mesh.size()
            num_kv_heads = key_states.size(1)
            if sp_size > num_kv_heads:
                assert sp_size % num_kv_heads == 0
                key_states = repeat_kv(key_states, sp_size // num_kv_heads)
                value_states = repeat_kv(value_states, sp_size // num_kv_heads)

            query_states = all_to_all(
                query_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh, training=self.training
            )
            key_states = all_to_all(
                key_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh, training=self.training
            )
            value_states = all_to_all(
                value_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh, training=self.training
            )

        # (bs, n , qh // sp, d)
        attn_output, _ = flash_attn_varlen_fwd(
            query_states.flatten(0, 1),
            key_states.flatten(0, 1),
            value_states.flatten(0, 1),
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
            self.scaling,
            True,
            window_size[0],
            window_size[1],
        )
        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            attn_output = all_to_all(
                attn_output, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh, training=self.training
            )

        return attn_output, None

    @torch.compiler.disable
    def forward_prefill(
        self,
        query_states,
        key_states,
        value_states,
        attention_args: AttentionArgs,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        q_states, layer_k_pool, layer_v_pool = self.append_kv_cache(
            query_states, key_states, value_states, attention_args
        )

        attn_output = attention_args.prefill_kernel.run(
            q_states,
            (layer_k_pool, layer_v_pool),
        )

        return attn_output, None

    @torch.compiler.disable
    def forward_decoding(
        self,
        query_states,
        key_states,
        value_states,
        attention_args: AttentionArgs,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        q_states, layer_k_pool, layer_v_pool = self.append_kv_cache(
            query_states, key_states, value_states, attention_args
        )

        attn_output = attention_args.decode_kernel.run(
            q_states,
            (layer_k_pool, layer_v_pool),
            q_len_per_req=1,
        )
        return attn_output, None

    def infer_shard(self, fsdp_config: FSDPConfig):
        tp_mesh = fsdp_config.infer_mesh["tp"]
        tp_size = tp_mesh.size()
        if tp_size > 1:
            self.q_proj = parallelize_module(self.q_proj, tp_mesh, parallelize_plan=ColwiseParallel())
            self.k_proj = parallelize_module(self.k_proj, tp_mesh, parallelize_plan=ColwiseParallel())
            self.v_proj = parallelize_module(
                self.v_proj, tp_mesh, parallelize_plan=ColwiseParallel()
            )  # output-dim shard
            self.o_proj = parallelize_module(
                self.o_proj, tp_mesh, parallelize_plan=RowwiseParallel()
            )  # input-dim shard


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        if isinstance(config, Qwen3MoeConfig):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states.to(dtype))
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states.to(dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def infer_shard(self, fsdp_config: FSDPConfig):
        self.self_attn.infer_shard(fsdp_config)
        self.mlp.infer_shard(fsdp_config)


class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config

    _can_record_outputs = {  # noqa: RUF012
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        else:
            self.rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen2Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        if hasattr(config, "rope_theta"):
            base = config.rope_theta  # compatible for old transformers
        else:
            base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = (
            "sliding_attention" in self.config.layer_types if hasattr(self.config, "layer_types") else False
        )

        # Initialize weights and apply final processing

    def forward(
        self,
        hidden_states: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # It may already have been prepared by e.g. `generate`

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return hidden_states


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.model.padding_idx)
        # Initialize weights and apply final processing
        self.post_init()

        self.using_inference_context = None

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        compute_logits=True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        hidden_states = self.embed_tokens(input_ids).bfloat16()
        hidden_states = self.model(hidden_states, position_ids=position_ids, **kwargs)
        hidden_states = self.norm(hidden_states.to(self.norm.weight.dtype))
        if compute_logits:
            return self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        else:
            return hidden_states


class FSDPQwen2ForCausalLM(Qwen2ForCausalLM, AlloyLMModel):
    device_type = "cuda"
    weight_map = {  # noqa: RUF012
        "norm.weight": "model.norm.weight",
        "embed_tokens.weight": "model.embed_tokens.weight",
    }

    def __init__(self, config, fsdp_config: FSDPConfig = DEFAULT_FSDP_CONFIG):
        Qwen2ForCausalLM.__init__(self, config)
        AlloyLMModel.__init__(self, fsdp_config=fsdp_config)

        self.gathered = False

    def apply_swa_config(self):
        """Add per-layer sliding window sizes (ws) to config for SWA engine."""
        config = self.config
        ws = []
        if hasattr(config, "layer_types"):
            for layer_type in config.layer_types:
                if layer_type == "full_attention":
                    ws.append(-1)
                else:
                    ws.append(config.sliding_window)
        else:
            if config.use_sliding_window:
                ws = [config.sliding_window] * config.num_hidden_layers
            else:
                ws = [-1] * config.num_hidden_layers
        config.ws = ws

    # forward and train

    def prepare_train_args(
        self,
        input_ids,
        position_ids,
        cu_seq_lens_q=None,
        cu_seq_lens_k=None,
        max_length_q=None,
        max_length_k=None,
        sequence_parallel_mesh=None,
        **kwargs,
    ):
        _input_ids = input_ids
        _position_ids = position_ids
        _cu_seq_lens_q = cu_seq_lens_q
        _cu_seq_lens_k = cu_seq_lens_k
        _max_length_q = max_length_q
        _max_length_k = max_length_k

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            multiple_of = sequence_parallel_mesh.size() * 1
        else:
            multiple_of = 1

        _input_ids = pad_to_multiple_of(_input_ids, 0, multiple_of, 1)
        _position_ids = pad_to_multiple_of(_position_ids, 0, multiple_of, 1)

        num_padded_tokens = _input_ids.numel() - input_ids.numel()

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _input_ids = split_for_sequence_parallel(_input_ids, dim=1, sp_mesh=sequence_parallel_mesh)
            _position_ids = split_for_sequence_parallel(_position_ids, dim=1, sp_mesh=sequence_parallel_mesh)

        if self.training and num_padded_tokens > 0:
            assert torch.any(cu_seq_lens_k == cu_seq_lens_q)
            _cu_seq_lens_q = _cu_seq_lens_q.tolist()
            _cu_seq_lens_q.append(_cu_seq_lens_q[-1] + num_padded_tokens)

            _cu_seq_lens_q = torch.IntTensor(_cu_seq_lens_q).to(cu_seq_lens_q.device)
            _cu_seq_lens_k = _cu_seq_lens_q

            _max_length_q = max(_max_length_q, num_padded_tokens)
            _max_length_k = _max_length_q

        return (_input_ids, _position_ids), {
            "cu_seq_lens_q": _cu_seq_lens_q,
            "cu_seq_lens_k": _cu_seq_lens_k,
            "max_length_q": _max_length_q,
            "max_length_k": _max_length_k,
            "sequence_parallel_mesh": self.fsdp_config.train_mesh["sp"],
        }

    def train_forward(self, input: TrainInput):
        cu_seq_lens_q = torch.nn.functional.pad(input.seq_lens, (1, 0), value=0).cumsum(dim=0, dtype=torch.int32)
        cu_seq_lens_k = cu_seq_lens_q
        max_length_q = input.seq_lens.max().item()
        max_length_k = max_length_q
        return self(
            input_ids=input.input_ids,
            position_ids=input.position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            sequence_parallel_mesh=self.fsdp_config.train_mesh["sp"],
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if self.shard_mode == "infer" and "attention_args" not in kwargs:
            tp_mesh = self.fsdp_config.infer_mesh["tp"]
            if tp_mesh.size() > 1:
                object_list = [input_ids, position_ids, kwargs]
                dist.broadcast_object_list(
                    object_list,
                    src=tp_mesh.mesh.flatten()[0].item(),
                    group=tp_mesh.get_group(),
                )
                input_ids, position_ids, kwargs = object_list
        if "attention_args" not in kwargs:
            (input_ids, position_ids), kwargs = self.prepare_train_args(input_ids, position_ids, **kwargs)
        return super().forward(input_ids=input_ids, position_ids=position_ids, **kwargs)

    # for hf checkpoint loading

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        *model_args,
        fsdp_config: FSDPConfig = DEFAULT_FSDP_CONFIG,
        **kwargs,
    ) -> "FSDPQwen2ForCausalLM":
        class HFCheckpointLoaderWithMapping(HFCheckpointLoader):
            def __init__(self, pretrained_model_name_or_path, mapping=None, cfg=None):
                if mapping is None:
                    mapping = {}
                super().__init__(pretrained_model_name_or_path)
                self.mapping = mapping
                self.cfg = cfg

            def load(self, key):
                key = self.mapping.get(key, key)
                m = re.match(r"^(.*\.mlp)\.gate_up_proj\.weight$", key)
                if m:
                    prefix = m.group(1)
                    gate = super().load(f"{prefix}.gate_proj.weight")
                    up = super().load(f"{prefix}.up_proj.weight")
                    return torch.cat([gate, up], dim=0)
                # Handle consolidated expert weights for Qwen3MoE:
                # HF stores per-expert weights as experts.{i}.gate_proj / up_proj / down_proj,
                # but Qwen3MoeExperts uses 3D stacked tensors: gate_up_proj and down_proj.
                m = re.match(r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$", key)
                if m and self.cfg is not None:
                    prefix = m.group(1)  # e.g. "model.layers.0.mlp.experts"
                    param_name = m.group(2)
                    num_experts = self.cfg.num_experts
                    if param_name == "gate_up_proj":
                        # Stack: [num_experts, 2*intermediate_dim, hidden_dim]
                        gate_list = [
                            super(HFCheckpointLoaderWithMapping, self).load(f"{prefix}.{i}.gate_proj.weight")
                            for i in range(num_experts)
                        ]
                        up_list = [
                            super(HFCheckpointLoaderWithMapping, self).load(f"{prefix}.{i}.up_proj.weight")
                            for i in range(num_experts)
                        ]
                        return torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate_list, up_list)], dim=0)
                    else:  # down_proj
                        # Stack: [num_experts, hidden_dim, intermediate_dim]
                        down_list = [
                            super(HFCheckpointLoaderWithMapping, self).load(f"{prefix}.{i}.down_proj.weight")
                            for i in range(num_experts)
                        ]
                        return torch.stack(down_list, dim=0)
                else:
                    return super().load(key)

        def lazy_init_fn_with_broadcast(module, module2name, checkpoint_loader):
            def broadcast_weight(module):
                for name, param in module.named_parameters(recurse=False):
                    dist.broadcast(param.data, src=0)
                for name, buffer in module.named_buffers(recurse=False):
                    dist.broadcast(buffer.data, src=0)

            if dist.get_rank() == 0:
                lazy_init_fn(module, module2name, checkpoint_loader)
            else:
                module.to_empty(device=torch.device("cuda"), recurse=False)
            dist.barrier()
            broadcast_weight(module)

        assert dist.is_initialized(), "FSDP loading requires torch.distributed to be initialized"
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        fsdp_config.init_device_mesh()
        with torch.device("meta"):
            model = cls(config, fsdp_config=fsdp_config).to(fsdp_config.shard_dtype)

        param_init_fn = partial(
            lazy_init_fn_with_broadcast,
            module2name={mod: name for name, mod in model.named_modules()},
            checkpoint_loader=HFCheckpointLoaderWithMapping(
                pretrained_model_name_or_path, mapping=cls.weight_map, cfg=config
            ),
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype,
            reduce_dtype=fsdp_config.reduce_dtype,
        )

        # Reinitialize rotary embedding on real device (computed, not from checkpoint)
        model.model.rotary_emb = Qwen2RotaryEmbedding(config)
        num_recompute_layers = int(config.num_hidden_layers * fsdp_config.recompute_ratio)

        for layer in tqdm(model.model.layers, desc="Loading layers"):
            layer.apply(param_init_fn)
            fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
            layer_idx = layer.self_attn.layer_idx
            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(layer, preserve_rng_state=False)
            # Compile the whole (checkpoint-wrapped) layer as one graph. Attention
            # is a custom op (no graph break), so this is the supported
            # compile(checkpoint(layer)) nesting: Dynamo's checkpoint HOP recomputes
            # from a stable AOTAutograd partition, avoiding the CheckpointError that
            # checkpoint(compile(layer)) hit on varlen data.
            if fsdp_config.torch_compile:
                layer = torch.compile(layer)
            model.model.layers[layer_idx] = layer

        if config.tie_word_embeddings:
            model.embed_tokens.apply(param_init_fn)
            model.lm_head.weight = model.embed_tokens.weight
        else:
            model.embed_tokens.apply(param_init_fn)
            model.lm_head.apply(param_init_fn)
        model.norm.apply(param_init_fn)

        fully_shard_model = partial(
            fully_shard,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=fsdp_config.lm_head_dtype,
                reduce_dtype=fsdp_config.lm_head_dtype,
            ),
            reshard_after_forward=True,
        )

        if not config.tie_word_embeddings:
            fully_shard_model(model.lm_head)
        fully_shard_model(model)
        return model

    def save_pretrained(
        self,
        save_directory,
        **kwargs,
    ):
        dtype = (
            getattr(torch, self.config.torch_dtype)
            if isinstance(self.config.torch_dtype, str)
            else self.config.torch_dtype
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(self.config)
        for name, param in self.state_dict().items():
            if self.fsdp_config.torch_compile and "_orig_mod." in name:
                name = name.replace("_orig_mod.", "")
            if isinstance(param, DTensor):
                full_param = param.to(dtype).full_tensor()
            else:
                full_param = param.to(dtype)
            if dist.get_rank() == 0:
                if name.endswith(".mlp.gate_up_proj.weight"):
                    prefix = name[: -len(".gate_up_proj.weight")]
                    gate, up = full_param.chunk(2, dim=0)
                    set_module_tensor_to_device(
                        model,
                        self.weight_map.get(f"{prefix}.gate_proj.weight", f"{prefix}.gate_proj.weight"),
                        "cpu",
                        gate.cpu(),
                    )
                    set_module_tensor_to_device(
                        model,
                        self.weight_map.get(f"{prefix}.up_proj.weight", f"{prefix}.up_proj.weight"),
                        "cpu",
                        up.cpu(),
                    )
                else:
                    set_module_tensor_to_device(model, self.weight_map.get(name, name), "cpu", full_param.cpu())
        if dist.get_rank() == 0:
            model.save_pretrained(save_directory, **kwargs)
        del model

    # shard modes

    @contextmanager
    def infer_context(self):
        gather_context = GatherContext(self)
        for module in gather_context.unshard():
            if module is not self and hasattr(module, "infer_shard"):
                module.infer_shard(self.fsdp_config)

        # deal with lm_head and embed_tokens
        tp_mesh = self.fsdp_config.infer_mesh["tp"]
        if tp_mesh.size() > 1:
            self.embed_tokens = parallelize_module(
                self.embed_tokens, tp_mesh, parallelize_plan=RowwiseParallel(input_layouts=Replicate())
            )
            lm_head_hook_handles = []
            if self.config.tie_word_embeddings:

                def forward_pre_hook(module, input):
                    (hidden_states,) = input
                    if not isinstance(hidden_states, DTensor):
                        hidden_states = DTensor.from_local(hidden_states, module.weight.device_mesh, (Replicate(),))
                    return (hidden_states,)

                def forward_hook(module, input, output):
                    if isinstance(output, DTensor):
                        return output.to_local()
                    else:
                        return output

                self.lm_head.weight = self.embed_tokens.weight
                lm_head_hook_handles.append(self.lm_head.register_forward_pre_hook(forward_pre_hook))
                lm_head_hook_handles.append(self.lm_head.register_forward_hook(forward_hook))
            else:
                self.lm_head = parallelize_module(
                    self.lm_head, tp_mesh, parallelize_plan=ColwiseParallel(output_layouts=Replicate())
                )
        # finish infer mode
        yield
        # back to train mode
        for _ in gather_context.reshard():
            pass
        if tp_mesh.size() > 1:
            for module in self.modules():
                # undo MoE tensor-parallel sharding: reset the transient TP flags
                # so the FSDP-restored (full) expert weights are used in train mode
                if hasattr(module, "disable_tp"):
                    module.disable_tp()
                # pop tp hooks for lm_head and embed_tokens
                if hasattr(module, "_distribute_module_applied") and module._distribute_module_applied:
                    delattr(module, "_distribute_module_applied")
                for key in copy.copy(list(module._forward_pre_hooks.keys())):
                    func = module._forward_pre_hooks[key]
                    if "distribute_module." in func.__qualname__:
                        module._forward_pre_hooks.pop(key)
                for key in copy.copy(list(module._forward_hooks.keys())):
                    func = module._forward_hooks[key]
                    if "distribute_module." in func.__qualname__:
                        module._forward_hooks.pop(key)
            for handle in lm_head_hook_handles:
                handle.remove()
            if self.config.tie_word_embeddings:
                # During inference embed_tokens was TP-parallelized and lm_head
                # was re-tied to that sharded DTensor. reshard() restores
                # embed_tokens.weight on its own module, but lm_head still points
                # at the stale inference-time DTensor, so re-tie it here.
                self.lm_head.weight = self.embed_tokens.weight

    def train_shard(self):
        if self.using_inference_context is not None:
            # back to fsdp
            self.using_inference_context.__exit__(None, None, None)
            self.using_inference_context = None
        self.shard_mode = "train"

    def infer_shard(self, max_prefill_length: int = 0):
        if self.using_inference_context is None:
            self.using_inference_context = self.infer_context()
            self.using_inference_context.__enter__()
        self.shard_mode = "infer"

        from .moe_layer import Qwen3MoeExpertsWithGroupGemm

        for module in self.modules():
            if isinstance(module, Qwen3MoeExpertsWithGroupGemm):
                module.prepare_for_cuda_graph(max_prefill_length + 128)

    # for inference

    def get_real_vocab_size(self, tokenizer):
        real_vocab_size = min(
            self.config.vocab_size,
            max(tokenizer.vocab_size, max(tokenizer.added_tokens_decoder.keys()) + 1),
        )
        get_logger().info(f"real_vocab_size: {real_vocab_size}, while model vocab_size: {self.config.vocab_size}")
        return -1 if real_vocab_size == self.config.vocab_size else real_vocab_size

    def create_cache(self, memory_usage=0.8, use_cuda_graph=True) -> SwaCacheManager:
        self.apply_swa_config()
        return SwaCacheManager(
            num_layers=self.config.num_hidden_layers,
            num_head=self.config.num_key_value_heads,
            head_dim=getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads),
            window_size=self.config.ws,
            memory_usage=memory_usage,
            pad_size=8,
            device="cuda",
            dtype=torch.bfloat16,
            block_size=256,
            use_cuda_graph=use_cuda_graph,
            prepare_on_init=False,
        )

    @torch.inference_mode()
    def prefill(self, device_sessions: list[DeviceSession], cache: SwaCacheManager) -> torch.Tensor:
        assert isinstance(cache, SwaCacheManager), f"Qwen prefill requires SwaCacheManager, got {type(cache).__name__}"

        input_ids, position_ids, attention_args = cache.get_args(device_sessions)
        InferKernel.prefill_kernel().plan(
            qo_indptr=attention_args.cu_seq_lens_q,
            paged_kv_indptr=attention_args.cum_num_block,
            paged_kv_indices=attention_args.block_index,
            paged_kv_last_page_len=attention_args.paged_kv_last_page_len,
            num_qo_heads=self.config.num_attention_heads,
            num_kv_heads=attention_args.kv_cache.shape[-2],
            head_dim_qk=attention_args.kv_cache.shape[-1],
            page_size=cache.block_size,
            causal=True,
            window_left=self.config.ws[0] if self.config.use_sliding_window else -1,
            q_data_type=attention_args.kv_cache.dtype,
            kv_data_type=attention_args.kv_cache.dtype,
        )
        attention_args.cuda()
        attention_args.prepare_for_fill_cache()

        hidden_states = self(
            input_ids=input_ids.cuda(),
            position_ids=position_ids.cuda(),
            compute_logits=False,
            attention_args=attention_args,
            prefilling=True,
            attention_mask={"full_attention": None, "sliding_attention": None},
        )
        logits_index = attention_args.cu_seq_lens_q[1:] - 1
        hidden_states = hidden_states.index_select(1, logits_index).squeeze(0)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

        for session in device_sessions:
            session.step()
            cache.release_cache_for_swa(session)
        return logits

    def decode(self, device_sessions: list[DeviceSession], cache: SwaCacheManager) -> torch.Tensor:
        assert isinstance(cache, SwaCacheManager), f"Qwen prefill requires SwaCacheManager, got {type(cache).__name__}"
        input_ids, position_ids, attention_args = cache.get_args(device_sessions)
        attention_args.decode_kernel.plan(
            indptr=attention_args.cum_num_block,
            indices=attention_args.block_index,
            last_page_len=attention_args.paged_kv_last_page_len,
            num_qo_heads=self.config.num_attention_heads,
            num_kv_heads=attention_args.kv_cache.shape[-2],
            head_dim=attention_args.kv_cache.shape[-1],
            page_size=cache.block_size,
            window_left=self.config.ws[0] if self.config.use_sliding_window else -1,
            q_data_type=attention_args.kv_cache.dtype,
            kv_data_type=attention_args.kv_cache.dtype,
            seq_lens=attention_args.num_cache_tokens,
        )
        attention_args.cuda()
        attention_args.prepare_for_fill_cache()

        if attention_args.cuda_graph:
            logits = self.cuda_graph_decoding(
                input_ids=input_ids.cuda(),
                position_ids=position_ids.cuda(),
                attention_args=attention_args,
                cache=cache,
            ).squeeze(0)
        else:
            logits = self(
                input_ids=input_ids.cuda(),
                position_ids=position_ids.cuda(),
                attention_args=attention_args,
                prefilling=False,
                attention_mask={"full_attention": None, "sliding_attention": None},
            ).squeeze(0)

        for session in device_sessions:
            session.step()
            cache.release_cache_for_swa(session)
        return logits

    def cuda_graph_decoding(self, input_ids, position_ids, attention_args, cache: SwaCacheManager):
        batch_size = input_ids.numel()
        ctx = cache.cuda_graphs.get(batch_size, None)
        if ctx is None:
            return self._cuda_graph_decoding_init(input_ids, position_ids, attention_args, cache)
        else:
            return self._cuda_graph_decoding_replay(input_ids, position_ids, attention_args, cache)

    def _cuda_graph_decoding_init(self, input_ids, position_ids, attention_args, cache: SwaCacheManager):
        batch_size = input_ids.numel()
        ctx = {"graph": None, "kwargs": {}}
        ctx["kwargs"]["input_ids"] = input_ids.clone()
        ctx["kwargs"]["position_ids"] = position_ids.clone()
        ctx["kwargs"]["attention_args"] = attention_args.create_copy_for_cuda_graph()
        if cache.graph_logits is None:
            cache.graph_logits = attention_args.kv_cache.new_empty(
                [1, attention_args.max_decode_batch_size, self.config.vocab_size]
            )

        # warm up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            output = self(
                **ctx["kwargs"],
                prefilling=False,
                attention_mask={"full_attention": None, "sliding_attention": None},
            )
            cache.graph_logits[:, : output.shape[1]].copy_(output)

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph), torch.no_grad():
            fake_output = self(
                **ctx["kwargs"],
                prefilling=False,
                attention_mask={"full_attention": None, "sliding_attention": None},
            )
            cache.graph_logits[:, : output.shape[1]].copy_(fake_output)

        ctx["graph"] = graph
        cache.cuda_graphs[batch_size] = ctx
        return output

    def _cuda_graph_decoding_replay(self, input_ids, position_ids, attention_args, cache: SwaCacheManager):
        batch_size = input_ids.numel()
        ctx = cache.cuda_graphs[batch_size]
        ctx["kwargs"]["input_ids"].copy_(input_ids)
        ctx["kwargs"]["position_ids"].copy_(position_ids)
        ctx["kwargs"]["attention_args"].copy_for_cuda_graph(attention_args)
        with DisableGcGollect():
            ctx["graph"].replay()
        return cache.graph_logits[:, :batch_size]
