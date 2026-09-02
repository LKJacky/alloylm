"""Flash-attn varlen attention exposed as torch custom ops.

``flash_attn.cute.flash_attn_varlen_func`` is a ``torch.autograd.Function`` built
on CUTE-DSL kernels that are opaque to Dynamo, so it can only live behind a graph
break (``@torch.compiler.disable``). That forces the fragile
``checkpoint(compile(layer))`` nesting, whose AOTAutograd-chosen saved-tensor
order is not stable across the forward vs. recompute call, producing a
``CheckpointError`` on varlen data.

Wrapping the private fwd/bwd kernels as ``torch.library.custom_op``s makes the
attention a single opaque graph node with a ``register_fake`` shape rule and a
hand-wired ``register_autograd`` mirroring ``FlashAttnVarlenFunc``. The whole
decoder layer then traces to one graph, enabling the supported
``compile(checkpoint(layer))`` nesting (Dynamo's checkpoint HOP) with a stable
partition. The op body runs the real kernel only at runtime; tracing uses the
fake impls, so this is safe (unlike ``allow_in_graph``, which would bake the
fake-mode kernel-skipping path into the graph). Outputs/grads are bit-identical
to ``flash_attn_varlen_func`` (validated). Used unconditionally in training; the
``compile`` wrapping around it is gated on ``FSDPConfig.torch_compile``.
"""

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


@torch.library.custom_op("alloylm::flash_attn_varlen_fwd", mutates_args=())
def flash_attn_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        return_lse=True,
    )
    return out, lse


@flash_attn_varlen_fwd.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, wl, wr):
    total_q, num_head = q.shape[0], q.shape[1]
    head_dim_v = v.shape[-1]
    out = q.new_empty((total_q, num_head, head_dim_v))
    lse = q.new_empty((num_head, total_q), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("alloylm::flash_attn_varlen_bwd", mutates_args=())
def flash_attn_varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale,
        causal,
        0.0,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return dq, dk, dv


@flash_attn_varlen_bwd.register_fake
def _(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, wl, wr):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _flash_attn_varlen_fwd_setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, wl, wr = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen_q, ctx.max_seqlen_k = max_seqlen_q, max_seqlen_k
    ctx.softmax_scale, ctx.causal = softmax_scale, causal
    ctx.window_size_left, ctx.window_size_right = wl, wr


def _flash_attn_varlen_backward(ctx, dout, dlse):
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    dq, dk, dv = flash_attn_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.max_seqlen_q,
        ctx.max_seqlen_k,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size_left,
        ctx.window_size_right,
    )
    # grads for (q, k, v) then None for the 8 non-differentiable inputs
    return dq, dk, dv, None, None, None, None, None, None, None, None


torch.library.register_autograd(
    "alloylm::flash_attn_varlen_fwd",
    _flash_attn_varlen_backward,
    setup_context=_flash_attn_varlen_fwd_setup_context,
)
