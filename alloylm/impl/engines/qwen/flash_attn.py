"""Varlen attention with one call site for both flash-attn generations.

- flash-attn 4 (``flash_attn.cute``): CUTE-DSL kernels wrapped as a custom op, keeps the
  ``compile(checkpoint(layer))`` nesting working (see ``flash_attn_fa4``).
- flash-attn 2 (``flash_attn.flash_attn_varlen_func``): the classic CUDA kernels, used when
  the environment has no ``flash_attn.cute`` (e.g. the ``alloylm_fa2`` venv).

The signature and the ``(output, lse)`` return shape are the same for both, so callers do
not need to know which one is installed.
"""

from __future__ import annotations

import torch

_FA4_ERROR: Exception | None = None
try:
    from .flash_attn_fa4 import flash_attn_varlen_fwd as _fa4_varlen_fwd

    _HAS_FA4 = True
except Exception as error:  # noqa: BLE001 - fa4 is optional, fall back to fa2
    _fa4_varlen_fwd = None
    _HAS_FA4 = False
    _FA4_ERROR = error

try:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen_fwd

    _HAS_FA2 = True
except Exception as error:  # noqa: BLE001 - neither generation installed is a real error
    _fa2_varlen_fwd = None
    _HAS_FA2 = False
    _FA2_ERROR = error

if not _HAS_FA4 and not _HAS_FA2:
    raise ImportError(f"no flash-attn available (fa4: {_FA4_ERROR}; fa2: {_FA2_ERROR})")


def _window(window_size_left, window_size_right):
    """fa2 wants -1 for "unbounded", fa4 wants None."""
    left = -1 if window_size_left is None else window_size_left
    right = -1 if window_size_right is None else window_size_right
    return left, right


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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Varlen attention forward; returns ``(output, lse)`` (lse is None on fa2)."""
    if _HAS_FA4:
        return _fa4_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
        )
    else:
        left, right = _window(window_size_left, window_size_right)
        out = _fa2_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(left, right),
        )
        return out, None


__all__ = ["flash_attn_varlen_fwd"]
