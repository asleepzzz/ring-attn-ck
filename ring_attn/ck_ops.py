"""Thin wrappers around the CK Tile FMHA bindings (ck_tile_fmha) used by every ring step.

Each wrapper launches exactly one CK forward or backward call on a (Q block, KV block) pair.
Layouts follow ck_tile_fmha / flash-attn: q, k, v, out are [B, S, H, D]; lse is fp32 [B, H, S_q]
in natural-log units.
"""

import torch
from ck_tile_fmha.interface import _wrapped_fmha_backward, _wrapped_fmha_forward


def block_fwd(q, k, v, softmax_scale, causal):
    out, lse, _, _ = _wrapped_fmha_forward(
        q, k, v,
        0.0,            # dropout_p: ring attention here is dropout-free
        softmax_scale,
        causal,
        -1, 0 if causal else -1,  # window (left, right): causal = (-1, 0), full = (-1, -1)
        0.0,            # softcap
        None,           # alibi_slopes
        False,          # return_softmax
        return_lse=True,
    )
    return out, lse


def block_bwd(dout, q, k, v, out, lse, softmax_scale, causal):
    """Gradients of one block, evaluated with the *global* out and lse of the full ring.

    Passing the global (out, lse) makes the kernel produce this block's exact share of
    dQ/dK/dV: P = exp(S - lse_global) and D = rowsum(dO * O_global) are what the
    full-sequence backward would use for these rows and columns.
    """
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _wrapped_fmha_backward(
        dout, q, k, v, out, lse,
        dq, dk, dv,
        0.0,            # dropout_p
        softmax_scale,
        causal,
        -1, 0 if causal else -1,
        0.0,            # softcap
        None,           # alibi_slopes
        False,          # deterministic
    )
    return dq, dk, dv
