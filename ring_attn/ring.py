"""Zigzag ring attention on top of CK Tile FMHA.

Rank r holds zigzag-sharded Q/K/V of local length 2c (chunks r and 2W-1-r). At ring step t it
holds the K/V shard of src = (r - t) mod W. With causal masking:

    t == 0      diagonal: local causal attention over the whole shard (exact, because the
                shard's two chunks are in global order and the second never feeds the first)
    src < r     every local query sees src's first chunk and none of its second chunk
                -> attention(Q, KV[:c]), no mask
    src > r     only the local second chunk sees src, and it sees all of it
                -> attention(Q[c:], KV), no mask

so every rank does one causal block plus W-1 equally sized mask-free blocks. Without causal
masking every step is attention(Q, KV) with no mask.

Forward: per-block CK forward with LSE, merged into an fp32 accumulator (merge.py).
Backward: K/V travel the ring again; each block's CK backward uses the global out/lse;
dQ accumulates locally in fp32, and an fp32 dK/dV accumulator travels with its K/V shard,
reaching the owner after one extra hop.
"""

import torch

from .ck_ops import block_bwd, block_fwd
from .comm import RingComm
from .merge import merge_into

# Fault injection for the test's negative control: index of a ring step whose block result
# is dropped (None = no fault).
FAULT_DROP_STEP = None


def _plan(t, rank, world, causal, c):
    """(q_rows, kv_rows, causal) for step t, as slices of the local sequence."""
    full = slice(0, 2 * c)
    if not causal:
        return full, full, False
    if t == 0:
        return full, full, True
    src = (rank - t) % world
    if src < rank:
        return full, slice(0, c), False
    return slice(c, 2 * c), full, False


def ring_fwd(q, k, v, softmax_scale, causal, comm):
    b, s, h, d = q.shape
    assert s % 2 == 0
    c = s // 2
    out_acc = torch.zeros(b, s, h, d, dtype=torch.float32, device=q.device)
    lse_acc = torch.full((b, h, s), float("-inf"), dtype=torch.float32, device=q.device)
    for t in range(comm.world):
        if t + 1 < comm.world:
            nk, nv = comm.start([k, v])
        qs, kvs, blk_causal = _plan(t, comm.rank, comm.world, causal, c)
        out, lse = block_fwd(q[:, qs], k[:, kvs], v[:, kvs], softmax_scale, blk_causal)
        if t != FAULT_DROP_STEP:
            merge_into(out_acc, lse_acc, out, lse, row_start=qs.start)
        if t + 1 < comm.world:
            comm.wait()
            k, v = nk, nv
    return out_acc.to(q.dtype), lse_acc


def ring_bwd(dout, q, k, v, out, lse, softmax_scale, causal, kv_comm, dkv_comm):
    s = q.shape[1]
    c = s // 2
    dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dk_acc = dv_acc = None
    for t in range(kv_comm.world):
        if t + 1 < kv_comm.world:
            nk, nv = kv_comm.start([k, v])
        qs, kvs, blk_causal = _plan(t, kv_comm.rank, kv_comm.world, causal, c)
        dq, dk, dv = block_bwd(
            dout[:, qs], q[:, qs], k[:, kvs], v[:, kvs],
            out[:, qs], lse[:, :, qs].contiguous(), softmax_scale, blk_causal,
        )
        if t == 0:
            dk_acc = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
            dv_acc = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        else:
            # Accumulator of the K/V shard held this step, sent by the previous rank.
            dkv_comm.wait()
            dk_acc, dv_acc = next_dk, next_dv
        if t != FAULT_DROP_STEP:
            dq_acc[:, qs] += dq
            dk_acc[:, kvs] += dk
            dv_acc[:, kvs] += dv
        next_dk, next_dv = dkv_comm.start([dk_acc, dv_acc])
        if t + 1 < kv_comm.world:
            kv_comm.wait()
            k, v = nk, nv
    # After W steps each accumulator sits one hop before its owner; the last exchange
    # (started above) delivers this rank's own dK/dV.
    dkv_comm.wait()
    return dq_acc.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


class _ZigzagRingAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, group):
        comm = RingComm(group)
        out, lse = ring_fwd(q, k, v, softmax_scale, causal, comm)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale, ctx.causal, ctx.group = softmax_scale, causal, group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = ring_bwd(
            dout.contiguous(), q, k, v, out, lse, ctx.softmax_scale, ctx.causal,
            RingComm(ctx.group), RingComm(ctx.group),
        )
        return dq, dk, dv, None, None, None


def zigzag_ring_attn(q, k, v, causal=True, softmax_scale=None, group=None):
    """q: [B, S_local, Hq, D], k/v: [B, S_local, Hkv, D], zigzag-sharded (zigzag.shard)."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _ZigzagRingAttn.apply(q.contiguous(), k.contiguous(), v.contiguous(), softmax_scale, causal, group)
