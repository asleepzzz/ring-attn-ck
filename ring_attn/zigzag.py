"""Zigzag sequence sharding for causal load balance.

The global sequence is cut into 2W equal chunks; rank r owns chunks r and 2W-1-r, stored
back to back. With this layout every non-diagonal ring step is mask-free and costs the same
on every rank (see ring.py).
"""

import torch


def shard(x, rank, world, seq_dim=1):
    chunks = x.chunk(2 * world, dim=seq_dim)
    assert all(c.shape[seq_dim] == chunks[0].shape[seq_dim] for c in chunks), (
        f"sequence length {x.shape[seq_dim]} must be divisible by 2 * world = {2 * world}"
    )
    return torch.cat([chunks[rank], chunks[2 * world - 1 - rank]], dim=seq_dim).contiguous()


def unshard(local_shards, seq_dim=1):
    """Inverse of shard(): local_shards[r] is rank r's shard."""
    world = len(local_shards)
    halves = [s.chunk(2, dim=seq_dim) for s in local_shards]
    ordered = [halves[r][0] for r in range(world)] + [halves[r][1] for r in reversed(range(world))]
    return torch.cat(ordered, dim=seq_dim)
