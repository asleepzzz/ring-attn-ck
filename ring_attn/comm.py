"""Ring point-to-point exchange (RCCL batch_isend_irecv).

start() posts "send to next, receive from previous" for a list of tensors and returns
immediately; the receive buffers are filled asynchronously on RCCL's stream. wait() makes
the current stream wait for completion, so compute launched between start() and wait()
overlaps with the transfer.
"""

import torch
import torch.distributed as dist


class RingComm:
    def __init__(self, group=None):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        # P2POp takes global ranks.
        self.next = dist.get_global_rank(group, (self.rank + 1) % self.world) if group else (self.rank + 1) % self.world
        self.prev = dist.get_global_rank(group, (self.rank - 1) % self.world) if group else (self.rank - 1) % self.world
        self._reqs = None

    def start(self, tensors):
        assert self._reqs is None, "previous exchange not waited"
        recv = [torch.empty_like(t) for t in tensors]
        ops = []
        for s, r in zip(tensors, recv):
            ops.append(dist.P2POp(dist.isend, s, self.next, self.group))
            ops.append(dist.P2POp(dist.irecv, r, self.prev, self.group))
        self._reqs = dist.batch_isend_irecv(ops)
        return recv

    def wait(self):
        for req in self._reqs:
            req.wait()
        self._reqs = None
