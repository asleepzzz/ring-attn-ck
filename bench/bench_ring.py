"""Ring attention performance on W GPUs.

torchrun --nproc_per_node=W bench/bench_ring.py --seqlens 16384,32768 --out results.jsonl

Per configuration (bf16, causal, B=1, Hq=32, Hkv=8, D=128 by default) it times, as the max over
ranks of the median of --iters runs:
  ring        full ring (RCCL P2P overlapped with compute, fp32 merge)
  nocomm      same loop with communication removed (each step reuses local K/V)
  kernels     nocomm and no merge/accumulate: only the CK calls of the plan
  comm        the W-1 K/V exchanges alone
and once on rank 0, the single-GPU CK fmha on the full sequence (single).
Forward (fwd) and backward (bwd) are timed separately; bwd uses a saved forward.
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ring_attn.ring as ring_mod  # noqa: E402
from ring_attn.ck_ops import block_bwd, block_fwd  # noqa: E402
from ring_attn.comm import RingComm  # noqa: E402


class NullComm:
    """Same interface as RingComm, no transfer: start() hands back the same tensors."""

    def __init__(self, rank, world):
        self.rank, self.world = rank, world

    def start(self, tensors):
        return list(tensors)

    def wait(self):
        pass


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    med = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(med, op=dist.ReduceOp.MAX)
    return med.item()


def kernels_only(q, k, v, dout, out, lse, scale, causal, rank, world, bwd):
    c = q.shape[1] // 2
    for t in range(world):
        qs, kvs, cz = ring_mod._plan(t, rank, world, causal, c)
        if bwd:
            block_bwd(dout[:, qs], q[:, qs], k[:, kvs], v[:, kvs], out[:, qs],
                      lse[:, :, qs].contiguous(), scale, cz)
        else:
            block_fwd(q[:, qs], k[:, kvs], v[:, kvs], scale, cz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="16384,32768,65536,131072")
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    dtype, causal, b = torch.bfloat16, True, 1
    scale = args.d ** -0.5

    for s in [int(x) for x in args.seqlens.split(",")]:
        sl = s // world
        g = torch.Generator(device=dev).manual_seed(rank)
        q = torch.randn(b, sl, args.hq, args.d, device=dev, dtype=dtype, generator=g)
        k = torch.randn(b, sl, args.hkv, args.d, device=dev, dtype=dtype, generator=g)
        v = torch.randn_like(k)
        dout = torch.randn_like(q)
        ring_comm, null = RingComm(), NullComm(rank, world)
        out, lse = ring_mod.ring_fwd(q, k, v, scale, causal, ring_comm)

        r = dict(world=world, seqlen=s, local=sl, hq=args.hq, hkv=args.hkv, d=args.d, batch=b)
        r["fwd_ring"] = timeit(lambda: ring_mod.ring_fwd(q, k, v, scale, causal, ring_comm), args.iters, args.warmup)
        r["fwd_nocomm"] = timeit(lambda: ring_mod.ring_fwd(q, k, v, scale, causal, null), args.iters, args.warmup)
        r["fwd_kernels"] = timeit(lambda: kernels_only(q, k, v, dout, out, lse, scale, causal, rank, world, False), args.iters, args.warmup)
        r["bwd_ring"] = timeit(lambda: ring_mod.ring_bwd(dout, q, k, v, out, lse, scale, causal, RingComm(), RingComm()), args.iters, args.warmup)
        r["bwd_nocomm"] = timeit(lambda: ring_mod.ring_bwd(dout, q, k, v, out, lse, scale, causal, null, null), args.iters, args.warmup)
        r["bwd_kernels"] = timeit(lambda: kernels_only(q, k, v, dout, out, lse, scale, causal, rank, world, True), args.iters, args.warmup)

        def comm_only():
            kk, vv = k, v
            for _ in range(world - 1):
                nk, nv = ring_comm.start([kk, vv])
                ring_comm.wait()
                kk, vv = nk, nv
        r["comm_kv"] = timeit(comm_only, args.iters, args.warmup)

        if rank == 0:
            # Single GPU, full sequence: the work the ring splits W ways.
            gq = torch.randn(b, s, args.hq, args.d, device=dev, dtype=dtype)
            gk = torch.randn(b, s, args.hkv, args.d, device=dev, dtype=dtype)
            gv, gdo = torch.randn_like(gk), torch.randn_like(gq)
            go, gl = block_fwd(gq, gk, gv, scale, True)
            ts = []
            for fn in (lambda: block_fwd(gq, gk, gv, scale, True),
                       lambda: block_bwd(gdo, gq, gk, gv, go, gl, scale, True)):
                for _ in range(3):
                    fn()
                tt = []
                for _ in range(max(5, args.iters // 2)):
                    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); fn(); e.record(); torch.cuda.synchronize()
                    tt.append(a.elapsed_time(e))
                ts.append(sorted(tt)[len(tt) // 2])
            r["fwd_single"], r["bwd_single"] = ts
            del gq, gk, gv, gdo, go, gl
            flops_fwd = 4 * b * args.hq * s * s * args.d / 2  # causal
            r["fwd_ring_tflops"] = flops_fwd / r["fwd_ring"] / 1e9
            r["bwd_ring_tflops"] = 2.5 * flops_fwd / r["bwd_ring"] / 1e9
            r["fwd_single_tflops"] = flops_fwd / r["fwd_single"] / 1e9
            r["bwd_single_tflops"] = 2.5 * flops_fwd / r["bwd_single"] / 1e9
            print(json.dumps(r), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        dist.barrier()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
