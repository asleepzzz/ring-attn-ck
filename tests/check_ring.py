"""Multi-GPU correctness check of zigzag ring attention.

Run:  torchrun --nproc_per_node=W tests/check_ring.py [--fault-step T]

Every rank builds the same full Q/K/V/dO (same seed), runs ring attention on its zigzag shard,
and the gathered out/dQ/dK/dV are compared on rank 0 against
  * ref32: fp32 PyTorch attention + autograd on the full sequence (ground truth), and
  * ck1:   single-GPU CK fmha_func on the full sequence (the error budget).
A case passes when, for each tensor, max|ring - ref32| <= 2 * max|ck1 - ref32| + 1e-4,
i.e. sharding the sequence over the ring costs at most 2x the error CK already has.

--fault-step T drops ring step T (forward and backward) as a negative control: the check
must then FAIL, which shows the comparison is sensitive to a missing block.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ring_attn.ring as ring_mod  # noqa: E402
from ck_tile_fmha import fmha_func  # noqa: E402
from ring_attn.ring import zigzag_ring_attn  # noqa: E402
from ring_attn.zigzag import shard, unshard  # noqa: E402

CASES = [
    # name,               dtype,          causal, B, chunk, Hq, Hkv, D
    ("bf16 causal gqa",   torch.bfloat16, True,  2, 256, 8, 2, 128),
    ("bf16 full gqa",     torch.bfloat16, False, 2, 256, 8, 2, 128),
    ("fp16 causal gqa",   torch.float16,  True,  2, 256, 8, 2, 128),
    ("fp16 full gqa",     torch.float16,  False, 2, 256, 8, 2, 128),
    ("bf16 causal mha d64 chunk200", torch.bfloat16, True, 1, 200, 8, 8, 64),
    ("bf16 causal gqa long",         torch.bfloat16, True, 1, 1024, 8, 1, 128),
]


RECORDS = []


def ref_fp32(q, k, v, dout, scale, causal):
    q, k, v = [x.detach().float().requires_grad_() for x in (q, k, v)]
    rep = q.shape[2] // k.shape[2]
    kk, vv = k.repeat_interleave(rep, 2), v.repeat_interleave(rep, 2)
    s = torch.einsum("bqhd,bkhd->bhqk", q, kk) * scale
    if causal:
        n = s.shape[-1]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vv)
    out.backward(dout.float())
    return out.detach(), q.grad, k.grad, v.grad


def ck_single(q, k, v, dout, causal):
    q, k, v = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    out = fmha_func(q, k, v, causal=causal)
    out.backward(dout)
    return out.detach(), q.grad, k.grad, v.grad


def gather_unshard(x, world):
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous())
    return unshard(parts)


def run_case(case, rank, world, dev):
    name, dtype, causal, b, chunk, hq, hkv, d = case
    s = 2 * world * chunk
    scale = d ** -0.5
    g = torch.Generator(device=dev).manual_seed(1234)
    q = torch.randn(b, s, hq, d, device=dev, dtype=dtype, generator=g)
    k = torch.randn(b, s, hkv, d, device=dev, dtype=dtype, generator=g)
    v = torch.randn(b, s, hkv, d, device=dev, dtype=dtype, generator=g)
    dout = torch.randn(b, s, hq, d, device=dev, dtype=dtype, generator=g)

    ql, kl, vl = [shard(x, rank, world).requires_grad_() for x in (q, k, v)]
    out_l = zigzag_ring_attn(ql, kl, vl, causal=causal, softmax_scale=scale)
    out_l.backward(shard(dout, rank, world))
    ring = [gather_unshard(t, world) for t in (out_l.detach(), ql.grad, kl.grad, vl.grad)]

    ok = torch.tensor(1, device=dev)
    if rank == 0:
        ref = ref_fp32(q, k, v, dout, scale, causal)
        ck1 = ck_single(q, k, v, dout, causal)
        lines, errs = [], {}
        for tname, r, c, t in zip(("out", "dq", "dk", "dv"), ring, ck1, ref):
            e_ring = (r.float() - t).abs().max().item()
            e_ck = (c.float() - t).abs().max().item()
            e_rc = (r.float() - c.float()).abs().max().item()
            good = e_ring <= 2 * e_ck + 1e-4
            ok &= int(good)
            errs[tname] = dict(ring=e_ring, ck1=e_ck, ring_vs_ck1=e_rc, ok=good)
            lines.append(f"    {tname:3s} ring-vs-fp32 {e_ring:.3e}  ck1-vs-fp32 {e_ck:.3e}  "
                         f"ring-vs-ck1 {e_rc:.3e}  {'ok' if good else 'FAIL'}")
        print(f"[W={world}] {name} (S={s}, B={b}, Hq={hq}, Hkv={hkv}, D={d}): "
              f"{'PASS' if ok.item() else 'FAIL'}", flush=True)
        print("\n".join(lines), flush=True)
        RECORDS.append(dict(world=world, case=name, seqlen=s, batch=b, hq=hq, hkv=hkv, d=d,
                            causal=causal, dtype=str(dtype).split(".")[-1], passed=bool(ok.item()), errors=errs))
    dist.broadcast(ok, 0)
    return bool(ok.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault-step", type=int, default=None)
    ap.add_argument("--json", default=None, help="append per-case results (rank 0) to this JSONL file")
    args = ap.parse_args()
    ring_mod.FAULT_DROP_STEP = args.fault_step

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)

    results = [run_case(c, rank, world, dev) for c in CASES]
    if rank == 0:
        print(f"[W={world}] {sum(results)}/{len(results)} cases passed"
              + (f" (fault injected at step {args.fault_step})" if args.fault_step is not None else ""),
              flush=True)
        if args.json:
            import json
            with open(args.json, "a") as f:
                for r in RECORDS:
                    r["fault_step"] = args.fault_step
                    f.write(json.dumps(r) + "\n")
    dist.destroy_process_group()
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
