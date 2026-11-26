# ring-attn-ck

Zigzag ring attention (forward + backward) built on CK Tile FMHA (`ck_tile_fmha`), phase 1:
one CK call per ring step, merging in PyTorch, communication through RCCL P2P.

| file | role |
|---|---|
| `ring_attn/ck_ops.py` | one CK fwd (with LSE) / bwd call on a (Q block, KV block) pair |
| `ring_attn/merge.py` | online-softmax merge into an fp32 (out, lse) accumulator |
| `ring_attn/zigzag.py` | 2W-chunk zigzag sharding (rank r owns chunks r and 2W-1-r) |
| `ring_attn/comm.py` | `batch_isend_irecv` ring exchange, overlapped with compute |
| `ring_attn/ring.py` | ring forward/backward + `zigzag_ring_attn` autograd function |

Causal step plan on rank r, step t, src = (r - t) mod W: t=0 local causal; src<r
`attn(Q, KV[:c])`; src>r `attn(Q[c:], KV)`; both mask-free and the same size on every rank.
Backward: dQ accumulates locally in fp32; an fp32 dK/dV accumulator travels with its K/V
shard and reaches the owner after one extra hop.

## Environment

Docker image `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0`,
8x MI300X / gfx942. `ck_tile_fmha` is built from
[ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) at `b99c5e7`:

```bash
cd composable_kernel
git apply /path/to/ring-attn-ck/patches/ck-python-binding-hipify-v2.patch
cd example/ck_tile/01_fmha/python
GPU_ARCHS=gfx942 MAX_JOBS=128 python3 -m pip install --no-build-isolation .
```

The patch is needed with torch 2.14 (hipify v2): the `ATen/cuda/*` includes in
`python/csrc/fmha_common.hpp` are not rewritten, so the header includes the
`ATen/hip`/`c10/hip` headers directly under `USE_ROCM && HIPIFY_V2`.

## Validation

```bash
python3 -m pytest -q tests/test_single_gpu.py          # building blocks
torchrun --nproc_per_node=W tests/check_ring.py       # W = 2, 4, 8
torchrun --nproc_per_node=4 tests/check_ring.py --fault-step 0   # negative control, must FAIL
```

`check_ring.py` compares gathered out/dQ/dK/dV against an fp32 PyTorch reference and passes
when the ring error is at most 2x the single-GPU CK error (+1e-4).
Cases: bf16/fp16, causal/full, GQA 8:2 and 8:1, MHA d64, chunk length 200 (not a tile multiple),
S up to 16384.

Results (2026-11-25, gfx942): 8/8 single-GPU tests; 6/6 cases at W=2, 4 and 8;
fault injection at step 0 or 3: 0/6 (exit 1), as expected.
Out and dQ errors match single-GPU CK; dK/dV are up to ~1.5x (bf16), because each block's
dK/dV is rounded to bf16 by CK before the fp32 sum.

## Performance and report

```bash
torchrun --nproc_per_node=W bench/bench_ring.py --out results/bench.jsonl   # W = 2, 4, 8
python3 bench/make_report.py results/bench.jsonl --validation results/validation.jsonl -o report.html
```

`bench_ring.py` times the full ring, the ring without communication, the CK calls alone,
the K/V exchange alone, and single-GPU CK on the whole sequence (bf16, causal, Hq=32, Hkv=8,
D=128). `report.html` explains the method and shows these numbers.
**Status:** performance has not been measured yet (the GPUs were busy); the current
`report.html` holds the explanation and the correctness results only.
`results/validation.jsonl` is parsed from the logs in `results/logs/` by `bench/parse_check_logs.py`.

## Not covered yet

dropout, varlen (group mode), sliding window, softcap/alibi; performance (no benchmark yet);
SDMA transport; kernel-level fp32 accumulation (fwd o_acc / bwd dq_acc across steps).
