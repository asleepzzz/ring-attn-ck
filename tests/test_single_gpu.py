"""Single-GPU checks of the building blocks: CK LSE semantics, merge, zigzag layout."""

import pytest
import torch

from ring_attn.ck_ops import block_bwd, block_fwd
from ring_attn.merge import merge_into
from ring_attn.zigzag import shard, unshard

DEV = "cuda"


def ref_attn(q, k, v, scale, causal):
    """fp32 reference returning (out [B,S,H,D], lse [B,H,S]); supports GQA."""
    hq, hk = q.shape[2], k.shape[2]
    k = k.repeat_interleave(hq // hk, dim=2)
    v = v.repeat_interleave(hq // hk, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    if causal:
        sq, sk = s.shape[-2:]
        s = s.masked_fill(torch.ones(sq, sk, dtype=torch.bool, device=s.device).triu(sk - sq + 1), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, dim=-1), v.float())
    return out, lse


def test_zigzag_roundtrip():
    x = torch.arange(2 * 48 * 3).reshape(2, 48, 3)
    for w in (1, 2, 4, 8):
        y = unshard([shard(x, r, w) for r in range(w)])
        assert torch.equal(x, y)
    # rank 0 of 4 owns chunks 0 and 7
    assert torch.equal(shard(x, 0, 4)[:, :6], x[:, 0:6])
    assert torch.equal(shard(x, 0, 4)[:, 6:], x[:, 42:48])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_ck_lse_matches_reference(dtype, causal):
    torch.manual_seed(0)
    q = torch.randn(2, 256, 8, 128, device=DEV, dtype=dtype)
    k = torch.randn(2, 256, 2, 128, device=DEV, dtype=dtype)
    v = torch.randn_like(k)
    scale = 128 ** -0.5
    out, lse = block_fwd(q, k, v, scale, causal)
    ref_out, ref_lse = ref_attn(q, k, v, scale, causal)
    assert lse.shape == ref_lse.shape and lse.dtype == torch.float32
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_merge_of_kv_halves_equals_full(dtype):
    """Attention over [KV0, KV1] == merge(attn(KV0), attn(KV1)): checks merge + LSE units."""
    torch.manual_seed(1)
    q = torch.randn(2, 192, 4, 64, device=DEV, dtype=dtype)
    k = torch.randn(2, 320, 4, 64, device=DEV, dtype=dtype)
    v = torch.randn_like(k)
    scale = 64 ** -0.5
    ref_out, ref_lse = ref_attn(q, k, v, scale, False)

    out_acc = torch.zeros(q.shape, dtype=torch.float32, device=DEV)
    lse_acc = torch.full((2, 4, 192), float("-inf"), device=DEV)
    for ks in (slice(0, 100), slice(100, 320)):  # uneven split on purpose
        o, l = block_fwd(q, k[:, ks], v[:, ks], scale, False)
        merge_into(out_acc, lse_acc, o, l)
    torch.testing.assert_close(lse_acc, ref_lse, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_acc, ref_out, atol=2e-2, rtol=2e-2)

    # Row-offset merge: only the last 64 rows get a third block.
    extra_k = torch.randn(2, 64, 4, 64, device=DEV, dtype=dtype)
    extra_v = torch.randn_like(extra_k)
    o, l = block_fwd(q[:, 128:], extra_k, extra_v, scale, False)
    merge_into(out_acc, lse_acc, o, l, row_start=128)
    ref_tail, _ = ref_attn(q[:, 128:], torch.cat([k, extra_k], 1), torch.cat([v, extra_v], 1), scale, False)
    torch.testing.assert_close(out_acc[:, 128:], ref_tail, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_acc[:, :128], ref_out[:, :128], atol=2e-2, rtol=2e-2)


def test_block_bwd_with_global_lse_sums_to_full_grad():
    """dQ of attn over [KV0, KV1] == sum of block backwards using the global out/lse."""
    torch.manual_seed(2)
    dtype = torch.bfloat16
    q = torch.randn(1, 256, 4, 128, device=DEV, dtype=dtype)
    k = torch.randn(1, 256, 4, 128, device=DEV, dtype=dtype)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    scale = 128 ** -0.5
    out, lse = block_fwd(q, k, v, scale, False)
    dq_full, dk_full, dv_full = block_bwd(do, q, k, v, out, lse, scale, False)

    dq = torch.zeros(q.shape, dtype=torch.float32, device=DEV)
    dks, dvs = [], []
    for ks in (slice(0, 128), slice(128, 256)):
        a, b, c = block_bwd(do, q, k[:, ks], v[:, ks], out, lse, scale, False)
        dq += a.float()
        dks.append(b)
        dvs.append(c)
    torch.testing.assert_close(dq, dq_full.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(torch.cat(dks, 1), dk_full, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(torch.cat(dvs, 1), dv_full, atol=2e-2, rtol=2e-2)
