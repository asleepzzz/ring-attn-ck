"""Online-softmax merge of partial attention results (fp32 accumulator)."""

import torch


def merge_into(out_acc, lse_acc, out_blk, lse_blk, row_start=0):
    """Merge one block result into the fp32 accumulator, in place.

    out_acc: fp32 [B, S, H, D]   lse_acc: fp32 [B, H, S]
    out_blk: [B, S_blk, H, D] (any float dtype), lse_blk: fp32 [B, H, S_blk]
    The block covers accumulator rows [row_start, row_start + S_blk).

    Each block output is normalized by its own softmax denominator exp(lse_blk), so
        lse = log(exp(lse_acc) + exp(lse_blk))
        out = out_acc * exp(lse_acc - lse) + out_blk * exp(lse_blk - lse)
    """
    s = out_blk.shape[1]
    o = out_acc[:, row_start:row_start + s]
    l = lse_acc[:, :, row_start:row_start + s]
    new_l = torch.logaddexp(l, lse_blk)
    # [B, H, S] -> [B, S, H, 1] to broadcast against [B, S, H, D]
    w_old = torch.exp(l - new_l).transpose(1, 2).unsqueeze(-1)
    w_blk = torch.exp(lse_blk - new_l).transpose(1, 2).unsqueeze(-1)
    o.mul_(w_old).add_(out_blk.float() * w_blk)
    l.copy_(new_l)
