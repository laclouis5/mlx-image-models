from typing import List

import mlx.core as mx
from mlx import nn

from .helpers import make_divisible
from .mlx_layers import sliding_windows
from .trace_utils import _assert


# (B, Bs, Bs, D), (2*Ws-1, D) -> (B, Bs, Ws, Bs, Ws)
def rel_logits_1d(q: mx.array, rel_k: mx.array, permute_mask: List[int]) -> mx.array:
    B, H, W, _ = q.shape
    bh = rel_k.shape[0]
    b = (bh + 1) // 2

    x = q @ rel_k.transpose(1, 0)
    x = x.reshape(-1, W, bh)
    x_pad = mx.pad(x, pad_width=[(0, 0), (0, 0), (0, 1)]).flatten(1)
    x_pad = mx.pad(x_pad, [(0, 0), (0, bh - W)])
    x_pad = x_pad.reshape(-1, W + 1, bh)
    x = x_pad[:, :W, b - 1 :]

    x = x.reshape(B, H, 1, W, b)
    x = mx.broadcast_to(x, (B, H, b, W, b))

    return x.transpose(permute_mask)


class PosEmbedRel(nn.Module):
    def __init__(self, block_size: int, win_size: int, dim_head: int, scale: float):
        super().__init__()

        self.block_size = block_size
        self.dim_head = dim_head

        self.height_rel = mx.random.normal((win_size * 2 - 1, dim_head)) * scale
        self.width_rel = mx.random.normal((win_size * 2 - 1, dim_head)) * scale

    def __call__(self, q: mx.array) -> mx.array:
        B, BB, HW, _ = q.shape  # (B*N, Nb, Bs*Bs, Dqk)

        # (B*N*Nb, Bs, Bs, Dqk)
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)

        # (B*N*Nb, Bs, Bs, Ws, Ws)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # (B*N*Nb, Bs, Bs, Dqk)
        q = q.transpose(0, 2, 1, 3)

        # (B*N*Nb, Bs, Bs, Ws, Ws)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        # (B*N*Nb, Bs, Bs, Ws, Ws)
        rel_logits = rel_logits_h + rel_logits_w

        # (B*N, Nb, Bs*Bs, Ws*Ws)
        return rel_logits.reshape(B, BB, HW, -1)


class HaloAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        feat_size: int | None = None,
        stride: int = 1,
        num_heads: int = 8,
        dim_head: int | None = None,
        block_size: int = 8,
        halo_size: int = 3,
        qk_ratio: int = 1.0,
        qkv_bias: bool = False,
        avg_down: bool = False,
        scale_pos_embed: bool = False,
    ):
        super().__init__()

        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)

        self.num_heads = num_heads
        self.dim_head_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2
        self.block_stride = 1
        use_avg_pool = False

        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride

        self.q = nn.Conv2d(
            dim, self.dim_out_qk, 1, stride=self.block_stride, bias=qkv_bias
        )

        self.kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, bias=qkv_bias)

        self.pos_embed = PosEmbedRel(
            block_size=self.block_size_ds,
            win_size=self.win_size,
            dim_head=self.dim_head_qk,
            scale=self.scale,
        )

        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()

    # FIXME: Use fast attn.
    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape

        _assert(H % self.block_size == 0, "")
        _assert(W % self.block_size == 0, "")

        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size

        num_blocks = num_h_blocks * num_w_blocks

        # (B, H, W, C) -> (B, H, W, Oqk)
        q = self.q(x)

        # (B, Nbh, Bs, Nbw, Bs, N, Dqk)
        q = q.reshape(
            B,
            num_h_blocks,
            self.block_size_ds,
            num_w_blocks,
            self.block_size_ds,
            self.num_heads,
            self.dim_head_qk,
        )

        # (B, N, Nbh, Nbw, Bs, Bs, Dqk)
        q = q.transpose(0, 5, 1, 3, 2, 4, 6)

        # (B*N, Nb, Bs*Bs, Dqk)
        q = q.reshape(B * self.num_heads, num_blocks, -1, self.dim_head_qk)

        # (B, C, H, W) -> (B, Oqk+Ov, H, W)
        kv = self.kv(x)

        # (B, H+2*Hs, W+2*Hs, Oqk+Ov)
        kv = mx.pad(
            kv,
            [
                (0, 0),
                (self.halo_size, self.halo_size),
                (self.halo_size, self.halo_size),
                (0, 0),
            ],
        )

        # (B, Nbh, Nbw, Ws*Ws, Oqk+Ov)
        kv = sliding_windows(
            kv,
            window_shape=(self.win_size, self.win_size),
            window_strides=(self.block_size, self.block_size),
        )

        # (B*N, Dqk+Dv, Nbh*Nbw, Ws*Ws)
        kv = kv.transpose(0, -1, 1, 2, 3, 4)
        kv = kv.reshape(
            B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1
        )

        # (B*N, Nbh*Nbw, Ws*Ws, Dqk+Dv)
        kv = kv.transpose(0, 2, 3, 1)

        # (B*N, Nbh*Nbw, Ws*Ws, Dqk), (B*N, Nbh*Nbw, Ws*Ws, Dv)
        _kv: list[mx.array] = mx.split(kv, (self.dim_head_qk,), axis=-1)
        k, v = _kv

        k = k.transpose(0, 1, 3, 2)

        # (B*N, Nb, Bs*Bs, Dqk) @ (B*N, Nbh*Nbw, Dqk, Ws*Ws) -> (B*N, Nb, Bs*Bs, Ws*Ws)
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale
        else:
            attn = (q @ k) * self.scale + self.pos_embed(q)

        # (B*N, Nb, Bs*Bs, Ws*Ws)
        attn = mx.softmax(attn, axis=-1)

        # (B*N, Nb, Bs*Bs, Ws*Ws) @ (B*N, Nbh*Nbw, Ws*Ws, Dv) -> (B*N, Dv, Bs*Bs, Nb)
        out = (attn @ v).transpose(0, 3, 2, 1)

        # (B, Ov, Bs, Bs, Nbh, Nbw)
        out = out.reshape(
            B,
            self.dim_out_v,
            self.block_size_ds,
            self.block_size_ds,
            num_h_blocks,
            num_w_blocks,
        )

        # (B, Nbh, Bs, Nbw, Bs, Ov)
        out = out.transpose(0, 4, 2, 5, 3, 1)

        # (B, H, W, Ov)
        out = out.reshape(
            B, H // self.block_stride, W // self.block_stride, self.dim_out_v
        )

        return self.pool(out)
