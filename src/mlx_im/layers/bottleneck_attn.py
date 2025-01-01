from typing import List

import mlx.core as mx
from mlx import nn

from .helpers import make_divisible, to_2tuple
from .trace_utils import _assert


def rel_logits_1d(q: mx.array, rel_k: mx.array, permute_mask: List[int]):
    B, H, W, _ = q.shape

    x = q @ rel_k.transpose(1, 0)
    x = x.reshape(-1, W, 2 * W - 1)
    x_pad = mx.pad(x, pad_width=[(0, 0), (0, 0), (0, 1)]).flatten(1)
    x_pad = mx.pad(x_pad, [(0, 0), (0, W - 1)])
    x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x = x_pad[:, :W, W - 1 :]

    x = x.reshape(B, H, 1, W, W)
    x = mx.broadcast_to(x, (B, H, H, W, W))

    return x.transpose(permute_mask)


class PosEmbedRel(nn.Module):
    def __init__(self, feat_size: int | tuple[int, int], dim_head: int, scale: float):
        super().__init__()
        self.height, self.width = to_2tuple(feat_size)
        self.dim_head = dim_head

        self.height_rel = mx.random.normal((self.height * 2 - 1, dim_head)) * scale
        self.width_rel = mx.random.normal((self.width * 2 - 1, dim_head)) * scale

    # (B, H*W, C)
    def __call__(self, q: mx.array) -> mx.array:
        B, HW, _ = q.shape

        # (B, H, W, C) -> (B, H, W, H, W)
        q = q.reshape(B, self.height, self.width, -1)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # (B, W, H, C) -> (B, H, W, H, W)
        q = q.transpose(0, 2, 1, 3)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        # (B, H, W, H, W) -> (B, HW, HW)
        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, HW, HW)
        return rel_logits


class BottleneckAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        feat_size: int | tuple[int, int] | None = None,
        stride: int = 1,
        num_heads: int = 4,
        dim_head: int | None = None,
        qk_ratio: float = 1.0,
        qkv_bias: bool = False,
        scale_pos_embed: bool = False,
    ):
        super().__init__()
        assert (
            feat_size is not None
        ), "A concrete feature size matching expected input (H, W) is required"

        dim_out = dim_out or dim  # O
        assert dim_out % num_heads == 0

        self.num_heads = num_heads  # N
        self.dim_head_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )  # O1/N
        self.dim_head_v = dim_out // self.num_heads  # O2/N
        self.dim_out_qk = num_heads * self.dim_head_qk  # O1
        self.dim_out_v = num_heads * self.dim_head_v  # O2
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed

        # D -> 2*O1 + O2
        self.qkv = nn.Conv2d(
            dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias
        )

        self.pos_embed = PosEmbedRel(
            feat_size, dim_head=self.dim_head_qk, scale=self.scale
        )

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape

        _assert(H == self.pos_embed.height, "")
        _assert(W == self.pos_embed.width, "")

        # (B, H, W, 2*O1+O2)
        x = self.qkv(x)

        q, k, v = mx.split(x, [self.dim_out_qk, 2 * self.dim_out_qk], axis=-1)

        # (B, H, W, N*O1) -> (B, H*W, N, O1/N) -> (B, N, H*W, O1/N)
        q = q.reshape(B, -1, self.num_heads, self.dim_head_qk).transpose(0, 2, 1, 3)

        # (B, H, W, N*O1) -> (B, H*W, N, O1/N) -> (B, N, H*W, O1/N)
        k = k.reshape(B, -1, self.num_heads, self.dim_head_qk).transpose(0, 2, 1, 3)

        # (B, H, W, N*O2) -> (B, H*W, N, O2/N) -> (B, N, H*W, O2/N)
        v = v.reshape(B, -1, self.num_heads, self.dim_head_v).transpose(0, 2, 1, 3)

        # (B*N, H*W, 01/N)
        pe = self.pos_embed(q.reshape(B * self.num_heads, H * W, -1))

        if self.scale_pos_embed:
            # (B, N, H*W, O2/N)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=pe * self.scale
            )
        else:
            # (B, N, H*W, O2/N)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=pe
            )

        # (B, N, H*W, O2/N)
        out = out.transpose(0, 2, 1, 3).reshape(B, H, W, -1)

        # (B, H/2, W/2, O2)
        return self.pool(out)
