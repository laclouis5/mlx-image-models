import mlx.core as mx
from mlx import nn

from .grid import ndgrid
from .helpers import make_divisible, to_2tuple


def rel_pos_indices(size: tuple[int, ...]) -> mx.array:
    size = to_2tuple(size)
    pos = mx.stack(ndgrid(mx.arange(size[0]), mx.arange(size[1]))).flatten(1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos  # (2, H * W, H * W)


class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        feat_size: int | None = None,
        stride: int = 1,
        num_heads: int = 4,
        dim_head: int = 16,
        r: int | None = 9,
        qk_ratio: float = 1.0,
        qkv_bias: bool = False,
    ):
        super().__init__()

        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, " should be divided by num_heads"

        self.dim_qk = (
            dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        )
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads

        self.qkv = nn.Conv2d(
            dim,
            num_heads * self.dim_qk + self.dim_qk + self.dim_v,
            kernel_size=1,
            bias=qkv_bias,
        )

        self.norm_q = nn.BatchNorm(num_heads * self.dim_qk)
        self.norm_v = nn.BatchNorm(self.dim_v)

        if r is not None:
            # local lambda convolution for pos
            self.conv_lambda = nn.Conv3d(
                1, self.dim_qk, (r, r, 1), padding=(r // 2, r // 2, 0)
            )
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            # relative pos embedding
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [2 * s - 1 for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = mx.zeros((rel_size[0], rel_size[1], self.dim_qk))
            self._rel_pos_indices = rel_pos_indices(feat_size)

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        M = H * W

        # (B, H, W, Dqk)
        qkv = self.qkv(x)

        q, k, v = mx.split(
            qkv,
            (self.num_heads * self.dim_qk, self.num_heads * self.dim_qk + self.dim_qk),
            axis=-1,
        )

        # (B, H, W, Oqk) -> (B, N, H*W, Dqk)
        q = self.norm_q(q)
        q = q.reshape(B, M, self.num_heads, self.dim_qk)
        q = q.transpose(0, 2, 1, 3)

        # (B, H, W, Dqk) -> (B, Dqk, H*W)
        k = k.reshape(B, M, self.dim_qk).transpose(0, 2, 1)
        k = mx.softmax(k, axis=-1)

        # (B, H, W, Dv) -> (B, H*W, Dv)
        v = self.norm_v(v)
        v = v.reshape(B, M, self.dim_v)

        # (B, Dqk, H*W) @ (B, H*W, Dv) -> (B, Dqk, Dv)
        content_lam = k @ v

        # (B, N, H*W, Dqk) @ (B, 1, Dqk, Dv) -> (B, N, H*W, Dv)
        content_out = q @ content_lam.reshape(B, 1, self.dim_qk, self.dim_v)

        if self.pos_emb is None:
            # (B, H*W, Dv) -> (B, H, W, Dv, 1) -> (B, H, W, Dv, Dqk)
            position_lam = self.conv_lambda(v.reshape(B, H, W, self.dim_v, 1))

            # (B, H, W, Dv, Dqk) -> (B, H, W, Dqk, Dv)
            position_lam = position_lam.transpose(0, 1, 2, 4, 3)

            # (B, H, W, Dqk, Dv) -> (B, 1, H*W, Dqk, Dv)
            position_lam = position_lam.reshape(B, 1, M, self.dim_qk, self.dim_v)
        else:
            # (H*W, H*W, Dqk)
            pos_emb = self.pos_emb[
                self._rel_pos_indices[0], self._rel_pos_indices[1], :
            ]

            # (B, H*W, H*W, Dqk)
            pos_emb = mx.broadcast_to(pos_emb, shape=(B, M, M, self.dim_qk))

            # (B, H*W, Dqk, H*W) @ (B, 1, H*W, Dv) -> (B, H*W, Dqk, Dv)
            position_lam = pos_emb.transpose(0, 1, 3, 2) @ v.reshape(
                B, 1, M, self.dim_v
            )

            # (B, 1, H*W, Dqk, Dv)
            position_lam = position_lam.reshape(B, 1, M, self.dim_qk, self.dim_v)

        # (B, N, H*W, 1, Dqk) @ (B, 1, H*W, Dqk, Dv) -> (B, N, H*W, Dv)
        q = q.reshape(B, self.num_heads, M, 1, self.dim_qk)
        position_out = (q @ position_lam).squeeze(-2)

        # (B, N, H*W, Dv) -> (B, H, W, Ov)
        out = (content_out + position_out).reshape(B, self.num_heads, H, W, self.dim_v)
        out = out.transpose(0, 2, 3, 1, 4)
        out = out.reshape(B, H, W, self.num_heads * self.dim_v)

        return self.pool(out)
