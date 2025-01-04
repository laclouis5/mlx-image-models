from typing import Callable

import mlx.core as mx
from mlx import nn

from .helpers import make_divisible


class RadixSoftmax(nn.Module):
    def __init__(self, radix: int, cardinality: int):
        super().__init__()

        self.radix = radix
        self.cardinality = cardinality

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        # (B, H, W, r*O)
        if self.radix > 1:
            # (B, H, W, c, r, O/c)
            x = x.reshape(b, h, w, self.cardinality, self.radix, -1)
            # (B, H, W, r, c, O/c)
            x = x.transpose(0, 1, 2, 4, 3, 5)
            # (B, H, W, r, c, O/c)
            x = mx.softmax(x, axis=3)
            # (B, H*W*c*r*O)
            x = x.reshape(b, h, w, -1)
        else:
            # (B, H, W, r*O)
            x = mx.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        radix: int = 2,
        rd_ratio: float = 0.25,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[[], nn.Module] | None = None,
        **kwargs,
    ):
        super().__init__()

        out_channels = out_channels if out_channels is not None else in_channels
        self.radix = radix
        mid_chs = out_channels * radix

        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor
            )
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding

        self.conv = nn.Conv2d(
            in_channels,
            mid_chs,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
            **kwargs,
        )

        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def __call__(self, x: mx.array) -> mx.array:
        # (B, H, W, C)
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)
        # (B, H, W, r*O)

        B, H, W, RO = x.shape

        if self.radix > 1:
            # (B, H, W, r, O)
            x = x.reshape((B, H, W, self.radix, RO // self.radix))

            # (B, H, W, O)
            x_gap = x.sum(axis=3)
        else:
            # (B, H, W, r*O) (r=1)
            x_gap = x

        # (B, 1, 1, r*O)
        x_gap = x_gap.mean(axis=(1, 2), keepdims=True)

        # (B, 1, 1, r*O)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        # (B, 1, 1, r*O)

        # (B, 1, 1, r*O)
        x_attn = self.rsoftmax(x_attn)

        if self.radix > 1:
            # (B, H, W, r, O) * (B, 1, 1, r, O) -> (B, H, W, r, O)
            x_attn = x_attn.reshape((B, 1, 1, self.radix, -1))

            # (B, H, W, O)
            out = (x * x_attn).sum(axis=3)

        else:
            # (B, H, W, r*O) * (B, 1, 1, r*O) -> (B, H, W, r*O) (r=1)
            out = x * x_attn

        # (B, H, W, O)
        return out
