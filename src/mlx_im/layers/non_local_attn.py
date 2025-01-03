from typing import Callable

import mlx.core as mx
from mlx import nn

from . import mlx_layers as L
from .conv_bn_act import ConvNormAct
from .helpers import make_divisible
from .trace_utils import _assert


class NonLocalAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        use_scale: bool = True,
        rd_ratio: float = 1 / 8,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        **kwargs,
    ):
        super().__init__()

        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)

        self.scale = in_channels**-0.5 if use_scale else 1.0

        self.t = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv2d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True)

        self.norm = nn.BatchNorm(in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x  # (B, H, W, C)

        # (B, H, W, R)
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, H, W, R = t.shape

        # (B, 1, H*W, R)
        t = t.reshape(B, 1, R, -1)
        p = p.reshape(B, 1, R, -1)
        g = g.reshape(B, 1, R, -1)

        # (B, H*W, R)
        x = mx.fast.scaled_dot_product_attention(t, p, g, scale=self.scale).squeeze(1)

        # (B, H, W, R)
        x = x.reshape(B, H, W, R)

        # (B, H, W, C)
        x = self.z(x)

        return self.norm(x) + shortcut


class BilinearAttnTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block_size: int,
        groups: int,
        act_layer: str | Callable[[], nn.Module] | None = nn.ReLU,
        norm_layer: str | Callable[[int], nn.Module] | None = nn.BatchNorm,
    ):
        super().__init__()

        self.conv1 = ConvNormAct(
            in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.conv_p = nn.Conv2d(
            groups, block_size * block_size * groups, kernel_size=(block_size, 1)
        )
        self.conv_q = nn.Conv2d(
            groups, block_size * block_size * groups, kernel_size=(1, block_size)
        )
        self.conv2 = ConvNormAct(
            in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer
        )

        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def _resize_mat(self, x: mx.array, t: int) -> mx.array:
        B, C, H, W = x.shape
        _assert(H == W, "")

        if t <= 1:
            return x

        x = x.reshape(B, C, H, W, 1, 1)

        # (B, C, H, W, t, t)
        x = x * mx.eye(t, t)

        # (B, C, H, t, W, t)
        x = x.transpose(0, 1, 2, 4, 3, 5)

        # (B, C, t*H, t*W)
        return x.reshape(B, C, t * H, t * W)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.shape[1] % self.block_size == 0, "")
        _assert(x.shape[2] % self.block_size == 0, "")

        B, H, W, C = x.shape

        # (B, H, W, G)
        out = self.conv1(x)

        # (B, K, 1, G)
        rp = L.adaptive_max_pool2d(out, output_size=(self.block_size, 1))

        # (B, 1, K, G)
        cp = L.adaptive_max_pool2d(out, output_size=(1, self.block_size))

        # (B, 1, 1, K*K*G) -> (B, G, K, K)
        p = mx.sigmoid(
            self.conv_p(rp).reshape(B, self.groups, self.block_size, self.block_size)
        )

        # (B, 1, 1, K*K*G) -> (B, G, K, K)
        q = mx.sigmoid(
            self.conv_q(cp).reshape(B, self.groups, self.block_size, self.block_size)
        )

        # (B, G, K, K)
        p = p / p.sum(axis=3, keepdims=True)

        # (B, G, K, K)
        q = q / q.sum(axis=2, keepdims=True)

        # (B, G, K, K) -> (B, G, 1, K, K) -> (B, G, C/G, K, K)
        p = mx.broadcast_to(
            p.reshape(B, self.groups, 1, self.block_size, self.block_size),
            shape=(
                B,
                self.groups,
                C // self.groups,
                self.block_size,
                self.block_size,
            ),
        )

        # (B, G, C/G, K, K) -> (B, C, K, K)
        p = p.reshape(B, C, self.block_size, self.block_size)

        # (B, G, K, K) -> (B, G, 1, K, K) -> (B, G, C/G, K, K)
        q = mx.broadcast_to(
            q.reshape(B, self.groups, 1, self.block_size, self.block_size),
            shape=(
                B,
                self.groups,
                C // self.groups,
                self.block_size,
                self.block_size,
            ),
        )

        # (B, G, C/G, K, K) -> (B, C, K, K)
        q = q.reshape(B, C, self.block_size, self.block_size)

        # (B, C, H, H)
        p = self._resize_mat(p, H // self.block_size)

        # (B, C, W, W)
        q = self._resize_mat(q, W // self.block_size)

        print(p.shape, q.shape)

        # (B, C, H, H) @ (B, H, W, C) -> (B, C, H, W)
        y = p @ x.transpose(0, 3, 1, 2)

        # (B, C, H, W) @ (B, C, W, W) -> (B, C, H, W)
        y = y @ q

        # (B, H, W, C)
        y = self.conv2(y.transpose(0, 2, 3, 1))
        return y


class BatNonLocalAttn(nn.Module):
    """BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(
        self,
        in_channels: int,
        block_size: int = 7,
        groups: int = 2,
        rd_ratio: float = 0.25,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        drop_rate: float = 0.2,
        act_layer: str | Callable[[], nn.Module] | None = nn.ReLU,
        norm_layer: str | Callable[[int], nn.Module] | None = nn.BatchNorm,
        **_,
    ):
        super().__init__()

        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)

        self.conv1 = ConvNormAct(
            in_channels, rd_channels, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.ba = BilinearAttnTransform(
            rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer
        )
        self.conv2 = ConvNormAct(
            rd_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer
        )
        self.dropout = nn.Dropout2d(p=drop_rate)

    def __call__(self, x: mx.array) -> mx.array:
        xl = self.conv1(x)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x
