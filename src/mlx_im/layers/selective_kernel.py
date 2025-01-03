from typing import Callable

import mlx.core as mx
from mlx import nn

from .conv_bn_act import ConvNormAct
from .helpers import make_divisible
from .trace_utils import _assert


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelAttn(nn.Module):
    def __init__(
        self,
        channels: int,
        num_paths: int = 2,
        attn_channels: int = 32,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm,
    ):
        super().__init__()

        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer()
        self.fc_select = nn.Conv2d(
            attn_channels, channels * num_paths, kernel_size=1, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.shape[1] == self.num_paths, "")

        # (B, P, H, W, C) -> (B, H, W, C) -> (B, 1, 1, C)
        x = x.sum(axis=1).mean(axis=(1, 2), keepdims=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)

        B, H, W, C = x.shape

        # (B, H, W, P, C)
        x = x.reshape(B, H, W, self.num_paths, C // self.num_paths)
        x = x.transpose(0, 3, 1, 2, 4)

        return mx.softmax(x, axis=1)


class SelectiveKernel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int | list[int] | None = None,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        rd_ratio: float = 1.0 / 16,
        rd_channels: int | None = None,
        rd_divisor: int = 8,
        keep_3x3: bool = True,
        split_input: int = True,
        act_layer: str | Callable[[], nn.Module] | None = nn.ReLU,
        norm_layer: str | Callable[[int], nn.Module] | None = nn.BatchNorm,
        aa_layer=None,
        drop_layer=None,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        kernel_size = kernel_size if kernel_size is not None else [3, 5]
        _kernel_valid(kernel_size)

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)

        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input

        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths

        groups = min(out_channels, groups)

        conv_kwargs = dict(
            stride=stride,
            groups=groups,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_layer=drop_layer,
        )

        self.paths = [
            ConvNormAct(
                in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs
            )
            for k, d in zip(kernel_size, dilation)
        ]

        attn_channels = rd_channels or make_divisible(
            out_channels * rd_ratio, divisor=rd_divisor
        )

        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if self.split_input:
            # (B, H, W, C)
            x_split = mx.split(x, self.num_paths, axis=-1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            # (B, H, W, C)
            x_paths = [op(x) for op in self.paths]

        x = mx.stack(x_paths, axis=1)
        x_attn = self.attn(x)
        x = x * x_attn

        return mx.sum(x, axis=1)
