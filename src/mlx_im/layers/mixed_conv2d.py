import mlx.core as mx
import numpy as np
from mlx import nn

from .conv2d_same import create_conv2d_pad
from .helpers import _int_tuple_2_t


def _split_channels(num_chan: int, num_groups: int) -> list[int]:
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] = 3,
        stride: _int_tuple_2_t = 1,
        padding: str | _int_tuple_2_t = "",
        dilation: _int_tuple_2_t = 1,
        depthwise: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.layers: list[nn.Module] = []

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for k, in_ch, out_ch in zip(kernel_size, in_splits, out_splits):
            conv_groups = in_ch if depthwise else 1
            self.layers.append(
                create_conv2d_pad(
                    in_ch,
                    out_ch,
                    k,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=conv_groups,
                    **kwargs,
                ),
            )

        self.splits = np.cumsum(in_splits).tolist()

    def __call__(self, x: mx.array) -> mx.array:
        x_split = mx.split(x, self.splits, axis=3)
        x_out = [c(s) for c, s in zip(self.layers, x_split)]
        return mx.concat(x_out, axis=3)
