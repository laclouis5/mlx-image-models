from typing import Optional, Tuple

import mlx.core as mx
from mlx import nn

from .config import is_exportable, is_scriptable
from .helpers import _int_tuple_2_t, to_2tuple
from .padding import get_padding_value, pad_same

_USE_EXPORT_CONV = False


def conv2d_same(
    x,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> mx.array:
    x = pad_same(x, weight.shape[1:3], stride, dilation)
    x = mx.conv2d(
        x, weight, stride=stride, padding=(0, 0), dilation=dilation, groups=groups
    )

    if bias is not None:
        return x + bias

    return x


class Conv2dSame(nn.Conv2d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _int_tuple_2_t,
        stride: _int_tuple_2_t = 1,
        padding: _int_tuple_2_t = 0,
        dilation: _int_tuple_2_t = 1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            to_2tuple(kernel_size),
            to_2tuple(stride),
            to_2tuple(0),
            to_2tuple(dilation),
            groups,
            bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return conv2d_same(
            x,
            self.weight,
            self.bias if hasattr(self, "bias") else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2dSameExport(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        raise NotImplementedError

    def forward(self, x: mx.array) -> mx.array:
        raise NotImplementedError


def create_conv2d_pad(in_chs: int, out_chs: int, kernel_size: _int_tuple_2_t, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if _USE_EXPORT_CONV and is_exportable():
            # older PyTorch ver needed this to export same padding reasonably
            assert not is_scriptable()  # Conv2DSameExport does not work with jit
            return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        else:
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
