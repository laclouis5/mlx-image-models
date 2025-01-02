from typing import List

import mlx.core as mx
from mlx import nn

from .helpers import to_2tuple
from .padding import get_padding_value, pad_same


def avg_pool2d_same(
    x: mx.array,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> mx.array:
    assert not ceil_mode, "`count_include_pad` not supported."
    assert count_include_pad, "`count_include_pad` not supported."
    # FIXME: how to deal with count_include_pad vs not for external padding?
    return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=(0, 0))(x)


class AvgPool2dSame(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size: int,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        assert not ceil_mode, "`count_include_pad` not supported."
        assert count_include_pad, "`count_include_pad` not supported."
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        super().__init__(kernel_size, stride, (0, 0))

    def __call__(self, x: mx.array) -> mx.array:
        x = pad_same(x, self.kernel_size, self.stride)
        return super().__call__(x)


def max_pool2d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    dilation: List[int] = (1, 1),
    ceil_mode: bool = False,
):
    assert not ceil_mode, "`count_include_pad` not supported."
    assert dilation == (1, 1), "`dilation` other than 1 not supported."
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return nn.MaxPool2d(kernel_size, stride, (0, 0))


class MaxPool2dSame(nn.MaxPool2d):
    def __init__(
        self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False
    ):
        assert not ceil_mode, "`count_include_pad` not supported."
        assert dilation == 1, "`dilation` other than 1 not supported."

        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        super().__init__(kernel_size, stride, (0, 0))

    def __call__(self, x: mx.array) -> mx.array:
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return super().__call__(x)


def create_pool2d(
    pool_type: str, kernel_size: List[int], stride=None, **kwargs
) -> nn.Module:
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
        padding, kernel_size, stride=stride, **kwargs
    )
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
