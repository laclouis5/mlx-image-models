from typing import Optional, Sequence, Tuple, Union

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.pooling import _Pool2d

from .helpers import _int_tuple_2_t, to_2tuple


class Flatten(nn.Module):
    def __init__(self, start_axis: int = 0, end_axis: int = -1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, start_axis=self.start_axis, end_axis=self.end_axis)


def adaptive_avg_pool2d(x: mx.array, output_size) -> mx.array:
    b, h, w, c = x.shape
    ho, wo = to_2tuple(output_size)
    x = x.reshape(b, ho, h // ho, wo, w // wo, c)
    return x.mean(axis=(2, 4))


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_avg_pool2d(x, output_size=self.output_size)


def adaptive_max_pool2d(x: mx.array, output_size: _int_tuple_2_t) -> mx.array:
    b, h, w, c = x.shape
    ho, wo = to_2tuple(output_size)
    x = x.reshape(b, ho, h // ho, wo, w // wo, c)
    return x.max(axis=(2, 4))


class AdaptiveMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_max_pool2d(x, output_size=self.output_size)


def _nanmean(
    v: mx.array,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    *,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    is_nan = mx.isnan(v)
    v = mx.nan_to_num(v)
    s = v.sum(axis=axis, keepdims=keepdims, stream=stream)
    c = (~is_nan).sum(axis=axis, keepdims=keepdims, stream=stream)
    return s / c


class AvgPool2d(_Pool2d):
    """AvgPool2d with support for `count_include_pad`."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        count_include_pad: bool = True,
    ):
        self._count_include_pad = count_include_pad
        need_padding = padding is not None and padding != 0 and padding != (0, 0)

        if self._count_include_pad or not need_padding:
            super().__init__(mx.mean, 0, kernel_size, stride, padding)
        else:
            super().__init__(_nanmean, mx.nan, kernel_size, stride, padding)
