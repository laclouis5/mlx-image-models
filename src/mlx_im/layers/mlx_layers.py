import operator
from itertools import accumulate
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


def _non_overlapping_sliding_windows(
    x: mx.array, shape: tuple[int, ...], window_shape: tuple[int, ...]
) -> mx.array:
    # Compute the intermediate shape
    new_shape = [shape[0]]
    for s, w in zip(shape[1:], window_shape):
        new_shape.append(s // w)
        new_shape.append(w)
    new_shape.append(shape[-1])

    last_axis = len(new_shape) - 1
    axis_order = [0, *range(1, last_axis, 2), *range(2, last_axis, 2), last_axis]

    x = x.reshape(new_shape)
    x = x.transpose(axis_order)
    return x


def sliding_windows(
    x: mx.array, window_shape: tuple[int, ...], window_strides: tuple[int, ...]
) -> mx.array:
    if x.ndim < 3:
        raise ValueError(
            f"To extract sliding windows at least 1 spatial dimension "
            f"(3 total) is needed but the input only has {x.ndim} dimensions."
        )

    spatial_dims = x.shape[1:-1]
    if not (len(spatial_dims) == len(window_shape) == len(window_strides)):
        raise ValueError(
            f"To extract sliding windows the window shapes and strides must have "
            f"the same number of spatial dimensions as the signal but the signal "
            f"has {len(spatial_dims)} dims and the window shape has {len(window_shape)} "
            f"and strides have {len(window_strides)}."
        )

    shape = x.shape
    if all(
        window == stride and size % window == 0
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ):
        return _non_overlapping_sliding_windows(x, shape, window_shape)

    strides = list(reversed(list(accumulate(reversed(shape + (1,)), operator.mul))))[1:]

    # Compute the output shape
    final_shape = [shape[0]]
    final_shape += [
        (size - window) // stride + 1
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ]
    final_shape += window_shape
    final_shape += [shape[-1]]

    # Compute the output strides
    final_strides = strides[:1]
    final_strides += [
        og_stride * stride for og_stride, stride in zip(strides[1:-1], window_strides)
    ]
    final_strides += strides[1:-1]
    final_strides += strides[-1:]  # should always be [1]

    return mx.as_strided(x, final_shape, final_strides)
