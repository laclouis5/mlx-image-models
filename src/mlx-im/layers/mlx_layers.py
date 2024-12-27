from typing import Tuple, Union

import mlx.core as mx
from mlx import nn

from .helpers import to_2tuple

_int_tuple_2_t = Union[int, Tuple[int, int]]


class Flatten(nn.Module):
    def __init__(self, start_axis: int = 0, end_axis: int = -1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, start_axis=self.start_axis, end_axis=self.end_axis)


def adaptive_avg_pool2d(x: mx.array, output_size) -> mx.array:
    b, h, w, c = x.shape
    ho, wo = output_size
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
