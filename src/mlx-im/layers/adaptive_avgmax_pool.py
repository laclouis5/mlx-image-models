from typing import Literal, Tuple, Union

import mlx.core as mx
from mlx import nn

from .format import FormatT, get_channel_dim, get_spatial_dim
from .mlx_layers import (
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    Flatten,
    adaptive_avg_pool2d,
    adaptive_max_pool2d,
)

_int_tuple_2_t = Union[int, Tuple[int, int]]


def adaptive_pool_feat_mult(pool_type: str = "avg") -> int:
    if pool_type.endswith("catavgmax"):
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x: mx.array, output_size: _int_tuple_2_t = 1) -> mx.array:
    b, h, w, c = x.shape
    ho, wo = output_size

    x = x.reshape(b, ho, h // ho, wo, w // wo, c)
    x_min = x.mean(axis=(2, 4))
    x_max = x.max(axis=(2, 4))

    return 0.5 * (x_min + x_max)


def adaptive_catavgmax_pool2d(x: mx.array, output_size: _int_tuple_2_t = 1) -> mx.array:
    b, h, w, c = x.shape
    ho, wo = output_size

    x = x.reshape(b, ho, h // ho, wo, w // wo, c)
    x_min = x.mean(axis=(2, 4))
    x_max = x.max(axis=(2, 4))

    return mx.concat([x_min, x_max], axis=3)


def select_adaptive_pool2d(
    x: mx.array,
    pool_type=Literal["avg", "avgmax", "catavgmax", "max"],
    output_size: _int_tuple_2_t = 1,
) -> mx.array:
    if pool_type == "avg":
        x = adaptive_avg_pool2d(x, output_size)
    elif pool_type == "avgmax":
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == "catavgmax":
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == "max":
        x = adaptive_max_pool2d(x, output_size)
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")
    return x


class FastAdaptiveAvgPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: FormatT = "NHWC"):
        super().__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def __call__(self, x: mx.array) -> mx.array:
        return x.mean(self.dim, keepdims=not self.flatten)


class FastAdaptiveMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = "NHWC"):
        super().__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def __call__(self, x: mx.array) -> mx.array:
        return x.max(self.dim, keepdims=not self.flatten)


class FastAdaptiveAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = "NHWC"):
        super().__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def __call__(self, x: mx.array) -> mx.array:
        x_avg = x.mean(self.dim, keepdims=not self.flatten)
        x_max = x.max(self.dim, keepdims=not self.flatten)
        return 0.5 * (x_avg + x_max)


class FastAdaptiveCatAvgMaxPool(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: str = "NHWC"):
        super().__init__()
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)

        if flatten:
            self.dim_cat = 3
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def __call__(self, x: mx.array) -> mx.array:
        x_avg = x.mean(self.dim_reduce, keepdims=not self.flatten)
        x_max = x.max(self.dim_reduce, keepdims=not self.flatten)
        return mx.concat([x_avg, x_max], axis=self.dim_cat)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    def __init__(
        self,
        output_size: _int_tuple_2_t = 1,
        pool_type: str = "fast",
        flatten: bool = False,
        input_fmt: str = "NHWC",
    ):
        super().__init__()
        assert input_fmt in ("NCHW", "NHWC")

        self.pool_type = pool_type or ""
        pool_type = pool_type.lower()

        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = Flatten(1) if flatten else nn.Identity()
        elif pool_type.startswith("fast") or input_fmt != "NCHW":
            assert (
                output_size == 1
            ), "Fast pooling and non NCHW input formats require output_size == 1."
            if pool_type.endswith("catavgmax"):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith("avgmax"):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith("max"):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type == "fast" or pool_type.endswith("avg"):
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            else:
                assert False, "Invalid pool type: %s" % pool_type
            self.flatten = nn.Identity()
        else:
            assert input_fmt == "NCHW"
            if pool_type == "avgmax":
                self.pool = AdaptiveAvgMaxPool2d(output_size)
            elif pool_type == "catavgmax":
                self.pool = AdaptiveCatAvgMaxPool2d(output_size)
            elif pool_type == "max":
                self.pool = AdaptiveMaxPool2d(output_size)
            elif pool_type == "avg":
                self.pool = AdaptiveAvgPool2d(output_size)
            else:
                raise AssertionError(f"Invalid pool type: {pool_type}")

            self.flatten = Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pool(x)
        return self.flatten(x)

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "pool_type="
            + self.pool_type
            + ", flatten="
            + str(self.flatten)
            + ")"
        )
