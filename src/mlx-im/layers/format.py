from enum import Enum
from typing import Union

import mlx.core as mx


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x: mx.array, fmt: Format) -> mx.array:
    if fmt == Format.NHWC:
        x = x.transpose(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(0, 2, 1)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def nhwc_to(x: mx.array, fmt: Format) -> mx.array:
    if fmt == Format.NCHW:
        x = x.transpose(0, 3, 1, 2)
    elif fmt == Format.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(1, 2).transpose(0, 2, 1)
    return x
