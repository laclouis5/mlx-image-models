"""Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""

import collections.abc
from itertools import repeat
from typing import Tuple, Union

_int_tuple_2_t = Union[int, Tuple[int, int]]


# From PyTorch internals
def _ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(
    v: int, divisor: int = 8, min_value: int | None = None, round_limit: float = 0.9
) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    # pads a tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n
