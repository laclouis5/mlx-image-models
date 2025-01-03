from typing import Tuple

import mlx.core as mx


def ndgrid(*tensors: mx.array) -> Tuple[mx.array, ...]:
    return mx.meshgrid(*tensors, indexing="ij")


def meshgrid(*tensors: mx.array) -> Tuple[mx.array, ...]:
    return mx.meshgrid(*tensors, indexing="xy")
