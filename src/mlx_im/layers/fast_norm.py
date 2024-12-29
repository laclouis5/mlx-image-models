from typing import Optional

import mlx.core as mx

# fast (ie lower precision LN) can be disabled with this flag if issues crop up
_USE_FAST_NORM = False  # defaulting to False for now


# NOTE: No support for "fast norms" in MLX as they are already included.
def is_fast_norm():
    return False


def set_fast_norm(enable=True):
    global _USE_FAST_NORM
    _USE_FAST_NORM = enable


def fast_group_norm(
    x: mx.array,
    num_groups: int,
    weight: Optional[mx.array] = None,
    bias: Optional[mx.array] = None,
    eps: float = 1e-5,
) -> mx.array:
    # NOTE: Implementation from `MLX.nn.GroupNorm`.
    batch, *rest, dims = x.shape
    group_size = dims // num_groups

    # Split into groups
    x = x.reshape(batch, -1, num_groups, group_size)
    x = x.transpose(0, 2, 1, 3).reshape(batch, num_groups, -1)

    # Normalize
    x = mx.fast.layer_norm(x, eps=eps, weight=None, bias=None)

    x = x.reshape(batch, num_groups, -1, group_size)
    x = x.transpose(0, 2, 1, 3).reshape(batch, *rest, dims)

    return (weight * x + bias) if weight is not None else x


def fast_layer_norm(
    x: mx.array,
    normalized_shape: list[int],
    weight: Optional[mx.array] = None,
    bias: Optional[mx.array] = None,
    eps: float = 1e-5,
) -> mx.array:
    # NOTE: `normalized_shape` (`dims`) is not used my the MLX implementation.
    # It is probably deduced from the `weight` shape.
    return mx.fast.layer_norm(x, weight=weight, bias=bias, eps=eps)


def rms_norm(
    x: mx.array,
    normalized_shape: list[int],
    weight: Optional[mx.array] = None,
    eps: float = 1e-5,
):
    # NOTE: `normalized_shape` (`dims`) is not used my the MLX implementation.
    # It is probably deduced from the `weight` shape.
    return mx.fast.rms_norm(x, weight=weight, eps=eps)


def fast_rms_norm(
    x: mx.array,
    normalized_shape: list[int],
    weight: Optional[mx.array] = None,
    eps: float = 1e-5,
) -> mx.array:
    return rms_norm(x, normalized_shape, weight, eps)
