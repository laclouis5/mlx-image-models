from typing import Type

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer
from .trace_utils import _assert


def inv_instance_rms(x: mx.array, eps: float = 1e-5) -> mx.array:
    rms = (x.square().mean(axis=(1, 2), keepdims=True) + eps).rsqrt()
    return mx.broadcast_to(rms, shape=x.shape)


class FilterResponseNormTlu2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        apply_act: bool = True,
        eps: float = 1e-5,
        rms: bool = True,
        **_,
    ):
        super().__init__()
        self.apply_act = apply_act
        self.rms = rms
        self.eps = eps

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.tau = mx.zeros(num_features) if apply_act else None

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)

        return mx.maximum(x, self.tau.reshape(v_shape)) if self.tau is not None else x


class FilterResponseNormAct2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        apply_act: bool = True,
        act_layer: str | Type[nn.Module] | None = nn.ReLU,
        inplace: bool = None,
        rms: bool = True,
        eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer, inplace=inplace)
        else:
            self.act = nn.Identity()

        self.rms = rms
        self.eps = eps

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)

        return self.act(x)
