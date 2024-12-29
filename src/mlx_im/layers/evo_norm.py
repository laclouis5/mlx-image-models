from typing import Sequence, Type, Union

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer
from .trace_utils import _assert


def instance_std(x: mx.array, eps: float = 1e-5) -> mx.array:
    std = mx.sqrt(x.var(axis=(1, 2), keepdims=True) + eps)
    return mx.broadcast_to(std, shape=x.shape)


def instance_std_tpu(x: mx.array, eps: float = 1e-5) -> mx.array:
    std = mx.sqrt(manual_var(x, dim=(1, 2)) + eps)
    return mx.broadcast_to(std, shape=x.shape)


def instance_rms(x: mx.array, eps: float = 1e-5) -> mx.array:
    rms = mx.sqrt(x.square().mean(axis=(1, 2), keepdims=True) + eps)
    return mx.broadcast_to(rms, shape=x.shape)


def manual_var(x: mx.array, dim: Union[int, Sequence[int]], diff_sqm: bool = False):
    # NOTE: Ignore TPU
    return mx.var(x, axis=dim)


def group_std(
    x: mx.array, groups: int = 32, eps: float = 1e-5, flatten: bool = False
) -> mx.array:
    B, H, W, C = x.shape
    _assert(C % groups == 0, "")

    if flatten:
        x = x.reshape(B, groups, -1)
        std = mx.sqrt(x.var(axis=2, keepdims=True) + eps)
    else:
        x = x.reshape(B, H, W, groups, C // groups)
        std = mx.sqrt(x.var(axis=(1, 2, 4), keepdims=True) + eps)

    return mx.broadcast_to(std, shape=x.shape).reshape(B, H, W, C)


def group_std_tpu(
    x,
    groups: int = 32,
    eps: float = 1e-5,
    diff_sqm: bool = False,
    flatten: bool = False,
):
    # This is a workaround for some stability / odd behaviour of .var and .std
    # running on PyTorch XLA w/ TPUs. These manual var impl are producing much better results
    B, C, H, W = x.shape
    _assert(C % groups == 0, "")
    if flatten:
        x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
        var = manual_var(x, dim=-1, diff_sqm=diff_sqm)
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        var = manual_var(x, dim=(2, 3, 4), diff_sqm=diff_sqm)
    return var.add(eps).sqrt().expand(x.shape).reshape(B, C, H, W)


def group_rms(x: mx.array, groups: int = 32, eps: float = 1e-5) -> mx.array:
    B, H, W, C = x.shape
    _assert(C % groups == 0, "")

    x = x.reshape(B, H, W, groups, C // groups)
    rms = mx.sqrt(mx.mean(x.square(), axis=(1, 2, 4), keepdims=True) + eps)
    return mx.broadcast_to(rms, shape=x.shape).reshape(B, H, W, C)


class EvoNorm2dB0(nn.Module):
    def __init__(
        self,
        num_features: int,
        apply_act: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-3,
        **_,
    ):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.v = mx.ones(num_features) if apply_act else None

        self.running_var = mx.ones(num_features)
        self.freeze(keys=["running_var"], recurse=False)

    def unfreeze(self, *, recurse=True, keys=None, strict=False):
        super().unfreeze(recurse=recurse, keys=keys, strict=strict)
        self.freeze(keys=["running_var"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        if self.v is not None:
            if self.training:
                var = x.var(axis=(0, 1, 2))
                n = x.size / x.shape[-1]

                self.running_var = self.running_var * (
                    1.0 - self.momentum
                ) + mx.stop_gradient(var) * self.momentum * (n / (n - 1))

            else:
                var = self.running_var

            left = mx.broadcast_to(mx.sqrt(var + self.eps).reshape(v_shape), x.shape)
            v = self.v.reshape(v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / mx.maximum(left, right)

        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dB1(nn.Module):
    def __init__(
        self,
        num_features: int,
        apply_act: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)

        self.running_var = mx.ones(num_features)
        self.freeze(keys=["running_var"], recurse=False)

    def unfreeze(self, *, recurse=True, keys=None, strict=False):
        super().unfreeze(recurse=recurse, keys=keys, strict=strict)
        self.freeze(keys=["running_var"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        if self.apply_act:
            if self.training:
                var = x.var(axis=(0, 1, 2))
                n = x.size / x.shape[-1]

                self.running_var = self.running_var * (
                    1.0 - self.momentum
                ) + mx.stop_gradient(var) * self.momentum * (n / (n - 1))
            else:
                var = self.running_var

            var = var.reshape(v_shape)
            left = mx.sqrt(var + self.eps)
            right = (x + 1) * instance_rms(x, self.eps)
            x = x / mx.maximum(left, right)

        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dB2(nn.Module):
    def __init__(
        self,
        num_features: int,
        apply_act: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)

        self.running_var = mx.ones(num_features)
        self.freeze(keys=["running_var"], recurse=False)

    def unfreeze(self, *, recurse=True, keys=None, strict=False):
        super().unfreeze(recurse=recurse, keys=keys, strict=strict)
        self.freeze(keys=["running_var"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        if self.apply_act:
            if self.training:
                var = x.var(axis=(0, 1, 2))
                n = x.size / x.shape[-1]

                self.running_var = self.running_var * (
                    1.0 - self.momentum
                ) + mx.stop_gradient(var) * self.momentum * (n / (n - 1))
            else:
                var = self.running_var

            var = var.reshape(v_shape)
            left = mx.sqrt(var + self.eps)
            right = instance_rms(x, self.eps) - x
            x = x / mx.maximum(left, right)

        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dS0(nn.Module):
    def __init__(
        self,
        num_features: int,
        groups: int = 32,
        group_size: int | None = None,
        apply_act: bool = True,
        eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        self.apply_act = apply_act
        self.eps = eps

        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.v = mx.ones(num_features) if apply_act else None

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        if self.v is not None:
            v = self.v.reshape(v_shape)
            x = x * mx.sigmoid(x * v) / group_std(x, self.groups, self.eps)
        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dS0a(EvoNorm2dS0):
    def __init__(
        self,
        num_features: int,
        groups: int = 32,
        group_size: int | None = None,
        apply_act: bool = True,
        eps: float = 1e-3,
        **_,
    ):
        super().__init__(
            num_features,
            groups=groups,
            group_size=group_size,
            apply_act=apply_act,
            eps=eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        d = group_std(x, self.groups, self.eps)

        if self.v is not None:
            v = self.v.reshape(v_shape)
            x = x * mx.sigmoid(x * v)

        x = x / d

        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dS1(nn.Module):
    def __init__(
        self,
        num_features: int,
        groups: int = 32,
        group_size: int | None = None,
        apply_act: bool = True,
        act_layer: str | Type[nn.Module] | None = None,
        eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        self.apply_act = apply_act

        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()

        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups

        self.eps = eps
        self.pre_act_norm = False

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        if self.apply_act:
            x = self.act(x) / group_std(x, self.groups, self.eps)

        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dS1a(EvoNorm2dS1):
    def __init__(
        self,
        num_features: int,
        groups: int = 32,
        group_size: int | None = None,
        apply_act: bool = True,
        act_layer: str | Type[nn.Module] | None = None,
        eps=1e-3,
        **_,
    ):
        super().__init__(
            num_features,
            groups=groups,
            group_size=group_size,
            apply_act=apply_act,
            act_layer=act_layer,
            eps=eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)
        x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dS2(nn.Module):
    def __init__(
        self,
        num_features: int,
        groups: int = 32,
        group_size: int | None = None,
        apply_act: bool = True,
        act_layer: str | Type[nn.Module] | None = None,
        eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        self.apply_act = apply_act
        self.eps = eps

        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()

        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        if self.apply_act:
            x = self.act(x) / group_rms(x, self.groups, self.eps)

        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)


class EvoNorm2dS2a(EvoNorm2dS2):
    def __init__(
        self,
        num_features: int,
        groups: int = 32,
        group_size: int | None = None,
        apply_act: bool = True,
        act_layer: str | Type[nn.Module] | None = None,
        eps: float = 1e-3,
        **_,
    ):
        super().__init__(
            num_features,
            groups=groups,
            group_size=group_size,
            apply_act=apply_act,
            act_layer=act_layer,
            eps=eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        _assert(x.ndim == 4, "expected 4D input")
        v_shape = (1, 1, 1, -1)

        x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.reshape(v_shape) + self.bias.reshape(v_shape)
