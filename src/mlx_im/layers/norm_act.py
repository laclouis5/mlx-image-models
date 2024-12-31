from typing import List, Type

import mlx.core as mx
from mlx import nn

from .create_act import create_act_layer
from .create_norm import FrozenBatchNorm2d
from .fast_norm import is_fast_norm


def _create_act(
    act_layer: str | Type[nn.Module] | None,
    act_kwargs: dict | None = None,
    inplace: bool = False,
    apply_act: bool = True,
) -> nn.Module:
    act_kwargs = act_kwargs or {}
    act_kwargs.setdefault("inplace", inplace)
    act = None

    if apply_act:
        act = create_act_layer(act_layer, **act_kwargs)

    return nn.Identity() if act is None else act


class BatchNormAct2d(nn.BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        apply_act: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
        inplace: bool = True,
        drop_layer: type[nn.Module] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = super().__call__(x)
        x = self.drop(x)
        return self.act(x)


class SyncBatchNormAct(BatchNormAct2d):
    pass


def convert_sync_batchnorm(module, process_group=None):
    raise NotImplementedError


class FrozenBatchNormAct2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        apply_act: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict | None = None,
        inplace: bool = True,
        drop_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = True

        self.weight = mx.ones(num_features)
        self.bias = mx.zeros(num_features)
        self.running_mean = mx.zeros(num_features)
        self.running_var = mx.ones(num_features)

        self.freeze(
            keys=["weight", "bias", "running_mean", "running_var"], recurse=False
        )

        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

    def unfreeze(self, *, recurse=True, keys=None, strict=False):
        super().unfreeze(recurse=recurse, keys=keys, strict=strict)
        self.freeze(
            keys=["weight", "bias", "running_mean", "running_var"], recurse=False
        )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        # TODO: Implement weights loading.
        raise NotImplementedError("TODO")

        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight.reshape(1, 1, 1, -1)
        b = self.bias.reshape(1, 1, 1, -1)
        rv = self.running_var.reshape(1, 1, 1, -1)
        rm = self.running_mean.reshape(1, 1, 1, -1)

        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        x = x * scale + bias
        x = self.act(self.drop(x))
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps}, act={self.act})"


def freeze_batch_norm_2d(module: nn.Module) -> nn.Module:
    # TODO: Make sur a **copy** of the weights are stored in the new module.
    res = module

    if isinstance(module, (BatchNormAct2d, SyncBatchNormAct)):
        res = FrozenBatchNormAct2d(module.num_features)

        res.num_features = module.num_features
        res.affine = hasattr(module, "weight")

        if res.affine:
            res.weight = mx.array(module.weight)
            res.bias = mx.array(module.bias)

        res.running_mean = mx.array(module.running_mean)
        res.running_var = mx.array(module.running_var)

        res.eps = module.eps
        res.drop = module.drop
        res.act = module.act
    elif isinstance(module, nn.BatchNorm):
        res = FrozenBatchNorm2d(module.num_features)

        res.num_features = module.num_features
        res.affine = hasattr(module, "weight")

        if res.affine:
            res.weight = mx.array(module.weight)
            res.bias = mx.array(module.bias)

        res.running_mean = mx.array(module.running_mean)
        res.running_var = mx.array(module.running_var)

        res.eps = module.eps
    else:
        # NOTE: MLX can store modules in containers such as `list`s and `dict`s, thus we have
        # to dig in those too.
        for name, child in module.children().items():
            if isinstance(child, list):
                new_child = [freeze_batch_norm_2d(c) for c in child]
                res.update_modules({name: new_child})
            elif isinstance(child, dict):
                new_child = {n: freeze_batch_norm_2d(c) for n, c in child.items()}
                res.update_modules({name: new_child})
            else:
                new_child = freeze_batch_norm_2d(child)
                if new_child is not child:
                    res.update_modules({name: new_child})

    return res


def unfreeze_batch_norm_2d(module: nn.Module) -> nn.Module:
    res = module
    if isinstance(module, FrozenBatchNormAct2d):
        res = BatchNormAct2d(module.num_features, eps=module.eps)

        if module.affine:
            res.weight = mx.array(module.weight)
            res.bias = mx.array(module.bias)

        res.running_mean = mx.array(module.running_mean)
        res.running_var = mx.array(module.running_var)

        res.drop = module.drop
        res.act = module.act
    elif isinstance(module, FrozenBatchNorm2d):
        res = nn.BatchNorm(module.num_features, eps=module.eps)

        if module.affine:
            res.weight = mx.array(module.weight)
            res.bias = mx.array(module.bias)

        res.running_mean = mx.array(module.running_mean)
        res.running_var = mx.array(module.running_var)
    else:
        # NOTE: MLX can store modules in containers such as `list`s and `dict`s, thus we have
        # to dig in those too.
        for name, child in module.children().items():
            if isinstance(child, list):
                new_child = [freeze_batch_norm_2d(c) for c in child]
                res.update_modules({name: new_child})
            elif isinstance(child, dict):
                new_child = {n: freeze_batch_norm_2d(c) for n, c in child.items()}
                res.update_modules({name: new_child})
            else:
                new_child = freeze_batch_norm_2d(child)
                if new_child is not child:
                    res.update_modules({name: new_child})
    return res


def _num_groups(num_channels: int, num_groups: int, group_size: int | None) -> int:
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups


class GroupNormAct(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine=True,
        group_size: int | None = None,
        apply_act: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict = None,
        inplace: bool = True,
        drop_layer: type[nn.Module] | None = None,
    ):
        super().__init__(
            _num_groups(num_channels, num_groups, group_size),
            dims=num_channels,
            eps=eps,
            affine=affine,
            pytorch_compatible=True,
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

        self._fast_norm = is_fast_norm()

    def __call__(self, x: mx.array) -> mx.array:
        x = super().__call__(x)
        x = self.drop(x)
        x = self.act(x)
        return x


class GroupNorm1Act(nn.GroupNorm):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        apply_act=True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict = None,
        inplace: bool = True,
        drop_layer: type[nn.Module] | None = None,
    ):
        super().__init__(
            num_groups=1,
            dims=num_channels,
            eps=eps,
            affine=affine,
            pytorch_compatible=True,
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

        self._fast_norm = is_fast_norm()

    def __call__(self, x: mx.array) -> mx.array:
        x = super().__call__(x)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct(nn.LayerNorm):
    def __init__(
        self,
        normalization_shape: int,
        eps: float = 1e-5,
        affine: bool = True,
        apply_act: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict = None,
        inplace: bool = True,
        drop_layer: type[nn.Module] | None = None,
    ):
        # NOTE: Why `normalization_shape` here?
        super().__init__(dims=normalization_shape, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )

        self._fast_norm = is_fast_norm()

    def __call__(self, x: mx.array) -> mx.array:
        x = super().__call__(x)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct2d(nn.LayerNorm):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        apply_act: bool = True,
        act_layer: str | type[nn.Module] | None = nn.ReLU,
        act_kwargs: dict = None,
        inplace: bool = True,
        drop_layer: type[nn.Module] | None = None,
    ):
        super().__init__(dims=num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(
            act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act
        )
        self._fast_norm = is_fast_norm()

    def __call__(self, x: mx.array) -> mx.array:
        x = super().__call__(x)
        x = self.drop(x)
        x = self.act(x)
        return x
