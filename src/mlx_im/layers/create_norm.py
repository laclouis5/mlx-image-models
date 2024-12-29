import functools
import types

from mlx import nn

from .norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d, RmsNorm, RmsNorm2d


class FrozenBatchNorm2d(nn.BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)
        self.freeze()
        self.eval()

    def unfreeze(self, *args, **kwargs):
        super().unfreeze(*args, **kwargs)
        self.freeze(
            keys=["weight", "bias", "running_mean", "running_var"], recurse=False
        )

    def train(self, mode=True):
        """This module is frozen and `training` is always false."""
        return super().train(False)


_NORM_MAP = dict(
    batchnorm=nn.BatchNorm,
    batchnorm2d=nn.BatchNorm,
    batchnorm1d=nn.BatchNorm,
    groupnorm=GroupNorm,
    groupnorm1=GroupNorm1,
    layernorm=LayerNorm,
    layernorm2d=LayerNorm2d,
    rmsnorm=RmsNorm,
    rmsnorm2d=RmsNorm2d,
    frozenbatchnorm2d=FrozenBatchNorm2d,
)
_NORM_TYPES = {m for n, m in _NORM_MAP.items()}


def create_norm_layer(layer_name: str, num_features: int, **kwargs) -> nn.Module:
    layer = get_norm_layer(layer_name)
    layer_instance = layer(num_features, **kwargs)
    return layer_instance


def get_norm_layer(norm_layer: str) -> nn.Module:
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    norm_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace("_", "").lower()
        norm_layer = _NORM_MAP[layer_name]
    else:
        norm_layer = norm_layer

    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)  # bind/rebind args
    return norm_layer
