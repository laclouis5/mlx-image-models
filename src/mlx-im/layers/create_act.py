from typing import Callable, Type, Union

from mlx import nn

from .activations import (
    GELUTanh,
    HardMish,
    HardSigmoid,
    QuickGELU,
    hard_mish,
    hard_sigmoid,
)

_ACT_FN_DEFAULT = dict(
    silu=nn.silu,
    swish=nn.silu,
    mish=nn.mish,
    relu=nn.relu,
    relu6=nn.relu6,
    leaky_relu=nn.leaky_relu,
    elu=nn.elu,
    celu=nn.celu,
    selu=nn.selu,
    gelu=nn.gelu,
    gelu_tanh=nn.gelu_approx,
    quick_gelu=nn.gelu_fast_approx,
    sigmoid=nn.sigmoid,
    tanh=nn.tanh,
    hard_sigmoid=hard_sigmoid,
    hard_swish=nn.hardswish,
    hard_mish=hard_mish,
)

_ACT_FNS = (_ACT_FN_DEFAULT,)

for a in _ACT_FNS:
    a.setdefault("hardsigmoid", a.get("hard_sigmoid"))
    a.setdefault("hardswish", a.get("hard_swish"))


_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=nn.Hardswish,
    hard_mish=HardMish,
    identity=nn.Identity,
)

_ACT_LAYERS = (_ACT_LAYER_DEFAULT,)

for a in _ACT_LAYERS:
    a.setdefault("hardsigmoid", a.get("hard_sigmoid"))
    a.setdefault("hardswish", a.get("hard_swish"))


def get_act_fn(name: Union[Callable, str] = "relu"):
    if not name:
        return None
    if isinstance(name, Callable):
        return name
    name = name.lower()
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name: Union[Type[nn.Module], str] = "relu"):
    """Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not name:
        return None
    name = name.lower()
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name: Union[Type[nn.Module], str], inplace=None, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        # recover if act layer doesn't have inplace arg
        return act_layer(**kwargs)
