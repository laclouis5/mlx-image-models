import collections.abc
import math
import re
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, Iterator, Tuple, Type, Union

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

__all__ = [
    "model_parameters",
    "named_apply",
    "named_modules",
    "named_modules_with_params",
    "adapt_input_conv",
    "group_with_matcher",
    "group_modules",
    "group_parameters",
    "flatten_modules",
    "checkpoint_seq",
]


def model_parameters(
    model: nn.Module, exclude_head: bool = False, head_name: str = "classifier"
):
    parameters = model.parameters()

    if exclude_head:
        del parameters[head_name]

    return [v for _, v in tree_flatten(parameters)]


def named_apply(
    fn: Callable[[nn.Module, str], None],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)

    for child_name, child_module in module.named_modules()[1:]:
        child_name = ".".join((name, child_name)) if name else child_name

        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )

    if depth_first and include_root:
        fn(module=module, name=name)

    return module


def named_modules(
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
):
    if not depth_first and include_root:
        yield name, module

    # NOTE: We reverse to match the PyTorch module order.
    for child_name, child_module in reversed(module.named_modules()[1:]):
        child_name = ".".join((name, child_name)) if name else child_name

        yield from named_modules(
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )

    if depth_first and include_root:
        yield name, module


def named_modules_with_params(
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
):
    raise NotImplementedError


MATCH_PREV_GROUP = (99_999,)


def group_with_matcher(
    named_objects: Iterator[Tuple[str, Any]],
    group_matcher: Union[Dict, Callable],
    return_values: bool = False,
    reverse: bool = False,
):
    raise NotImplementedError


def group_parameters(
    module: nn.Module,
    group_matcher,
    return_values: bool = False,
    reverse: bool = False,
):
    raise NotImplementedError


def group_modules(
    module: nn.Module,
    group_matcher,
    return_values: bool = False,
    reverse: bool = False,
):
    raise NotImplementedError


def flatten_modules(
    named_modules: Iterator[Tuple[str, nn.Module]],
    depth: int = 1,
    prefix: Union[str, Tuple[str, ...]] = "",
    module_types: Union[str, Tuple[Type[nn.Module]]] = "sequential",
):
    prefix_is_tuple = isinstance(prefix, tuple)
    if isinstance(module_types, str):
        if module_types == "container":
            module_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        else:
            module_types = (nn.Sequential,)
    for name, module in named_modules:
        if depth and isinstance(module, module_types):
            yield from flatten_modules(
                module.named_children(),
                depth - 1,
                prefix=(name,) if prefix_is_tuple else name,
                module_types=module_types,
            )
        else:
            if prefix_is_tuple:
                name = prefix + (name,)
                yield name, module
            else:
                if prefix:
                    name = ".".join([prefix, name])
                yield name, module


def checkpoint_seq(
    functions, x, every=1, flatten=False, skip_last=False, preserve_rng_state=True
):
    raise NotImplementedError


def adapt_input_conv(in_chans: int, conv_weight: mx.array) -> mx.array:
    conv_type = conv_weight.dtype
    conv_weight = (
        conv_weight.float()
    )  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight
