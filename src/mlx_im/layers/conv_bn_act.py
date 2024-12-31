from typing import Any, Dict, Optional, Type

import mlx.core as mx
from mlx import nn

from .blur_pool import create_aa
from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer
from .typing import LayerType, PadType


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: PadType = "",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_norm: bool = True,
        apply_act: bool = True,
        norm_layer: LayerType = nn.BatchNorm,
        act_layer: Optional[LayerType] = nn.ReLU,
        aa_layer: Optional[LayerType] = None,
        drop_layer: Optional[Type[nn.Module]] = None,
        conv_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        use_aa = aa_layer is not None and stride > 1

        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1 if use_aa else stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **conv_kwargs,
        )

        if apply_norm:
            # NOTE for backwards compatibility with models that use separate norm and act layer definitions
            norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
            # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
            if drop_layer:
                norm_kwargs["drop_layer"] = drop_layer
            self.bn = norm_act_layer(
                out_channels,
                apply_act=apply_act,
                act_kwargs=act_kwargs,
                **norm_kwargs,
            )
        else:
            # NOTE: This will likely cause issues during weight loading. In MLX, layers
            # in `Sequential` have integer names and cannot be assigned str names.
            # This results in a path name of `bn.layers.0` instead of `bn.drop`. The later
            # could be achieved in MLX `self.bn = {"drop": layer}` and then inplementing the sequential behavior manually, but there is no `Sequential` module in this setup and
            # this would also cause issues for weights loading.
            layers = []
            if drop_layer:
                norm_kwargs["drop_layer"] = drop_layer
                layers.append(drop_layer())
            self.bn = nn.Sequential(*layers)

        self.aa = create_aa(
            aa_layer, out_channels, stride=stride, enable=use_aa, noop=None
        )

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        aa = getattr(self, "aa", None)
        if aa is not None:
            x = self.aa(x)
        return x


ConvBnAct = ConvNormAct
ConvNormActAa = ConvNormAct  # backwards compat, when they were separate
