from functools import partial
from typing import Optional, Type

import mlx.core as mx
import numpy as np
from mlx import nn

from .padding import get_padding
from .typing import LayerType


class BlurPool2d(nn.Module):
    def __init__(
        self,
        channels: Optional[int] = None,
        filt_size: int = 3,
        stride: int = 2,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride

        # NOTE: This is not equivalent but may be sufficient.
        self.pad_mode = "edge" if pad_mode == "reflect" else pad_mode
        pad = get_padding(filt_size, stride, dilation=1)
        self.padding = ((0, 0), (pad, pad), (pad, pad), (0, 0))

        coeffs = mx.array(
            (np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32)
        )

        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, :, :, None]

        if channels is not None:
            blur_filter = mx.repeat(blur_filter, repeats=channels, axis=0)

        # NOTE: Underscored so the array is not considered as a parameter by MLX.
        self._filt = blur_filter

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, pad_width=self.padding, mode=self.pad_mode)
        print(x.squeeze())
        if self.channels is None:
            channels = x.shape[-1]
            weight = mx.broadcast_to(
                self._filt, shape=(channels, self.filt_size, self.filt_size, 1)
            )
        else:
            channels = self.channels
            weight = self._filt

        return mx.conv2d(x, weight, stride=self.stride, groups=channels)


def create_aa(
    aa_layer: LayerType,
    channels: Optional[int] = None,
    stride: int = 2,
    enable: bool = True,
    noop: Optional[Type[nn.Module]] = nn.Identity,
) -> nn.Module:
    if not aa_layer or not enable:
        return noop() if noop is not None else None

    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace("_", "").replace("-", "")
        if aa_layer == "avg" or aa_layer == "avgpool":
            aa_layer = nn.AvgPool2d
        elif aa_layer == "blur" or aa_layer == "blurpool":
            aa_layer = BlurPool2d
        elif aa_layer == "blurpc":
            aa_layer = partial(BlurPool2d, pad_mode="constant")

        else:
            assert False, f"Unknown anti-aliasing layer ({aa_layer})."
    try:
        return aa_layer(channels=channels, stride=stride)
    except TypeError:
        return aa_layer(stride)
