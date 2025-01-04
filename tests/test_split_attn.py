import mlx.core as mx
import numpy as np
import pytest
import timm.layers.split_attn as timm_m
import torch

import mlx_im.layers.split_attn as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("radix", [1, 2])
@pytest.mark.parametrize("groups", [1, 4])
def test_radix_softmax(radix, groups):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 1, 1, radix * 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.RadixSoftmax(radix=radix, cardinality=groups)
    mod_timm = timm_m.RadixSoftmax(radix=radix, cardinality=groups)

    out_timm = mod_timm(x_torch).reshape(2, radix * 128, 1, 1)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("out_channels", [128, 64])
@pytest.mark.parametrize("radix", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 4])
def test_split_attn(out_channels, radix, stride, groups):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.SplitAttn(
        128, out_channels=out_channels, stride=stride, groups=groups, radix=radix
    )
    mod_timm = timm_m.SplitAttn(
        128, out_channels=out_channels, stride=stride, groups=groups, radix=radix
    )

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
