import mlx.core as mx
import numpy as np
import pytest
import timm.layers.selective_kernel as timm_m
import torch

import mlx_im.layers.selective_kernel as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("out_channels", [128, 64])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("keep_3x3", [True, False])
@pytest.mark.parametrize("split_input", [True, False])
def test_lambda_layer(out_channels, stride, keep_3x3, split_input):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.SelectiveKernel(
        128,
        out_channels=out_channels,
        stride=stride,
        keep_3x3=keep_3x3,
        split_input=split_input,
    )
    mod_timm = timm_m.SelectiveKernel(
        128,
        out_channels=out_channels,
        stride=stride,
        keep_3x3=keep_3x3,
        split_input=split_input,
    )

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-4
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
