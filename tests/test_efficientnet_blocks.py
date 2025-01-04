import mlx.core as mx
import numpy as np
import pytest
import timm.models._efficientnet_blocks as timm_m
import torch

import mlx_im.models._efficientnet_blocks as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize(
    "block_name",
    [
        "SqueezeExcite",
        "ConvBnAct",
        "DepthwiseSeparableConv",
        "InvertedResidual",
        "LayerScale2d",
        "UniversalInvertedResidual",
        "MobileAttention",
        "CondConvResidual",
        "EdgeResidual",
    ],
)
def test_blocks(block_name):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    args = [128]

    if block_name not in ["LayerScale2d", "SqueezeExcite"]:
        args.append(64)  # Out channels

    if block_name == "ConvBnAct":
        args.append(3)  # Kernel size

    kwargs = {}

    if block_name == "CondConvResidual":
        kwargs["num_experts"] = 4

    mod_mlx = getattr(mlx_m, block_name)(*args, **kwargs)
    mod_timm = getattr(timm_m, block_name)(*args, **kwargs)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-4
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
