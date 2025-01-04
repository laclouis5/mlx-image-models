import mlx.core as mx
import numpy as np
import pytest
import timm.layers as timm_m
import torch

import mlx_im.layers as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize(
    "attn_type",
    [
        "se",
        "ese",
        "eca",
        "ecam",
        "ceca",
        "ge",
        "gc",
        "gca",
        "cbam",
        "lcbam",
        "sk",
        "splat",
        "lambda",
        "bottleneck",
        "halo",
        "nl",
        "bat",
        True,
    ],
)
def test_create_attn(attn_type):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    attn_kwarg = {}

    if attn_type == "bottleneck":
        attn_kwarg["feat_size"] = (32, 48)
    elif attn_type == "bat":
        attn_kwarg["block_size"] = 8
        attn_kwarg["drop_rate"] = 0.0

    # NOTE: The module has the same name as the exported function.
    mod_mlx = mlx_m.create_attn(attn_type=attn_type, channels=128, **attn_kwarg)
    mod_timm = timm_m.create_attn(attn_type=attn_type, channels=128, **attn_kwarg)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
