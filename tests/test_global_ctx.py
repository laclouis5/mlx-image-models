import mlx.core as mx
import numpy as np
import pytest
import timm.layers.global_context as timm_m
import torch

import mlx_im.layers.global_context as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("fuse_add", [True, False])
@pytest.mark.parametrize("fuse_scale", [True, False])
def test_global_ctx(fuse_add, fuse_scale):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.GlobalContext(
        channels=128, fuse_add=fuse_add, fuse_scale=fuse_scale
    )
    mod_timm = timm_m.GlobalContext(
        channels=128, fuse_add=fuse_add, fuse_scale=fuse_scale
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
