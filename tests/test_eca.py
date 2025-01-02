import mlx.core as mx
import numpy as np
import pytest
import timm.layers.eca as timm_m
import torch

import mlx_im.layers.eca as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("use_mlp", [True, False])
def test_eca(use_mlp):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.EcaModule(channels=128, use_mlp=use_mlp)
    mod_timm = timm_m.EcaModule(channels=128, use_mlp=use_mlp)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


def test_ceca():
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.CecaModule(channels=128)
    mod_timm = timm_m.CecaModule(channels=128)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
